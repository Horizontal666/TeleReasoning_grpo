#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path


DEFAULT_PREVIEW_CHARS = 120
MAIN_COMPARISON_METRICS = ("group_mean_mean", "group_std_mean", "best_of_n_mean")


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_or_none(values):
    return statistics.mean(values) if values else None


def pstdev_or_zero(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def fraction_matching(values, predicate):
    if not values:
        return None
    return sum(1 for value in values if predicate(value)) / len(values)


def extract_prompt_preview(text, max_chars):
    if not text:
        return ""

    marker = "\nuser\n"
    assistant_marker = "\nassistant\n"
    if marker in text:
        start = text.index(marker) + len(marker)
        end = text.find(assistant_marker, start)
        if end == -1:
            preview = text[start:]
        else:
            preview = text[start:end]
    else:
        preview = text

    preview = " ".join(preview.split())
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3] + "..."


def prompt_group_components(record, max_preview_chars):
    input_text = str(record.get("input", ""))
    ground_truth = record.get("gts")
    canonical = json.dumps(
        {"input": input_text, "gts": ground_truth},
        ensure_ascii=False,
        sort_keys=True,
    )
    prompt_hash = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    preview = extract_prompt_preview(input_text, max_preview_chars)
    return f"prompt:{prompt_hash}", prompt_hash, preview


def parse_run_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Invalid --run value: {value!r}. Expected label=/abs/path")
    label, path_str = value.split("=", 1)
    label = label.strip()
    path = Path(path_str.strip()).expanduser()
    if not label:
        raise argparse.ArgumentTypeError(f"Invalid --run value: {value!r}. Empty label.")
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"Run path must be absolute: {path}")
    return label, path


def list_step_files(rollout_dir):
    files = []
    for path in rollout_dir.glob("*.jsonl"):
        try:
            step = int(path.stem)
        except ValueError:
            continue
        files.append((step, path))
    return [path for _, path in sorted(files)]


def load_records(step_file):
    with step_file.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_step_number(file_step, records):
    warnings = []
    record_steps = []
    for record in records:
        step_value = record.get("step")
        if step_value is None:
            continue
        try:
            record_steps.append(int(step_value))
        except (TypeError, ValueError):
            warnings.append(f"Non-integer record step ignored: {step_value!r}")

    if not record_steps:
        return file_step, warnings

    unique_steps = sorted(set(record_steps))
    step = unique_steps[0]
    if len(unique_steps) > 1:
        warnings.append(f"Multiple record step values {unique_steps}; using {step}")
    if step != file_step:
        warnings.append(f"Filename step {file_step} differs from record step {step}; using record step")
    return step, warnings


def resolve_group_mode(records, requested_mode):
    if requested_mode == "uid":
        if any(record.get("uid") in (None, "") for record in records):
            raise ValueError("Requested uid grouping, but at least one record is missing uid.")
        return "uid"
    if requested_mode == "prompt":
        return "prompt"
    if all(record.get("uid") not in (None, "") for record in records):
        return "uid"
    return "prompt"


def build_group_rows(records, run_label, step, group_mode, expected_group_size, max_preview_chars):
    grouped = defaultdict(list)
    group_meta = {}

    for index, record in enumerate(records):
        score = safe_float(record.get("score"))
        if score is None:
            raise ValueError(f"Missing numeric score in step {step}, record index {index}.")

        acc = safe_float(record.get("acc"))
        if acc is None and score in (0.0, 1.0):
            acc = score

        input_text = str(record.get("input", ""))
        if group_mode == "uid":
            uid = str(record.get("uid"))
            group_key = f"uid:{uid}"
            prompt_hash = hashlib.sha1(input_text.encode("utf-8")).hexdigest()[:12]
            prompt_preview = extract_prompt_preview(input_text, max_preview_chars)
        else:
            group_key, prompt_hash, prompt_preview = prompt_group_components(record, max_preview_chars)

        grouped[group_key].append({"score": score, "acc": acc})
        group_meta[group_key] = {
            "prompt_hash": prompt_hash,
            "prompt_preview": prompt_preview,
        }

    group_rows = []
    for group_key, items in grouped.items():
        scores = [item["score"] for item in items]
        accs = [item["acc"] for item in items if item["acc"] is not None]
        group_size = len(items)
        is_ambiguous = group_mode == "prompt" and group_size != expected_group_size
        if is_ambiguous and group_size > expected_group_size and group_size % expected_group_size == 0:
            ambiguity_reason = "merged_duplicate_prompt"
        elif is_ambiguous:
            ambiguity_reason = "unexpected_group_size"
        else:
            ambiguity_reason = ""

        group_rows.append(
            {
                "run_label": run_label,
                "step": step,
                "group_key": group_key,
                "group_mode": group_mode,
                "group_size": group_size,
                "group_score_mean": mean_or_none(scores),
                "group_score_std": pstdev_or_zero(scores),
                "group_score_min": min(scores),
                "group_score_max": max(scores),
                "group_acc_mean": mean_or_none(accs),
                "is_ambiguous": is_ambiguous,
                "ambiguity_reason": ambiguity_reason,
                "prompt_hash": group_meta[group_key]["prompt_hash"],
                "prompt_preview": group_meta[group_key]["prompt_preview"],
            }
        )

    return group_rows


def build_step_row(run_label, step_file, step, records, group_mode, expected_group_size, group_rows, warnings):
    scores = [safe_float(record.get("score")) for record in records]
    scores = [value for value in scores if value is not None]
    accs = [safe_float(record.get("acc")) for record in records]
    accs = [value for value in accs if value is not None]
    group_means = [row["group_score_mean"] for row in group_rows if row["group_score_mean"] is not None]
    group_stds = [row["group_score_std"] for row in group_rows if row["group_score_std"] is not None]
    group_bests = [row["group_score_max"] for row in group_rows]
    group_worsts = [row["group_score_min"] for row in group_rows]
    group_acc_means = [row["group_acc_mean"] for row in group_rows if row["group_acc_mean"] is not None]

    return {
        "run_label": run_label,
        "step": step,
        "file_name": step_file.name,
        "group_mode": group_mode,
        "exact_grouping": group_mode == "uid",
        "num_records": len(records),
        "num_groups": len(group_rows),
        "expected_group_size": expected_group_size,
        "ambiguous_group_count": sum(1 for row in group_rows if row["is_ambiguous"]),
        "score_mean": mean_or_none(scores),
        "score_std": pstdev_or_zero(scores) if scores else None,
        "acc_mean": mean_or_none(accs),
        "acc_std": pstdev_or_zero(accs) if accs else None,
        "group_mean_mean": mean_or_none(group_means),
        "group_mean_std": pstdev_or_zero(group_means) if group_means else None,
        "group_std_mean": mean_or_none(group_stds),
        "group_std_median": statistics.median(group_stds) if group_stds else None,
        "group_std_max": max(group_stds) if group_stds else None,
        "best_of_n_mean": mean_or_none(group_bests),
        "worst_of_n_mean": mean_or_none(group_worsts),
        "all_correct_group_rate": fraction_matching(group_acc_means, lambda value: math.isclose(value, 1.0)),
        "all_wrong_group_rate": fraction_matching(group_acc_means, lambda value: math.isclose(value, 0.0)),
        "mixed_group_rate": fraction_matching(group_acc_means, lambda value: 0.0 < value < 1.0),
        "warnings": " | ".join(warnings),
    }


def analyze_step_file(step_file, run_label, expected_group_size, requested_group_mode, max_preview_chars):
    records = load_records(step_file)
    file_step = int(step_file.stem)
    step, warnings = resolve_step_number(file_step, records)
    group_mode = resolve_group_mode(records, requested_group_mode)
    group_rows = build_group_rows(
        records=records,
        run_label=run_label,
        step=step,
        group_mode=group_mode,
        expected_group_size=expected_group_size,
        max_preview_chars=max_preview_chars,
    )
    step_row = build_step_row(
        run_label=run_label,
        step_file=step_file,
        step=step,
        records=records,
        group_mode=group_mode,
        expected_group_size=expected_group_size,
        group_rows=group_rows,
        warnings=warnings,
    )
    return step_row, group_rows


def summarize_run(run_label, rollout_dir, step_rows, group_rows):
    def average_metric(metric_name):
        values = [row[metric_name] for row in step_rows if row[metric_name] is not None]
        return mean_or_none(values)

    group_mode_counts = defaultdict(int)
    for row in step_rows:
        group_mode_counts[row["group_mode"]] += 1

    warning_steps = [row["step"] for row in step_rows if row["warnings"]]
    return {
        "run_label": run_label,
        "rollout_dir": str(rollout_dir),
        "num_steps": len(step_rows),
        "step_range": [min(row["step"] for row in step_rows), max(row["step"] for row in step_rows)] if step_rows else None,
        "group_mode_counts": dict(sorted(group_mode_counts.items())),
        "exact_grouping_step_count": sum(1 for row in step_rows if row["exact_grouping"]),
        "total_records": sum(row["num_records"] for row in step_rows),
        "total_groups": len(group_rows),
        "total_ambiguous_groups": sum(1 for row in group_rows if row["is_ambiguous"]),
        "mean_score_mean": average_metric("score_mean"),
        "mean_acc_mean": average_metric("acc_mean"),
        "mean_group_mean_mean": average_metric("group_mean_mean"),
        "mean_group_std_mean": average_metric("group_std_mean"),
        "mean_best_of_n_mean": average_metric("best_of_n_mean"),
        "mean_worst_of_n_mean": average_metric("worst_of_n_mean"),
        "mean_all_correct_group_rate": average_metric("all_correct_group_rate"),
        "mean_all_wrong_group_rate": average_metric("all_wrong_group_rate"),
        "mean_mixed_group_rate": average_metric("mixed_group_rate"),
        "warning_step_count": len(warning_steps),
        "warning_steps": warning_steps[:20],
    }


def compare_two_runs(left_summary, right_summary, left_steps, right_steps):
    left_by_step = {row["step"]: row for row in left_steps}
    right_by_step = {row["step"]: row for row in right_steps}
    common_steps = sorted(set(left_by_step) & set(right_by_step))

    metric_delta_means = {}
    per_step_deltas = []
    for step in common_steps:
        row = {
            "step": step,
            "left_score_mean": left_by_step[step]["score_mean"],
            "right_score_mean": right_by_step[step]["score_mean"],
        }
        for metric in MAIN_COMPARISON_METRICS:
            left_value = left_by_step[step][metric]
            right_value = right_by_step[step][metric]
            if left_value is None or right_value is None:
                row[f"delta_{metric}"] = None
            else:
                row[f"delta_{metric}"] = right_value - left_value
        per_step_deltas.append(row)

    for metric in MAIN_COMPARISON_METRICS:
        values = [row[f"delta_{metric}"] for row in per_step_deltas if row[f"delta_{metric}"] is not None]
        metric_delta_means[metric] = mean_or_none(values)

    top_step_changes = {}
    for metric in MAIN_COMPARISON_METRICS:
        deltas = [row for row in per_step_deltas if row[f"delta_{metric}"] is not None]
        if metric == "group_std_mean":
            improvements = sorted(deltas, key=lambda row: row[f"delta_{metric}"])[:5]
            deteriorations = sorted(deltas, key=lambda row: row[f"delta_{metric}"], reverse=True)[:5]
        else:
            improvements = sorted(deltas, key=lambda row: row[f"delta_{metric}"], reverse=True)[:5]
            deteriorations = sorted(deltas, key=lambda row: row[f"delta_{metric}"])[:5]
        top_step_changes[metric] = {
            "improvements": improvements,
            "deteriorations": deteriorations,
        }

    return {
        "left_run": left_summary["run_label"],
        "right_run": right_summary["run_label"],
        "common_step_count": len(common_steps),
        "left_only_steps": sorted(set(left_by_step) - set(right_by_step)),
        "right_only_steps": sorted(set(right_by_step) - set(left_by_step)),
        "metric_delta_means": metric_delta_means,
        "top_step_changes": top_step_changes,
    }


def analyze_run(run_label, rollout_dir, expected_group_size, requested_group_mode, max_preview_chars):
    step_rows = []
    group_rows = []
    for step_file in list_step_files(rollout_dir):
        step_row, step_group_rows = analyze_step_file(
            step_file=step_file,
            run_label=run_label,
            expected_group_size=expected_group_size,
            requested_group_mode=requested_group_mode,
            max_preview_chars=max_preview_chars,
        )
        step_rows.append(step_row)
        group_rows.extend(step_group_rows)

    step_rows.sort(key=lambda row: row["step"])
    group_rows.sort(key=lambda row: (row["step"], row["group_key"]))
    summary = summarize_run(run_label, rollout_dir, step_rows, group_rows)
    ambiguous_rows = [row for row in group_rows if row["is_ambiguous"]]
    return summary, step_rows, group_rows, ambiguous_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def print_run_overview(summary):
    print(
        f"[Run] {summary['run_label']}: steps={summary['num_steps']}, "
        f"groups={summary['total_groups']}, ambiguous_groups={summary['total_ambiguous_groups']}, "
        f"group_modes={summary['group_mode_counts']}"
    )
    print(
        f"  mean_score={summary['mean_score_mean']}, "
        f"mean_acc={summary['mean_acc_mean']}, "
        f"mean_group_mean={summary['mean_group_mean_mean']}, "
        f"mean_group_std={summary['mean_group_std_mean']}, "
        f"mean_best_of_n={summary['mean_best_of_n_mean']}"
    )
    print(
        f"  mean_all_correct_group_rate={summary['mean_all_correct_group_rate']}, "
        f"mean_all_wrong_group_rate={summary['mean_all_wrong_group_rate']}, "
        f"mean_mixed_group_rate={summary['mean_mixed_group_rate']}"
    )


def print_comparison_overview(comparison):
    print(
        f"[Compare] {comparison['left_run']} -> {comparison['right_run']}: "
        f"common_steps={comparison['common_step_count']}, "
        f"left_only={len(comparison['left_only_steps'])}, "
        f"right_only={len(comparison['right_only_steps'])}"
    )
    print(f"  mean_deltas={comparison['metric_delta_means']}")
    for metric in MAIN_COMPARISON_METRICS:
        improvements = comparison["top_step_changes"][metric]["improvements"][:3]
        deteriorations = comparison["top_step_changes"][metric]["deteriorations"][:3]
        print(f"  {metric} top improvements={improvements}")
        print(f"  {metric} top deteriorations={deteriorations}")


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze GRPO rollout JSONL logs by step and reconstructed group.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_arg,
        help="Run specification in the form label=/abs/path. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for summary.json, steps.csv, groups.csv, and ambiguous_groups.csv.",
    )
    parser.add_argument(
        "--expected-group-size",
        type=int,
        default=8,
        help="Expected GRPO rollout group size.",
    )
    parser.add_argument(
        "--group-mode",
        choices=("auto", "uid", "prompt"),
        default="auto",
        help="Grouping mode. 'auto' prefers uid when present and falls back to prompt.",
    )
    parser.add_argument(
        "--max-prompt-preview-chars",
        type=int,
        default=DEFAULT_PREVIEW_CHARS,
        help="Maximum characters kept in prompt preview columns.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        parser.error(f"--output-dir must be absolute: {output_dir}")

    run_summaries = []
    all_step_rows = []
    all_group_rows = []
    all_ambiguous_rows = []

    for run_label, rollout_dir in args.run:
        if not rollout_dir.exists():
            parser.error(f"Rollout directory does not exist: {rollout_dir}")
        summary, step_rows, group_rows, ambiguous_rows = analyze_run(
            run_label=run_label,
            rollout_dir=rollout_dir,
            expected_group_size=args.expected_group_size,
            requested_group_mode=args.group_mode,
            max_preview_chars=args.max_prompt_preview_chars,
        )
        run_summaries.append(summary)
        all_step_rows.extend(step_rows)
        all_group_rows.extend(group_rows)
        all_ambiguous_rows.extend(ambiguous_rows)
        print_run_overview(summary)

    comparisons = []
    step_rows_by_label = defaultdict(list)
    for row in all_step_rows:
        step_rows_by_label[row["run_label"]].append(row)

    summaries_by_label = {summary["run_label"]: summary for summary in run_summaries}
    for left_label, right_label in combinations([summary["run_label"] for summary in run_summaries], 2):
        comparison = compare_two_runs(
            left_summary=summaries_by_label[left_label],
            right_summary=summaries_by_label[right_label],
            left_steps=step_rows_by_label[left_label],
            right_steps=step_rows_by_label[right_label],
        )
        comparisons.append(comparison)
        print_comparison_overview(comparison)

    write_summary(
        output_dir / "summary.json",
        {
            "runs": run_summaries,
            "comparisons": comparisons,
        },
    )
    write_csv(output_dir / "steps.csv", all_step_rows)
    write_csv(output_dir / "groups.csv", all_group_rows)
    write_csv(output_dir / "ambiguous_groups.csv", all_ambiguous_rows)


if __name__ == "__main__":
    main()
