#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def parse_labeled_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Invalid value {value!r}. Expected label=/abs/path")
    label, path_str = value.split("=", 1)
    label = label.strip()
    path = Path(path_str.strip()).expanduser()
    if not label:
        raise argparse.ArgumentTypeError(f"Invalid value {value!r}. Empty label.")
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"Path must be absolute: {path}")
    return label, path


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_offline_runs(path):
    payload = load_json(path)
    runs = payload.get("runs", [])
    return {run["run_label"]: run for run in runs}


def load_rollout_runs(path):
    payload = load_json(path)
    runs = payload.get("runs", [])
    return {run["run_label"]: run for run in runs}


def load_metadata_map(metadata_entries, metadata_dir):
    metadata_map = {}

    if metadata_dir is not None and metadata_dir.exists():
        for path in sorted(metadata_dir.glob("*.json")):
            payload = load_json(path)
            run_label = payload.get("run_label")
            if run_label:
                metadata_map[run_label] = payload

    for run_label, path in metadata_entries:
        payload = load_json(path)
        payload.setdefault("run_label", run_label)
        metadata_map[run_label] = payload

    return metadata_map


def build_compare_rows(offline_map, rollout_map, metadata_map):
    run_labels = sorted(set(offline_map) | set(rollout_map) | set(metadata_map))
    rows = []
    for run_label in run_labels:
        offline = offline_map.get(run_label, {})
        rollout = rollout_map.get(run_label, {})
        metadata = metadata_map.get(run_label, {})

        total_training_steps = metadata.get("total_training_steps")
        duration_seconds = metadata.get("duration_seconds")
        duration_hours = None
        steps_per_hour = None
        hours_100_steps = None
        if duration_seconds not in (None, 0) and total_training_steps not in (None, 0):
            duration_hours = duration_seconds / 3600.0
            steps_per_hour = total_training_steps / duration_hours if duration_hours > 0 else None
            hours_100_steps = 100.0 / steps_per_hour if steps_per_hour else None
        elif duration_seconds not in (None, 0):
            duration_hours = duration_seconds / 3600.0

        rows.append(
            {
                "run_label": run_label,
                "offline_correct": offline.get("correct"),
                "offline_dedup_correct": offline.get("dedup_correct"),
                "offline_overlap30_correct": offline.get("overlap30_correct"),
                "boxed_rate": offline.get("boxed_rate"),
                "numeric_parse_success": offline.get("numeric_parse_success"),
                "group_mean_mean": rollout.get("mean_group_mean_mean"),
                "group_std_mean": rollout.get("mean_group_std_mean"),
                "best_of_n_mean": rollout.get("mean_best_of_n_mean"),
                "all_wrong_group_rate": rollout.get("mean_all_wrong_group_rate"),
                "all_correct_group_rate": rollout.get("mean_all_correct_group_rate"),
                "mixed_group_rate": rollout.get("mean_mixed_group_rate"),
                "total_training_steps": total_training_steps,
                "duration_hours": duration_hours,
                "steps_per_hour": steps_per_hour,
                "hours_100_steps": hours_100_steps,
                "status": metadata.get("status"),
            }
        )

    return rows


def write_outputs(output_dir, rows, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "root_cause_summary.json"
    csv_path = output_dir / "root_cause_summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata | {"runs": rows}, handle, ensure_ascii=False, indent=2)

    fieldnames = [
        "run_label",
        "offline_correct",
        "offline_dedup_correct",
        "offline_overlap30_correct",
        "boxed_rate",
        "numeric_parse_success",
        "group_mean_mean",
        "group_std_mean",
        "best_of_n_mean",
        "all_wrong_group_rate",
        "all_correct_group_rate",
        "mixed_group_rate",
        "total_training_steps",
        "duration_hours",
        "steps_per_hour",
        "hours_100_steps",
        "status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Join TeleMath offline eval summaries, rollout summaries, and run metadata."
    )
    parser.add_argument("--offline-summary", type=Path, required=True)
    parser.add_argument("--rollout-summary", type=Path, default=None)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument(
        "--metadata",
        action="append",
        type=parse_labeled_path,
        default=[],
        help="Optional run label and metadata JSON, e.g. A0=/abs/path/run_metadata.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    offline_map = load_offline_runs(args.offline_summary)
    rollout_map = load_rollout_runs(args.rollout_summary) if args.rollout_summary else {}
    metadata_map = load_metadata_map(args.metadata, args.metadata_dir)

    rows = build_compare_rows(offline_map, rollout_map, metadata_map)
    json_path, csv_path = write_outputs(
        args.output_dir,
        rows,
        {
            "offline_summary": str(args.offline_summary),
            "rollout_summary": str(args.rollout_summary) if args.rollout_summary else None,
            "metadata_dir": str(args.metadata_dir) if args.metadata_dir else None,
        },
    )

    print(f"Wrote root-cause summary JSON to: {json_path}")
    print(f"Wrote root-cause summary CSV to: {csv_path}")
    for row in rows:
        print(
            f"{row['run_label']}: offline_correct={row['offline_correct']}, "
            f"all_wrong_group_rate={row['all_wrong_group_rate']}, "
            f"best_of_n_mean={row['best_of_n_mean']}, "
            f"steps_per_hour={row['steps_per_hour']}"
        )


if __name__ == "__main__":
    main()
