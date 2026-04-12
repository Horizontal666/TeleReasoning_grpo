#!/usr/bin/env python3
"""Analyze rollout errors and browse samples in the terminal.

This tool is intentionally standard-library only so it can run in lean
environments. The error categories are heuristic:

- suspected_truncation: output looks incomplete and likely stopped before the
  final answer was produced.
- likely_process_error: the model stayed on task but seems to have failed late
  (for example, coherent partial solution with no boxed answer, or a near-miss
  final answer).
- likely_early_offtrack: the model appears to drift early, answer the wrong
  task, or produce a confidently wrong final answer.

Use `analyze` to write machine-generated tags, then use `browse` to inspect and
optionally append manual labels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import pydoc
import re
import shutil
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "into",
    "same",
    "between",
    "using",
    "use",
    "given",
    "please",
    "reason",
    "step",
    "final",
    "answer",
    "within",
    "put",
    "what",
    "which",
    "when",
    "where",
    "therefore",
    "thus",
    "then",
    "need",
    "find",
    "determine",
    "calculate",
    "compute",
    "show",
    "derive",
}

LEAK_PATTERNS = [
    r"<\|im_",
    r"\b_system\b",
    r"\b_user\b",
    r"\bassistant\b",
    r"#\s*user",
    r"#\s*assistant",
    r"\[nextlink\]",
    r"helpuser",
    r"linked list",
    r"paraphrase",
    r"i apologize but the question you asked",
    r"write a program",
    r"cookies may be set",
    r"google ads toolbox",
]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def compact(text: Any, limit: int = 180) -> str:
    text = normalize_ws(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_]*", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def question_prefix_overlap(question: str, output_prefix: str) -> float:
    question_tokens = token_set(question)
    prefix_tokens = token_set(output_prefix)
    if not question_tokens:
        return 0.0
    return len(question_tokens & prefix_tokens) / len(question_tokens)


def has_prompt_leak(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in LEAK_PATTERNS)


def has_repetition(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())[:2000]
    if not tokens:
        return False

    max_run = 1
    current_run = 1
    for previous, current in zip(tokens, tokens[1:]):
        if previous == current:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    if max_run >= 8:
        return True

    token_counts = Counter(tokens)
    _, frequency = token_counts.most_common(1)[0]
    return frequency >= 25 and frequency / len(tokens) >= 0.08


def has_open_box_tail(text: str) -> bool:
    tail = text[-300:]
    return "\\boxed{" in tail and "}" not in tail.split("\\boxed{")[-1]


def unfinished_tail(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return True
    tail = stripped[-240:]
    if tail.count("```") % 2 == 1:
        return True
    if has_open_box_tail(stripped):
        return True
    if stripped.endswith(("\\", "/", "(", "[", "{", "=", "+", "-", "*", ":", ",")):
        return True
    if stripped[-1].isalnum() and not re.search(r"[.!?\]\)\}]$", stripped):
        return True
    if re.search(r"(for|if|then|because|approximately|approx|therefore|thus|let|suppose)\s*$", tail.lower()):
        return True
    return False


def normalize_answer_text(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    return text


def parse_numeric_literal(text: Any) -> float | None:
    value = str(text or "").strip()
    if not value:
        return None

    value = value.replace("$", "").replace("%", "").replace(",", "").strip()
    value = re.sub(r"\\text\{([^}]*)\}", "", value).strip()
    value = value.replace("−", "-").replace("–", "-")
    value = value.replace(" ", "")

    if "=" in value and value.count("=") == 1:
        left, right = value.split("=", 1)
        if re.fullmatch(r"[A-Za-z_\\{}]+", left):
            value = right

    if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value):
        return float(value)

    fraction_match = re.fullmatch(r"[-+]?\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", value)
    if fraction_match:
        denominator = float(fraction_match.group(2))
        if denominator == 0:
            return None
        return float(fraction_match.group(1)) / denominator

    plain_fraction_match = re.fullmatch(r"[-+]?\d+/[-+]?\d+", value)
    if plain_fraction_match:
        numerator, denominator_text = value.split("/", 1)
        denominator = float(denominator_text)
        if denominator == 0:
            return None
        return float(numerator) / denominator

    return None


def is_near_miss(pred: Any, ground_truth: Any, rel_tol: float) -> bool:
    pred_num = parse_numeric_literal(pred)
    gt_num = parse_numeric_literal(ground_truth)
    if pred_num is not None and gt_num is not None:
        scale = max(1.0, abs(gt_num))
        return abs(pred_num - gt_num) / scale <= rel_tol

    pred_norm = normalize_answer_text(pred)
    gt_norm = normalize_answer_text(ground_truth)
    if not pred_norm or not gt_norm:
        return False
    if len(pred_norm) > 60 or len(gt_norm) > 60:
        return False
    return difflib.SequenceMatcher(None, pred_norm, gt_norm).ratio() >= 0.85


def infer_summary_path(rollout_jsonl: Path) -> Path | None:
    summary_path = rollout_jsonl.with_name("summary.json")
    return summary_path if summary_path.exists() else None


def infer_dataset_path(rollout_jsonl: Path) -> Path | None:
    summary_path = infer_summary_path(rollout_jsonl)
    if summary_path is None:
        return None
    try:
        summary = read_json(summary_path)
    except Exception:
        return None
    data_path = summary.get("data_path")
    if not data_path:
        return None
    path = Path(data_path)
    return path if path.exists() else None


def infer_sample_summary_path(rollout_jsonl: Path) -> Path | None:
    sample_summary_path = rollout_jsonl.with_name("sample_summary.jsonl")
    return sample_summary_path if sample_summary_path.exists() else None


def load_dataset_map(dataset_jsonl: Path | None) -> dict[int, dict[str, Any]]:
    if dataset_jsonl is None:
        return {}
    dataset_map: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(iter_jsonl(dataset_jsonl)):
        dataset_map[index] = row
    return dataset_map


def extract_reference_solution(dataset_row: dict[str, Any] | None) -> str:
    if not dataset_row:
        return ""
    extra_info = dataset_row.get("extra_info") if isinstance(dataset_row.get("extra_info"), dict) else {}
    candidates = [
        extra_info.get("answer"),
        extra_info.get("raw_solution"),
        dataset_row.get("solution"),
        dataset_row.get("answer"),
    ]
    for candidate in candidates:
        candidate_text = normalize_ws(candidate)
        if candidate_text:
            return candidate_text
    return ""


def extract_raw_reference_answer(dataset_row: dict[str, Any] | None) -> str:
    if not dataset_row:
        return ""
    extra_info = dataset_row.get("extra_info") if isinstance(dataset_row.get("extra_info"), dict) else {}
    return normalize_ws(extra_info.get("reference_answer"))


def extract_reference_answer(dataset_row: dict[str, Any] | None, fallback_gt: Any) -> str:
    if dataset_row:
        extra_info = dataset_row.get("extra_info") if isinstance(dataset_row.get("extra_info"), dict) else {}
        for candidate in (
            extra_info.get("reference_answer"),
            extra_info.get("normalized_answer"),
            extra_info.get("answer"),
        ):
            candidate_text = normalize_ws(candidate)
            if candidate_text:
                return candidate_text
        reward_model = dataset_row.get("reward_model")
        if isinstance(reward_model, dict):
            candidate_text = normalize_ws(reward_model.get("ground_truth"))
            if candidate_text:
                return candidate_text
    return normalize_ws(fallback_gt)


def extract_question(dataset_row: dict[str, Any] | None, rollout_row: dict[str, Any]) -> str:
    if dataset_row:
        prompt = dataset_row.get("prompt")
        if isinstance(prompt, list):
            for message in prompt:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = normalize_ws(message.get("content"))
                    if content:
                        return content
        extra_info = dataset_row.get("extra_info") if isinstance(dataset_row.get("extra_info"), dict) else {}
        for candidate in (extra_info.get("question_with_unit"), extra_info.get("question_text")):
            candidate_text = normalize_ws(candidate)
            if candidate_text:
                return candidate_text
    return normalize_ws(rollout_row.get("question"))


def analyze_rollout_row(
    rollout_row: dict[str, Any],
    dataset_row: dict[str, Any] | None,
    truncation_char_threshold: int,
    near_miss_rel_tol: float,
    process_overlap_threshold: float,
) -> dict[str, Any]:
    output_text = normalize_ws(rollout_row.get("output"))
    question_text = extract_question(dataset_row, rollout_row)
    output_prefix = output_text[:1200]

    no_final_answer = not bool(rollout_row.get("format_correct"))
    prompt_leak = has_prompt_leak(output_prefix) or has_prompt_leak(output_text[-400:])
    repetition = has_repetition(output_text)
    incomplete_tail = unfinished_tail(output_text)
    overlap = question_prefix_overlap(question_text, output_prefix)
    open_box_tail = has_open_box_tail(output_text)
    output_chars = len(output_text)
    raw_reference_answer = extract_raw_reference_answer(dataset_row)
    reference_answer_disagrees_with_gts = bool(
        raw_reference_answer and normalize_answer_text(raw_reference_answer) != normalize_answer_text(rollout_row.get("gts"))
    )

    suspected_truncation = no_final_answer and (
        open_box_tail or (incomplete_tail and output_chars >= truncation_char_threshold)
    )
    coherent_no_final_answer = no_final_answer and overlap >= process_overlap_threshold and not prompt_leak and not repetition
    near_miss_final_answer = bool(
        rollout_row.get("format_correct")
        and is_near_miss(rollout_row.get("pred"), rollout_row.get("gts"), near_miss_rel_tol)
    )
    likely_process_error = coherent_no_final_answer or near_miss_final_answer
    likely_early_offtrack = prompt_leak or repetition or (not likely_process_error and not suspected_truncation)

    reason_tags: list[str] = []
    if suspected_truncation:
        reason_tags.append("suspected_truncation")
    if coherent_no_final_answer:
        reason_tags.append("coherent_no_final_answer")
    if near_miss_final_answer:
        reason_tags.append("near_miss_final_answer")
    if prompt_leak:
        reason_tags.append("prompt_leak")
    if repetition:
        reason_tags.append("repetition")
    if incomplete_tail:
        reason_tags.append("unfinished_tail")
    if no_final_answer:
        reason_tags.append("missing_final_answer")
    if reference_answer_disagrees_with_gts:
        reason_tags.append("reference_answer_disagrees_with_gts")

    if suspected_truncation:
        primary_label = "suspected_truncation"
    elif likely_process_error:
        primary_label = "likely_process_error"
    else:
        primary_label = "likely_early_offtrack"

    return {
        "sample_index": rollout_row.get("sample_index"),
        "rollout_index": rollout_row.get("rollout_index"),
        "uid": rollout_row.get("uid"),
        "difficulty": rollout_row.get("difficulty"),
        "sample_avg_acc": rollout_row.get("sample_avg_acc"),
        "sample_avg_score": rollout_row.get("sample_avg_score"),
        "acc": bool(rollout_row.get("acc")),
        "format_correct": bool(rollout_row.get("format_correct")),
        "pred": rollout_row.get("pred"),
        "gts": rollout_row.get("gts"),
        "primary_label": primary_label,
        "reason_tags": reason_tags,
        "flags": {
            "suspected_truncation": suspected_truncation,
            "likely_process_error": likely_process_error,
            "likely_early_offtrack": likely_early_offtrack,
            "coherent_no_final_answer": coherent_no_final_answer,
            "near_miss_final_answer": near_miss_final_answer,
            "prompt_leak": prompt_leak,
            "repetition": repetition,
            "unfinished_tail": incomplete_tail,
            "missing_final_answer": no_final_answer,
            "reference_answer_disagrees_with_gts": reference_answer_disagrees_with_gts,
        },
        "heuristics": {
            "output_chars": output_chars,
            "question_prefix_overlap": round(overlap, 6),
            "open_box_tail": open_box_tail,
            "truncation_char_threshold": truncation_char_threshold,
            "near_miss_rel_tol": near_miss_rel_tol,
            "process_overlap_threshold": process_overlap_threshold,
        },
    }


def build_examples(analysis_rows: list[dict[str, Any]], max_per_label: int = 8) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in analysis_rows:
        label = row["primary_label"]
        if len(grouped[label]) >= max_per_label:
            continue
        grouped[label].append(
            {
                "sample_index": row["sample_index"],
                "rollout_index": row["rollout_index"],
                "pred": row.get("pred"),
                "gts": row.get("gts"),
                "reason_tags": row.get("reason_tags", []),
                "output_chars": row.get("heuristics", {}).get("output_chars"),
            }
        )
    return dict(grouped)


def summarize_analysis(
    rollout_rows: list[dict[str, Any]],
    analysis_rows: list[dict[str, Any]],
    rollout_jsonl: Path,
    dataset_jsonl: Path | None,
    analysis_jsonl: Path,
    truncation_char_threshold: int,
    near_miss_rel_tol: float,
    process_overlap_threshold: float,
) -> dict[str, Any]:
    correct_rollouts = sum(1 for row in rollout_rows if row.get("acc"))
    wrong_rollouts = len(analysis_rows)
    wrong_sample_indexes = {row["sample_index"] for row in analysis_rows}

    primary_counts = Counter(row["primary_label"] for row in analysis_rows)
    flag_counts = Counter()
    flag_sample_sets: dict[str, set[int]] = defaultdict(set)
    primary_sample_sets: dict[str, set[int]] = defaultdict(set)
    sample_primary_counts: dict[int, Counter[str]] = defaultdict(Counter)

    for row in analysis_rows:
        sample_index = int(row["sample_index"])
        primary = row["primary_label"]
        primary_sample_sets[primary].add(sample_index)
        sample_primary_counts[sample_index][primary] += 1
        for flag_name, value in row.get("flags", {}).items():
            if value:
                flag_counts[flag_name] += 1
                flag_sample_sets[flag_name].add(sample_index)

    dominant_primary_counts = Counter()
    precedence = {
        "suspected_truncation": 0,
        "likely_process_error": 1,
        "likely_early_offtrack": 2,
    }
    for counter in sample_primary_counts.values():
        dominant = sorted(counter.items(), key=lambda item: (-item[1], precedence.get(item[0], 99), item[0]))[0][0]
        dominant_primary_counts[dominant] += 1

    return {
        "rollout_jsonl": str(rollout_jsonl),
        "dataset_jsonl": str(dataset_jsonl) if dataset_jsonl else None,
        "analysis_jsonl": str(analysis_jsonl),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "heuristics": {
            "truncation_char_threshold": truncation_char_threshold,
            "near_miss_rel_tol": near_miss_rel_tol,
            "process_overlap_threshold": process_overlap_threshold,
            "notes": [
                "suspected_truncation is heuristic because finish_reason is not present in rollout_jsonl",
                "likely_process_error focuses on coherent no-final-answer outputs and near-miss final answers",
                "likely_early_offtrack is the residual bucket for prompt drift, repetition, or confidently wrong outputs",
            ],
        },
        "counts": {
            "total_rollouts": len(rollout_rows),
            "correct_rollouts": correct_rollouts,
            "wrong_rollouts": wrong_rollouts,
            "wrong_samples": len(wrong_sample_indexes),
        },
        "row_level": {
            "primary_label_counts": dict(primary_counts),
            "flag_counts": dict(flag_counts),
        },
        "sample_level": {
            "samples_with_primary_label": {label: len(indexes) for label, indexes in primary_sample_sets.items()},
            "samples_with_flag": {flag: len(indexes) for flag, indexes in flag_sample_sets.items()},
            "dominant_primary_label_counts": dict(dominant_primary_counts),
        },
        "examples": build_examples(analysis_rows),
    }


def default_analysis_paths(rollout_jsonl: Path) -> tuple[Path, Path]:
    stem = rollout_jsonl.stem
    analysis_jsonl = rollout_jsonl.with_name(f"{stem}.error_analysis.jsonl")
    summary_json = rollout_jsonl.with_name(f"{stem}.error_summary.json")
    return analysis_jsonl, summary_json


def split_category_for_sample(
    per_rollout_analysis: list[dict[str, Any]],
) -> tuple[str, dict[str, int]]:
    primary_counts = Counter(row.get("primary_label") for row in per_rollout_analysis)
    flag_counts = Counter()
    for row in per_rollout_analysis:
        for flag_name, value in row.get("flags", {}).items():
            if value:
                flag_counts[flag_name] += 1

    if flag_counts["reference_answer_disagrees_with_gts"] > 0:
        return "dirty_label", dict(flag_counts)
    if flag_counts["suspected_truncation"] > 0 or flag_counts["coherent_no_final_answer"] >= 2:
        return "finish_issue", dict(flag_counts)
    if flag_counts["near_miss_final_answer"] > 0 or primary_counts["likely_process_error"] >= 2:
        return "process_error", dict(flag_counts)
    return "early_offtrack", dict(flag_counts)


def run_split(args: argparse.Namespace) -> int:
    rollout_jsonl = Path(args.rollout_jsonl).resolve()
    dataset_jsonl = Path(args.dataset_jsonl).resolve() if args.dataset_jsonl else infer_dataset_path(rollout_jsonl)
    sample_summary_jsonl = (
        Path(args.sample_summary_jsonl).resolve() if args.sample_summary_jsonl else infer_sample_summary_path(rollout_jsonl)
    )
    default_analysis_jsonl, _ = default_analysis_paths(rollout_jsonl)
    analysis_jsonl = Path(args.analysis_jsonl).resolve() if args.analysis_jsonl else default_analysis_jsonl

    if dataset_jsonl is None or not dataset_jsonl.exists():
        print("Dataset jsonl not found. Pass --dataset-jsonl or make sure summary.json points to a valid data_path.", file=sys.stderr)
        return 1
    if sample_summary_jsonl is None or not sample_summary_jsonl.exists():
        print("sample_summary.jsonl not found. Pass --sample-summary-jsonl.", file=sys.stderr)
        return 1
    if not analysis_jsonl.exists():
        print(f"Analysis jsonl not found: {analysis_jsonl}", file=sys.stderr)
        print("Run the analyze subcommand first, or pass --analysis-jsonl.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else rollout_jsonl.parent / "sample_splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_map = load_dataset_map(dataset_jsonl)
    analysis_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(analysis_jsonl):
        analysis_by_sample[int(row["sample_index"])].append(row)

    target_sample_indexes: list[int] = []
    sample_summary_rows: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(sample_summary_jsonl):
        if row.get("difficulty") != args.difficulty:
            continue
        sample_index = int(row["sample_index"])
        target_sample_indexes.append(sample_index)
        sample_summary_rows[sample_index] = row

    split_to_rows: dict[str, list[dict[str, Any]]] = {
        "dirty_label": [],
        "finish_issue": [],
        "process_error": [],
        "early_offtrack": [],
    }
    split_meta_rows: list[dict[str, Any]] = []

    for sample_index in sorted(target_sample_indexes):
        dataset_row = dataset_map.get(sample_index)
        if dataset_row is None:
            print(f"Missing dataset row for sample_index={sample_index}", file=sys.stderr)
            return 1
        per_rollout_analysis = analysis_by_sample.get(sample_index, [])
        category, flag_counts = split_category_for_sample(per_rollout_analysis)
        split_to_rows[category].append(dataset_row)

        primary_counts = Counter(row.get("primary_label") for row in per_rollout_analysis)
        split_meta_rows.append(
            {
                "sample_index": sample_index,
                "category": category,
                "difficulty": args.difficulty,
                "uid": sample_summary_rows.get(sample_index, {}).get("uid"),
                "avg_acc": sample_summary_rows.get(sample_index, {}).get("avg_acc"),
                "avg_score": sample_summary_rows.get(sample_index, {}).get("avg_score"),
                "primary_label_counts": dict(primary_counts),
                "flag_counts": flag_counts,
            }
        )

    file_map = {
        "dirty_label": output_dir / f"{args.difficulty}.dirty_label.dataset.jsonl",
        "finish_issue": output_dir / f"{args.difficulty}.finish_issue.dataset.jsonl",
        "process_error": output_dir / f"{args.difficulty}.process_error.dataset.jsonl",
        "early_offtrack": output_dir / f"{args.difficulty}.early_offtrack.dataset.jsonl",
    }
    for category, rows in split_to_rows.items():
        write_jsonl(file_map[category], rows)

    meta_jsonl = output_dir / f"{args.difficulty}.split_meta.jsonl"
    write_jsonl(meta_jsonl, split_meta_rows)

    summary_json = output_dir / f"{args.difficulty}.split_summary.json"
    summary_payload = {
        "rollout_jsonl": str(rollout_jsonl),
        "dataset_jsonl": str(dataset_jsonl),
        "sample_summary_jsonl": str(sample_summary_jsonl),
        "analysis_jsonl": str(analysis_jsonl),
        "difficulty": args.difficulty,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rules": {
            "dirty_label": "Any rollout for the sample has reference_answer_disagrees_with_gts",
            "finish_issue": "Else, suspected_truncation appears in any rollout or coherent_no_final_answer appears in at least 2 rollouts",
            "process_error": "Else, near_miss_final_answer appears in any rollout or likely_process_error is the primary label in at least 2 rollouts",
            "early_offtrack": "Residual bucket after the three rules above",
        },
        "counts": {category: len(rows) for category, rows in split_to_rows.items()},
        "files": {category: str(path) for category, path in file_map.items()},
        "meta_jsonl": str(meta_jsonl),
    }
    write_json(summary_json, summary_payload)

    print(f"Split difficulty={args.difficulty} into {output_dir}")
    for category, path in file_map.items():
        print(f"{category:>15}: {len(split_to_rows[category])} -> {path}")
    print(f"{'meta':>15}: {len(split_meta_rows)} -> {meta_jsonl}")
    print(f"{'summary':>15}: {summary_json}")
    return 0


def run_analyze(args: argparse.Namespace) -> int:
    rollout_jsonl = Path(args.rollout_jsonl).resolve()
    dataset_jsonl = Path(args.dataset_jsonl).resolve() if args.dataset_jsonl else infer_dataset_path(rollout_jsonl)

    analysis_jsonl, summary_json = default_analysis_paths(rollout_jsonl)
    if args.analysis_jsonl:
        analysis_jsonl = Path(args.analysis_jsonl).resolve()
    if args.summary_json:
        summary_json = Path(args.summary_json).resolve()

    rollout_rows = list(iter_jsonl(rollout_jsonl))
    dataset_map = load_dataset_map(dataset_jsonl)

    analysis_rows: list[dict[str, Any]] = []
    for rollout_row in rollout_rows:
        if rollout_row.get("acc"):
            continue
        dataset_row = dataset_map.get(int(rollout_row["sample_index"]))
        analysis_rows.append(
            analyze_rollout_row(
                rollout_row=rollout_row,
                dataset_row=dataset_row,
                truncation_char_threshold=args.truncation_char_threshold,
                near_miss_rel_tol=args.near_miss_rel_tol,
                process_overlap_threshold=args.process_overlap_threshold,
            )
        )

    summary = summarize_analysis(
        rollout_rows=rollout_rows,
        analysis_rows=analysis_rows,
        rollout_jsonl=rollout_jsonl,
        dataset_jsonl=dataset_jsonl,
        analysis_jsonl=analysis_jsonl,
        truncation_char_threshold=args.truncation_char_threshold,
        near_miss_rel_tol=args.near_miss_rel_tol,
        process_overlap_threshold=args.process_overlap_threshold,
    )

    write_jsonl(analysis_jsonl, analysis_rows)
    write_json(summary_json, summary)

    print(f"Analyzed wrong rollouts: {summary['counts']['wrong_rollouts']}")
    print(f"Wrong samples: {summary['counts']['wrong_samples']}")
    print(f"Analysis JSONL: {analysis_jsonl}")
    print(f"Summary JSON:   {summary_json}")
    print(json.dumps(summary["row_level"]["primary_label_counts"], ensure_ascii=False, indent=2))
    return 0


def load_analysis_map(analysis_jsonl: Path) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["sample_index"]), int(row["rollout_index"])): row
        for row in iter_jsonl(analysis_jsonl)
    }


def load_manual_labels(path: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = (int(row["sample_index"]), int(row["rollout_index"]))
        latest[key] = row
    return latest


def append_manual_label(path: Path, sample_index: int, rollout_index: int, label: str, note: str) -> None:
    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_index": sample_index,
        "rollout_index": rollout_index,
        "manual_label": label,
        "note": note,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def wrap_block(text: str, width: int) -> str:
    paragraphs = text.splitlines() or [text]
    wrapped_parts: list[str] = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            wrapped_parts.append("")
            continue
        wrapped_parts.append(textwrap.fill(paragraph, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(wrapped_parts)


def render_record(
    position: int,
    total: int,
    rollout_row: dict[str, Any],
    analysis_row: dict[str, Any] | None,
    dataset_row: dict[str, Any] | None,
    manual_label: dict[str, Any] | None,
) -> str:
    width = max(88, min(140, shutil.get_terminal_size(fallback=(120, 40)).columns - 2))
    header_line = "=" * width
    question = extract_question(dataset_row, rollout_row)
    reference_solution = extract_reference_solution(dataset_row)
    reference_answer = extract_reference_answer(dataset_row, rollout_row.get("gts"))

    analysis_row = analysis_row or {}
    flags = analysis_row.get("flags", {})
    heuristics = analysis_row.get("heuristics", {})

    lines = [
        header_line,
        f"Record {position}/{total} | sample_index={rollout_row.get('sample_index')} | rollout_index={rollout_row.get('rollout_index')} | uid={rollout_row.get('uid')}",
        f"primary_label={analysis_row.get('primary_label', 'N/A')}",
        f"reason_tags={', '.join(analysis_row.get('reason_tags', [])) or 'none'}",
        (
            "flags="
            + ", ".join(f"{name}={value}" for name, value in flags.items())
            if flags
            else "flags=N/A"
        ),
        (
            "heuristics="
            f"output_chars={heuristics.get('output_chars')}, "
            f"question_prefix_overlap={heuristics.get('question_prefix_overlap')}, "
            f"unfinished_tail={flags.get('unfinished_tail')}, "
            f"reference_answer_disagrees_with_gts={flags.get('reference_answer_disagrees_with_gts')}"
            if heuristics
            else "heuristics=N/A"
        ),
        f"score={rollout_row.get('score')} | reward={rollout_row.get('reward')} | format_correct={rollout_row.get('format_correct')}",
        f"pred={rollout_row.get('pred')}",
        f"gts={rollout_row.get('gts')}",
        f"reference_answer={reference_answer}",
    ]

    if manual_label:
        lines.append(
            "manual_label="
            f"{manual_label.get('manual_label')} | note={manual_label.get('note') or ''} | "
            f"timestamp={manual_label.get('timestamp')}"
        )
    else:
        lines.append("manual_label=None")

    lines.extend(
        [
            "",
            "[Question]",
            wrap_block(question, width),
            "",
            "[Reference Solution]",
            wrap_block(reference_solution or "(not found in dataset row)", width),
            "",
            "[Model Output]",
            wrap_block(str(rollout_row.get("output") or ""), width),
            "",
            "[Commands]",
            "n next | p previous | j <position> | s <sample_index> | tag <label> [note] | q quit | ? help",
            header_line,
        ]
    )
    return "\n".join(lines)


def filter_records(
    rollout_rows: list[dict[str, Any]],
    analysis_map: dict[tuple[int, int], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rollout_rows:
        key = (int(row["sample_index"]), int(row["rollout_index"]))
        analysis_row = analysis_map.get(key)
        if args.only_wrong and row.get("acc"):
            continue
        if args.primary_label and (not analysis_row or analysis_row.get("primary_label") != args.primary_label):
            continue
        if args.flag:
            if not analysis_row or not analysis_row.get("flags", {}).get(args.flag):
                continue
        if args.sample_index is not None and int(row["sample_index"]) != args.sample_index:
            continue
        if args.rollout_index is not None and int(row["rollout_index"]) != args.rollout_index:
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: (int(row["sample_index"]), int(row["rollout_index"])))
    return filtered


def browse_help() -> str:
    return "\n".join(
        [
            "Commands:",
            "  n                    next record",
            "  p                    previous record",
            "  j <position>         jump to filtered-list position (1-based)",
            "  s <sample_index>     jump to the first record for a sample_index",
            "  tag <label> [note]   append a manual label record",
            "  q                    quit",
            "  ?                    show this help",
        ]
    )


def run_browse(args: argparse.Namespace) -> int:
    rollout_jsonl = Path(args.rollout_jsonl).resolve()
    dataset_jsonl = Path(args.dataset_jsonl).resolve() if args.dataset_jsonl else infer_dataset_path(rollout_jsonl)

    default_analysis_jsonl, _ = default_analysis_paths(rollout_jsonl)
    analysis_jsonl = Path(args.analysis_jsonl).resolve() if args.analysis_jsonl else default_analysis_jsonl
    if not analysis_jsonl.exists():
        print(f"Analysis file not found: {analysis_jsonl}", file=sys.stderr)
        print("Run the analyze subcommand first, or pass --analysis-jsonl.", file=sys.stderr)
        return 1

    manual_labels_path = Path(args.manual_labels_jsonl).resolve() if args.manual_labels_jsonl else None
    analysis_map = load_analysis_map(analysis_jsonl)
    dataset_map = load_dataset_map(dataset_jsonl)
    manual_labels = load_manual_labels(manual_labels_path)
    rollout_rows = list(iter_jsonl(rollout_jsonl))
    filtered_rows = filter_records(rollout_rows, analysis_map, args)

    if not filtered_rows:
        print("No records matched the current filters.")
        return 0

    position = 0
    if args.start_position:
        position = max(0, min(len(filtered_rows) - 1, args.start_position - 1))

    while True:
        rollout_row = filtered_rows[position]
        key = (int(rollout_row["sample_index"]), int(rollout_row["rollout_index"]))
        analysis_row = analysis_map.get(key)
        dataset_row = dataset_map.get(int(rollout_row["sample_index"]))
        manual_label = manual_labels.get(key)

        record_text = render_record(position + 1, len(filtered_rows), rollout_row, analysis_row, dataset_row, manual_label)
        pydoc.pager(record_text)

        try:
            command = input("browse> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not command or command == "n":
            position = (position + 1) % len(filtered_rows)
            continue
        if command == "p":
            position = (position - 1) % len(filtered_rows)
            continue
        if command == "q":
            return 0
        if command == "?":
            print(browse_help())
            continue
        if command.startswith("j "):
            _, _, remainder = command.partition(" ")
            if remainder.isdigit():
                requested = int(remainder)
                if 1 <= requested <= len(filtered_rows):
                    position = requested - 1
                else:
                    print(f"Position out of range: {requested}")
            else:
                print("Usage: j <position>")
            continue
        if command.startswith("s "):
            _, _, remainder = command.partition(" ")
            if remainder.lstrip("-").isdigit():
                sample_index = int(remainder)
                found = next(
                    (
                        idx
                        for idx, row in enumerate(filtered_rows)
                        if int(row["sample_index"]) == sample_index
                    ),
                    None,
                )
                if found is None:
                    print(f"sample_index={sample_index} not found in the filtered set")
                else:
                    position = found
            else:
                print("Usage: s <sample_index>")
            continue
        if command.startswith("tag "):
            if manual_labels_path is None:
                print("Pass --manual-labels-jsonl to enable manual tagging.")
                continue
            _, _, remainder = command.partition(" ")
            parts = remainder.split(None, 1)
            if not parts:
                print("Usage: tag <label> [note]")
                continue
            label = parts[0]
            note = parts[1] if len(parts) > 1 else ""
            append_manual_label(
                path=manual_labels_path,
                sample_index=int(rollout_row["sample_index"]),
                rollout_index=int(rollout_row["rollout_index"]),
                label=label,
                note=note,
            )
            manual_labels = load_manual_labels(manual_labels_path)
            print(
                "Saved manual label "
                f"{label!r} for sample_index={rollout_row['sample_index']} rollout_index={rollout_row['rollout_index']}"
            )
            continue

        print("Unknown command. Enter ? for help.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="write heuristic error labels for wrong rollouts")
    analyze_parser.add_argument("--rollout-jsonl", required=True, help="path to rollout jsonl, for example outputs/.../0.jsonl")
    analyze_parser.add_argument("--dataset-jsonl", help="optional dataset jsonl; if omitted, infer from summary.json")
    analyze_parser.add_argument("--analysis-jsonl", help="output path for the per-rollout analysis jsonl")
    analyze_parser.add_argument("--summary-json", help="output path for the aggregate summary json")
    analyze_parser.add_argument(
        "--truncation-char-threshold",
        type=int,
        default=8000,
        help="character proxy used for suspected truncation when finish_reason is unavailable",
    )
    analyze_parser.add_argument(
        "--near-miss-rel-tol",
        type=float,
        default=0.05,
        help="relative tolerance for tagging near-miss final answers",
    )
    analyze_parser.add_argument(
        "--process-overlap-threshold",
        type=float,
        default=0.12,
        help="minimum question-prefix token overlap for coherent no-final-answer detection",
    )
    analyze_parser.set_defaults(func=run_analyze)

    browse_parser = subparsers.add_parser("browse", help="browse rollout samples in the terminal")
    browse_parser.add_argument("--rollout-jsonl", required=True, help="path to rollout jsonl")
    browse_parser.add_argument("--dataset-jsonl", help="optional dataset jsonl; if omitted, infer from summary.json")
    browse_parser.add_argument("--analysis-jsonl", help="path to the analysis jsonl produced by analyze")
    browse_parser.add_argument("--manual-labels-jsonl", help="append manual labels to this jsonl during browsing")
    browse_parser.add_argument("--only-wrong", action="store_true", default=True, help="show only wrong rollouts (default: true)")
    browse_parser.add_argument(
        "--primary-label",
        choices=["suspected_truncation", "likely_process_error", "likely_early_offtrack"],
        help="filter by primary label",
    )
    browse_parser.add_argument(
        "--flag",
        choices=[
            "suspected_truncation",
            "likely_process_error",
            "likely_early_offtrack",
            "coherent_no_final_answer",
            "near_miss_final_answer",
            "prompt_leak",
            "repetition",
            "unfinished_tail",
            "missing_final_answer",
            "reference_answer_disagrees_with_gts",
        ],
        help="filter by a boolean analysis flag",
    )
    browse_parser.add_argument("--sample-index", type=int, help="only show this sample_index")
    browse_parser.add_argument("--rollout-index", type=int, help="only show this rollout_index")
    browse_parser.add_argument("--start-position", type=int, help="1-based starting position in the filtered set")
    browse_parser.set_defaults(func=run_browse)

    split_parser = subparsers.add_parser("split", help="split samples into four dataset jsonl files")
    split_parser.add_argument("--rollout-jsonl", required=True, help="path to rollout jsonl")
    split_parser.add_argument("--dataset-jsonl", help="optional dataset jsonl; if omitted, infer from summary.json")
    split_parser.add_argument("--sample-summary-jsonl", help="optional sample_summary jsonl; if omitted, infer next to rollout")
    split_parser.add_argument("--analysis-jsonl", help="path to the analysis jsonl produced by analyze")
    split_parser.add_argument("--difficulty", default="too_hard", help="which difficulty bucket to split, default: too_hard")
    split_parser.add_argument("--output-dir", help="directory for the four split jsonl files")
    split_parser.set_defaults(func=run_split)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
