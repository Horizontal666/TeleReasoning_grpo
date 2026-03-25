#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = REPO_ROOT / "data/eval_benchmark_CT/telemath/telemath_chat_template.json"
DEFAULT_TRAIN_PARQUET = REPO_ROOT / "data/GRPO/telemath_self_gen_v0/train.parquet"
DEFAULT_TEST_PARQUET = REPO_ROOT / "data/GRPO/telemath_self_gen_v0/test.parquet"


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


def normalize_text(value):
    text = str(value or "").strip().strip('"').strip("'")
    return " ".join(text.split())


def strip_telemath_suffix(question_text):
    text = str(question_text or "")
    text = re.sub(r"\n+Requirements:\s.*$", "", text, flags=re.S)
    text = re.sub(
        r"\s+Please reason step by step, and put your final answer within \\boxed\{\}\.\s*$",
        "",
        text,
    )
    return normalize_text(text)


def iter_prompt_messages(prompt_value):
    if prompt_value is None:
        return []
    if hasattr(prompt_value, "tolist"):
        prompt_value = prompt_value.tolist()
    if isinstance(prompt_value, dict):
        return [prompt_value]
    if isinstance(prompt_value, (list, tuple)):
        return list(prompt_value)
    return []


def question_from_prompt(prompt_value):
    if isinstance(prompt_value, str):
        return strip_telemath_suffix(prompt_value)
    messages = iter_prompt_messages(prompt_value)
    user_parts = [
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return strip_telemath_suffix("\n".join(user_parts))


def load_overlap_questions(train_parquet, test_parquet):
    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)
    train_questions = set(train_df["prompt"].map(question_from_prompt))
    test_questions = set(test_df["prompt"].map(question_from_prompt))
    return {question for question in (train_questions & test_questions) if question}


def load_eval_dataset_questions(dataset_path):
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        payload = payload["items"]
    return [normalize_text(item.get("question", "")) for item in payload if isinstance(item, dict)]


def fraction_true(values):
    if not values:
        return None
    return sum(bool(value) for value in values) / len(values)


def present_fraction(values):
    if not values:
        return None

    def is_present(value):
        if value is None:
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    return sum(1 for value in values if is_present(value)) / len(values)


def mean_or_none(values):
    if not values:
        return None
    return sum(values) / len(values)


def load_eval_records(eval_json_path):
    with eval_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {eval_json_path}, got {type(payload).__name__}")
    return payload


def summarize_eval_run(run_label, eval_json_path, overlap_questions):
    records = load_eval_records(eval_json_path)
    per_question_correct = defaultdict(list)
    overlap_correct = []

    boxed_values = []
    parsed_values = []
    correct_values = []
    math_verify_values = []

    for record in records:
        question = normalize_text(record.get("question", ""))
        correct = bool(record.get("correct", False))
        correct_math_verify = bool(record.get("correct_math_verify", False))
        boxed = bool(str(record.get("answer_segment", "") or "").strip())
        parsed = record.get("parsed_value")

        correct_values.append(correct)
        math_verify_values.append(correct_math_verify)
        boxed_values.append(boxed)
        parsed_values.append(parsed)

        if question:
            per_question_correct[question].append(correct)
            if question in overlap_questions:
                overlap_correct.append(correct)

    dedup_correct = mean_or_none(
        [mean_or_none(question_values) for question_values in per_question_correct.values()]
    )

    return {
        "run_label": run_label,
        "eval_json": str(eval_json_path),
        "num_samples": len(records),
        "num_unique_questions": len(per_question_correct),
        "duplicate_question_count": len(records) - len(per_question_correct),
        "correct": fraction_true(correct_values),
        "correct_count": sum(correct_values),
        "correct_math_verify": fraction_true(math_verify_values),
        "boxed_rate": fraction_true(boxed_values),
        "numeric_parse_success": present_fraction(parsed_values),
        "dedup_correct": dedup_correct,
        "overlap30_count": len(overlap_correct),
        "overlap30_correct": fraction_true(overlap_correct),
    }


def write_summary(output_dir, summary_rows, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata | {"runs": summary_rows}, handle, ensure_ascii=False, indent=2)

    fieldnames = [
        "run_label",
        "eval_json",
        "num_samples",
        "num_unique_questions",
        "duplicate_question_count",
        "correct",
        "correct_count",
        "correct_math_verify",
        "boxed_rate",
        "numeric_parse_success",
        "dedup_correct",
        "overlap30_count",
        "overlap30_correct",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_path, summary_csv_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Summarize TeleMath raw evaluation JSONs into a compact root-cause metrics table."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_labeled_path,
        required=True,
        help="Run label and raw TeleMath eval JSON, e.g. sft=/abs/path/result.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="TeleMath benchmark dataset used for evaluation.",
    )
    parser.add_argument(
        "--train-parquet",
        type=Path,
        default=DEFAULT_TRAIN_PARQUET,
        help="TeleMath GRPO train parquet for overlap-30 detection.",
    )
    parser.add_argument(
        "--test-parquet",
        type=Path,
        default=DEFAULT_TEST_PARQUET,
        help="TeleMath GRPO test parquet for overlap-30 detection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for summary.json and summary.csv.",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    overlap_questions = load_overlap_questions(args.train_parquet, args.test_parquet)
    dataset_questions = load_eval_dataset_questions(args.dataset)

    summary_rows = []
    for run_label, eval_json_path in args.run:
        summary_rows.append(summarize_eval_run(run_label, eval_json_path, overlap_questions))

    summary_rows.sort(key=lambda row: row["run_label"])
    metadata = {
        "dataset": str(args.dataset),
        "dataset_question_count": len(dataset_questions),
        "train_parquet": str(args.train_parquet),
        "test_parquet": str(args.test_parquet),
        "train_test_overlap_question_count": len(overlap_questions),
    }
    summary_path, summary_csv_path = write_summary(args.output_dir, summary_rows, metadata)

    print(f"Wrote TeleMath eval summary JSON to: {summary_path}")
    print(f"Wrote TeleMath eval summary CSV to: {summary_csv_path}")
    for row in summary_rows:
        print(
            f"{row['run_label']}: correct={row['correct']:.4f}, "
            f"dedup_correct={row['dedup_correct']:.4f}, "
            f"overlap30_correct={row['overlap30_correct']:.4f}, "
            f"boxed_rate={row['boxed_rate']:.4f}, "
            f"numeric_parse_success={row['numeric_parse_success']:.4f}"
        )


if __name__ == "__main__":
    main()
