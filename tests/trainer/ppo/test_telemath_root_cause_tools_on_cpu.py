import importlib.util
import json
from pathlib import Path

import pandas as pd


COLLECT_PATH = Path(
    "/dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl/examples/grpo_TeleInfer/collect_telemath_eval_metrics.py"
)
SUMMARY_PATH = Path(
    "/dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl/examples/grpo_TeleInfer/summarize_telemath_root_cause.py"
)


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_collect_telemath_eval_metrics_and_join_summary(tmp_path):
    collect_module = load_module(COLLECT_PATH, "collect_telemath_eval_metrics")
    summary_module = load_module(SUMMARY_PATH, "summarize_telemath_root_cause")

    eval_dataset = [
        {"question": "Q1"},
        {"question": "Q2"},
        {"question": "Q3"},
    ]
    eval_dataset_path = tmp_path / "dataset.json"
    write_json(eval_dataset_path, eval_dataset)

    train_df = pd.DataFrame(
        {
            "prompt": [
                "Q1\n\nRequirements:\n1. step by step",
                "Train only question",
            ]
        }
    )
    test_df = pd.DataFrame(
        {
            "prompt": [
                "Q1\n\nRequirements:\n1. step by step",
                "Test only question",
            ]
        }
    )
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    eval_records = [
        {
            "question": "Q1",
            "answer_segment": "1",
            "parsed_value": 1.0,
            "correct_math_verify": True,
            "correct": True,
        },
        {
            "question": "Q1",
            "answer_segment": "",
            "parsed_value": None,
            "correct_math_verify": False,
            "correct": False,
        },
        {
            "question": "Q2",
            "answer_segment": "2",
            "parsed_value": 2.0,
            "correct_math_verify": True,
            "correct": True,
        },
        {
            "question": "Q3",
            "answer_segment": "",
            "parsed_value": None,
            "correct_math_verify": False,
            "correct": False,
        },
    ]
    eval_json_path = tmp_path / "eval.json"
    write_json(eval_json_path, eval_records)

    collect_output_dir = tmp_path / "collect_summary"
    collect_module.main(
        [
            "--run",
            f"baseline={eval_json_path}",
            "--dataset",
            str(eval_dataset_path),
            "--train-parquet",
            str(train_path),
            "--test-parquet",
            str(test_path),
            "--output-dir",
            str(collect_output_dir),
        ]
    )

    collect_summary = json.loads((collect_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert collect_summary["train_test_overlap_question_count"] == 1
    baseline = collect_summary["runs"][0]
    assert baseline["run_label"] == "baseline"
    assert baseline["num_samples"] == 4
    assert baseline["num_unique_questions"] == 3
    assert baseline["correct"] == 0.5
    assert baseline["boxed_rate"] == 0.5
    assert baseline["numeric_parse_success"] == 0.5
    assert baseline["dedup_correct"] == 0.5
    assert baseline["overlap30_count"] == 2
    assert baseline["overlap30_correct"] == 0.5

    rollout_summary = {
        "runs": [
            {
                "run_label": "baseline",
                "mean_group_mean_mean": 0.125,
                "mean_group_std_mean": 0.1,
                "mean_best_of_n_mean": 0.25,
                "mean_all_wrong_group_rate": 0.75,
                "mean_all_correct_group_rate": 0.05,
                "mean_mixed_group_rate": 0.20,
            }
        ]
    }
    rollout_summary_path = tmp_path / "rollout_summary.json"
    write_json(rollout_summary_path, rollout_summary)

    metadata = {
        "run_label": "baseline",
        "status": "success",
        "duration_seconds": 7200,
        "total_training_steps": 100,
    }
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_json(metadata_dir / "baseline.json", metadata)

    root_cause_output_dir = tmp_path / "root_cause"
    summary_module.main(
        [
            "--offline-summary",
            str(collect_output_dir / "summary.json"),
            "--rollout-summary",
            str(rollout_summary_path),
            "--metadata-dir",
            str(metadata_dir),
            "--output-dir",
            str(root_cause_output_dir),
        ]
    )

    root_cause_summary = json.loads(
        (root_cause_output_dir / "root_cause_summary.json").read_text(encoding="utf-8")
    )
    row = root_cause_summary["runs"][0]
    assert row["run_label"] == "baseline"
    assert row["offline_correct"] == 0.5
    assert row["all_wrong_group_rate"] == 0.75
    assert row["best_of_n_mean"] == 0.25
    assert row["steps_per_hour"] == 50.0
    assert row["hours_100_steps"] == 2.0
