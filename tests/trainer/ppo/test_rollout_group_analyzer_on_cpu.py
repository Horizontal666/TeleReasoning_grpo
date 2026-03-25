import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from verl.trainer.ppo.ray_trainer import _build_generation_dump_records


ANALYZER_PATH = Path(
    "/workspace/wbh/202509_InferenceModel/Inference/verl/examples/grpo_TeleInfer/analyze_rollout_groups.py"
)


def load_analyzer_module():
    spec = importlib.util.spec_from_file_location("rollout_group_analyzer", ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_group_records(step, prompt, gts, scores, uid=None):
    records = []
    for idx, score in enumerate(scores):
        record = {
            "input": prompt,
            "output": f"{prompt}-out-{idx}",
            "gts": gts,
            "score": score,
            "acc": score,
            "step": step,
        }
        if uid is not None:
            record["uid"] = uid
        records.append(record)
    return records


def read_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_generation_dump_records_includes_uid():
    records = _build_generation_dump_records(
        inputs=["prompt-a", "prompt-b"],
        outputs=["out-a", "out-b"],
        gts=["1", "0"],
        scores=[1.0, 0.0],
        step=5,
        reward_extra_infos_dict={"acc": [1.0, 0.0]},
        extra_columns={"uid": ["uid-1", "uid-2"], "bad": ["only-one"]},
    )

    assert records == [
        {
            "input": "prompt-a",
            "output": "out-a",
            "gts": "1",
            "score": 1.0,
            "step": 5,
            "acc": 1.0,
            "uid": "uid-1",
        },
        {
            "input": "prompt-b",
            "output": "out-b",
            "gts": "0",
            "score": 0.0,
            "step": 5,
            "acc": 0.0,
            "uid": "uid-2",
        },
    ]


def test_build_generation_dump_records_converts_numpy_scalars_to_python():
    records = _build_generation_dump_records(
        inputs=["prompt-a"],
        outputs=["out-a"],
        gts=["1"],
        scores=[np.float32(1.0)],
        step=5,
        reward_extra_infos_dict={
            "acc": [np.bool_(True)],
            "format_correct": [np.bool_(False)],
        },
        extra_columns={"uid": ["uid-1"]},
    )

    assert records == [
        {
            "input": "prompt-a",
            "output": "out-a",
            "gts": "1",
            "score": 1.0,
            "step": 5,
            "acc": True,
            "format_correct": False,
            "uid": "uid-1",
        }
    ]
    json.dumps(records[0], ensure_ascii=False)


def test_rollout_group_analyzer_exact_uid_mode_and_comparison(tmp_path):
    analyzer = load_analyzer_module()
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    output_dir = tmp_path / "analysis"

    write_jsonl(
        run_a / "1.jsonl",
        make_group_records(1, "prompt-1", "1", [1, 1, 0, 1], uid="u1")
        + make_group_records(1, "prompt-2", "0", [0, 0, 0, 0], uid="u2")
        + make_group_records(1, "prompt-3", "1", [1, 1, 1, 1], uid="u3"),
    )
    write_jsonl(
        run_a / "2.jsonl",
        make_group_records(2, "prompt-4", "1", [0, 0, 1, 0], uid="u4")
        + make_group_records(2, "prompt-5", "0", [0, 0, 0, 0], uid="u5")
        + make_group_records(2, "prompt-6", "1", [1, 1, 0, 1], uid="u6"),
    )

    write_jsonl(
        run_b / "1.jsonl",
        make_group_records(1, "prompt-1", "1", [1, 1, 1, 1], uid="u1")
        + make_group_records(1, "prompt-2", "0", [0, 0, 0, 0], uid="u2")
        + make_group_records(1, "prompt-3", "1", [1, 1, 1, 1], uid="u3"),
    )
    write_jsonl(
        run_b / "2.jsonl",
        make_group_records(2, "prompt-4", "1", [1, 1, 1, 0], uid="u4")
        + make_group_records(2, "prompt-5", "0", [0, 0, 0, 0], uid="u5")
        + make_group_records(2, "prompt-6", "1", [1, 1, 1, 1], uid="u6"),
    )

    analyzer.main(
        [
            "--run",
            f"before={run_a}",
            "--run",
            f"after={run_b}",
            "--output-dir",
            str(output_dir),
            "--expected-group-size",
            "4",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["runs"]) == 2
    assert summary["runs"][0]["group_mode_counts"] == {"uid": 2}
    assert summary["runs"][0]["total_ambiguous_groups"] == 0
    assert len(summary["comparisons"]) == 1
    assert summary["comparisons"][0]["common_step_count"] == 2
    assert summary["comparisons"][0]["metric_delta_means"]["group_mean_mean"] > 0

    step_rows = read_csv_rows(output_dir / "steps.csv")
    assert len(step_rows) == 4
    assert all(row["group_mode"] == "uid" for row in step_rows)
    assert all(row["exact_grouping"] == "True" for row in step_rows)

    group_rows = read_csv_rows(output_dir / "groups.csv")
    assert len(group_rows) == 12
    assert all(row["is_ambiguous"] == "False" for row in group_rows)


def test_rollout_group_analyzer_flags_ambiguous_prompt_groups(tmp_path):
    analyzer = load_analyzer_module()
    run_dir = tmp_path / "prompt_only"
    output_dir = tmp_path / "analysis_prompt"

    records = (
        make_group_records(1, "prompt-duplicate", "1", [1, 1, 1, 1], uid=None)
        + make_group_records(1, "prompt-duplicate", "1", [0, 0, 0, 0], uid=None)
        + make_group_records(1, "prompt-other", "0", [0, 0, 0, 0], uid=None)
    )
    write_jsonl(run_dir / "1.jsonl", records)

    analyzer.main(
        [
            "--run",
            f"fallback={run_dir}",
            "--output-dir",
            str(output_dir),
            "--expected-group-size",
            "4",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["runs"][0]["group_mode_counts"] == {"prompt": 1}
    assert summary["runs"][0]["total_ambiguous_groups"] == 1

    step_rows = read_csv_rows(output_dir / "steps.csv")
    assert len(step_rows) == 1
    assert step_rows[0]["group_mode"] == "prompt"
    assert step_rows[0]["exact_grouping"] == "False"
    assert step_rows[0]["ambiguous_group_count"] == "1"

    ambiguous_rows = read_csv_rows(output_dir / "ambiguous_groups.csv")
    assert len(ambiguous_rows) == 1
    assert ambiguous_rows[0]["is_ambiguous"] == "True"
    assert ambiguous_rows[0]["ambiguity_reason"] == "merged_duplicate_prompt"
    assert ambiguous_rows[0]["group_size"] == "8"
