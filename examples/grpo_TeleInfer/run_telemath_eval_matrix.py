#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
EVAL_SCRIPT = REPO_ROOT / "TeleReasoning_Eval_git/old_vllmUnify/telemath_eval_vllm_mathverify.py"
COLLECT_SCRIPT = REPO_ROOT / "Inference/verl/examples/grpo_TeleInfer/collect_telemath_eval_metrics.py"
DEFAULT_DATASET = REPO_ROOT / "data/eval_benchmark_CT/telemath/telemath_chat_template.json"
DEFAULT_TRAIN_PARQUET = REPO_ROOT / "data/GRPO/telemath_self_gen_v0/train.parquet"
DEFAULT_TEST_PARQUET = REPO_ROOT / "data/GRPO/telemath_self_gen_v0/test.parquet"
DEFAULT_RAW_OUTPUT_DIR = REPO_ROOT / "outputs/eval/root_cause_baseline/raw"
DEFAULT_SUMMARY_OUTPUT_DIR = REPO_ROOT / "outputs/eval/root_cause_baseline/summary"

DEFAULT_MODEL_MATRIX = {
    "sft_merged": REPO_ROOT / "outputs/model_FT_merged/Qwen2.5-7B-Instruct-telemath_self_gen_v0_peft_checkpoint-10",
    "grpo_0302_step100": REPO_ROOT
    / "outputs/grpo/checkpoints/TeleReasoning_GRPO/Qwen2.5-7B-Instruct-telemath_self_gen_v0_peft_checkpoint-10_telemath_self_gen_v0_0302_flowrl/global_step_100/actor/huggingface",
    "grpo_0302_step150": REPO_ROOT
    / "outputs/grpo/checkpoints/TeleReasoning_GRPO/Qwen2.5-7B-Instruct-telemath_self_gen_v0_peft_checkpoint-10_telemath_self_gen_v0_0302_flowrl/global_step_150/actor/huggingface",
    "grpo_0313_step100": REPO_ROOT
    / "outputs/grpo/checkpoints/TeleReasoning_GRPO/Qwen2.5-7B-Instruct-telemath_self_gen_v0_peft_checkpoint-10_telemath_self_gen_v0_0313_flowrl/global_step_100/actor/huggingface",
}


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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the fixed TeleMath baseline evaluation matrix and summarize the results."
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_labeled_path,
        default=[],
        help="Run label and HF model directory, e.g. sft=/abs/path/model",
    )
    parser.add_argument(
        "--eval-json",
        action="append",
        type=parse_labeled_path,
        default=[],
        help="Existing raw eval JSON to include without rerunning inference.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--train-parquet", type=Path, default=DEFAULT_TRAIN_PARQUET)
    parser.add_argument("--test-parquet", type=Path, default=DEFAULT_TEST_PARQUET)
    parser.add_argument("--raw-output-dir", type=Path, default=DEFAULT_RAW_OUTPUT_DIR)
    parser.add_argument("--summary-output-dir", type=Path, default=DEFAULT_SUMMARY_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rel-tol", type=float, default=0.01)
    parser.add_argument("--abs-tol", type=float, default=0.01)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun inference even if the raw eval JSON already exists.",
    )
    return parser


def run_eval(label, model_dir, output_path, args):
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--model-dir",
        str(model_dir),
        "--run-name",
        label,
        "--dataset",
        str(args.dataset),
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--rel-tol",
        str(args.rel_tol),
        "--abs-tol",
        str(args.abs_tol),
        "--output-path",
        str(output_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def summarize_runs(run_json_paths, args):
    cmd = [
        sys.executable,
        str(COLLECT_SCRIPT),
        "--dataset",
        str(args.dataset),
        "--train-parquet",
        str(args.train_parquet),
        "--test-parquet",
        str(args.test_parquet),
        "--output-dir",
        str(args.summary_output_dir),
    ]
    for label, path in run_json_paths:
        cmd.extend(["--run", f"{label}={path}"])
    print("Summarizing:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    model_entries = list(args.model)
    if not model_entries and not args.eval_json:
        model_entries = list(DEFAULT_MODEL_MATRIX.items())

    args.raw_output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_output_dir.mkdir(parents=True, exist_ok=True)

    run_json_paths = []
    for label, model_dir in model_entries:
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing model directory for {label}: {model_dir}")
        output_path = args.raw_output_dir / f"{label}_telemath_eval_vllm_mathverify.json"
        if output_path.exists() and not args.force:
            print(f"Skipping existing eval for {label}: {output_path}")
        else:
            run_eval(label, model_dir, output_path, args)
        run_json_paths.append((label, output_path))

    for label, eval_json_path in args.eval_json:
        if not eval_json_path.exists():
            raise FileNotFoundError(f"Missing eval JSON for {label}: {eval_json_path}")
        run_json_paths.append((label, eval_json_path))

    if not run_json_paths:
        raise ValueError("No models or eval JSONs were provided.")

    summarize_runs(run_json_paths, args)


if __name__ == "__main__":
    main()
