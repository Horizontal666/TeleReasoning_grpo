from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

from utils.evals import EvalItem, load_eval_spec, summarize


def parse_args() -> argparse.Namespace:
    """CLI for launching a Hugging Face model evaluation over supported datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name or local path.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer name or path.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/3GPP-TSG/3gpp_class_eval.json"),
        help="Path to the evaluation dataset (auto-detected format).",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for predictions and metrics.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device identifier when not using device map (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Transformers device_map hint (e.g. 'auto', 'balanced'); enables multi-GPU sharding.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for remote inference (e.g. http://127.0.0.1:6005/v1).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key for remote inference servers (defaults to empty string).",
    )
    parser.add_argument("--dtype", type=str, default=None, help="Optional torch dtype (e.g. float16, bfloat16).")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of evaluation samples.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 disables sampling.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling value.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling value.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty for generation.")
    parser.add_argument("--log-samples", type=int, default=10, help="Print interim metrics every N samples.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders.")
    return parser.parse_args()


def resolve_device(device_arg: Optional[str]) -> torch.device:
    """Pick the runtime device, defaulting to CUDA when available."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_dtype(name: Optional[str]):
    """Resolve a dtype string into a torch dtype object."""
    if not name:
        return None
    key = name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return mapping[key]


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load tokenizer/config/model trio while respecting trust_remote_code."""
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model_cls = AutoModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else AutoModelForCausalLM

    model_kwargs: Dict[str, object] = {}
    dtype = parse_dtype(args.dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = model_cls.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, **model_kwargs)
    return model, tokenizer, config


def build_generation_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    """Collect generation kwargs and enable sampling only when requested."""
    gen_kwargs: Dict[str, object] = {"max_new_tokens": args.max_new_tokens}
    if args.repetition_penalty:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.temperature and args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            gen_kwargs["top_k"] = args.top_k
    else:
        gen_kwargs["do_sample"] = False
    return gen_kwargs


def create_remote_client(args: argparse.Namespace):
    """Instantiate an OpenAI-compatible client when --api-base is provided."""
    if not args.api_base:
        return None
    if OpenAI is None:
        raise RuntimeError(
            "Remote inference requires the 'openai' package. Install with: pip install openai>=1.42"
        )
    return OpenAI(base_url=args.api_base, api_key=args.api_key or "EMPTY")


def chat_complete_remote(client, args: argparse.Namespace, prompt: str) -> str:
    """Send a prompt to the remote inference endpoint and return the generated text."""
    extra_body: Dict[str, object] = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        extra_body["repetition_penalty"] = args.repetition_penalty

    request_kwargs: Dict[str, object] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
    }
    if args.top_p is not None:
        request_kwargs["top_p"] = args.top_p
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**request_kwargs)
    if not response.choices:
        return ""
    choice = response.choices[0]
    if choice.message and choice.message.content:
        return choice.message.content
    return ""


def resolve_generation_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    """Determine which device input tensors should reside on for generation."""
    device_attr = getattr(model, "device", None)
    if device_attr is not None:
        return torch.device(device_attr)

    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        first = next(iter(hf_device_map.values()))
        if isinstance(first, (list, tuple)):
            if not first:
                return fallback
            first = first[0]
        return torch.device(first)
    return fallback


def main() -> None:
    """Execute the full evaluation pipeline and emit metrics/predictions."""
    args = parse_args()

    # Load dataset spec (prompts, parsing, scoring) based on file structure.
    spec = load_eval_spec(args.dataset)
    items: List[EvalItem] = spec.items
    if args.max_samples is not None and args.max_samples > 0:
        items = items[: args.max_samples]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    client = create_remote_client(args)
    using_remote = client is not None
    model = tokenizer = config = None
    gen_kwargs: Dict[str, object] = {}
    using_device_map = False
    device = None

    if not using_remote:
        model, tokenizer, config = load_model_and_tokenizer(args)
        using_device_map = bool(args.device_map and args.device_map.lower() != "none")
        device = resolve_device(args.device) if not using_device_map else None
        if not using_device_map:
            model.to(device)
        model.eval()
        gen_kwargs = build_generation_kwargs(args)

    results: List[Dict[str, object]] = []
    for idx, item in enumerate(items):
        prompt = item.prompt
        t0 = time.time()

        if using_remote:
            text = chat_complete_remote(client, args, prompt)
            latency = time.time() - t0
        else:
            assert tokenizer is not None and config is not None and model is not None
            encoded = tokenizer(prompt, return_tensors="pt")
            if using_device_map:
                fallback = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                target_device = resolve_generation_device(model, fallback)
            else:
                target_device = device
            encoded = {k: v.to(target_device) for k, v in encoded.items()}

            with torch.inference_mode():
                output_ids = model.generate(**encoded, **gen_kwargs)
            latency = time.time() - t0

            generated = output_ids[0]
            if not getattr(config, "is_encoder_decoder", False):
                input_len = encoded["input_ids"].shape[-1]
                generated = generated[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        pred_value = spec.parse_prediction(text, item)
        label_str = spec.format_label(item)
        pred_str = spec.format_pred(pred_value, item)
        correct = spec.is_correct(item, pred_value)

        result = {
            "id": item.id,
            "label": label_str,
            "pred": pred_str,
            "correct": bool(correct),
            "prompt": prompt,
            "response_text": text,
            "latency": latency,
            "dataset": spec.name,
        }
        if item.metadata:
            result["metadata"] = item.metadata
        extras = spec.extras(item, pred_value)
        if extras:
            for key, value in extras.items():
                if value is not None:
                    result[key] = value
        results.append(result)

        if args.log_samples and (idx + 1) % args.log_samples == 0:
            interim = summarize(results, is_interim=True, per_class=spec.per_class)
            print(f"=== Interim summary @ {idx + 1} samples ===")
            print(json.dumps(interim, indent=2))
            print("-" * 60)

    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = summarize(results, per_class=spec.per_class)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
