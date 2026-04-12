#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    class _TqdmFallback:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable, *args, **kwargs)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value}")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def normalize_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_obj(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            return normalize_obj(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            return normalize_obj(value.item())
        except Exception:
            pass
    return value


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}, got {type(payload).__name__}")
            records.append(normalize_obj(payload))
    return records


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return [normalize_obj(item) for item in payload]
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return [normalize_obj(item) for item in payload["data"]]
        raise ValueError(f"Unsupported JSON structure in {path}; expected a list or a dict with 'data'.")
    raise ValueError(f"Unsupported JSON root in {path}: {type(payload).__name__}")


def load_parquet_records(path: Path) -> list[dict[str, Any]]:
    loaders = []

    try:
        import pandas as pd  # type: ignore

        loaders.append(("pandas", lambda: pd.read_parquet(path).to_dict(orient="records")))
    except Exception:
        pass

    try:
        from datasets import load_dataset  # type: ignore

        def _load_with_datasets() -> list[dict[str, Any]]:
            dataset = load_dataset("parquet", data_files=str(path), split="train")
            return [normalize_obj(dataset[i]) for i in range(len(dataset))]

        loaders.append(("datasets", _load_with_datasets))
    except Exception:
        pass

    try:
        import pyarrow.parquet as pq  # type: ignore

        loaders.append(("pyarrow", lambda: pq.read_table(path).to_pylist()))
    except Exception:
        pass

    errors: list[str] = []
    for name, loader in loaders:
        try:
            return [normalize_obj(item) for item in loader()]
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    joined_errors = "; ".join(errors) if errors else "no parquet loader is available"
    raise RuntimeError(
        "Unable to read parquet dataset. Install one of `pandas`, `datasets`, or `pyarrow` in the runtime env. "
        f"Details: {joined_errors}"
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl_records(path)
    if suffix == ".json":
        return load_json_records(path)
    if suffix == ".parquet":
        return load_parquet_records(path)
    raise ValueError(f"Unsupported dataset format for {path}. Expected .parquet, .jsonl, or .json")


def load_external_object(module_path: Path, object_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise AttributeError(f"{object_name} was not found in {module_path}") from exc


def load_reward_fn(module_path: Path, fn_name: str):
    if not module_path.is_file():
        raise FileNotFoundError(f"Reward file not found: {module_path}")
    return load_external_object(module_path, fn_name)


def filter_supported_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in signature.parameters}


def extract_content_text(content: Any) -> str:
    content = normalize_obj(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)


def fallback_chat_render(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        parts.append(f"{role}\n{extract_content_text(message.get('content', ''))}")
    parts.append("assistant\n")
    return "\n".join(parts)


def render_prompt(tokenizer, prompt_obj: Any, add_generation_prompt: bool = True) -> tuple[str, list[int]]:
    prompt_obj = normalize_obj(prompt_obj)

    if isinstance(prompt_obj, list):
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_obj,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            prompt_text = fallback_chat_render(prompt_obj)

        try:
            prompt_token_ids = tokenizer.apply_chat_template(
                prompt_obj,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    else:
        prompt_text = str(prompt_obj)
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    return prompt_text, list(prompt_token_ids)


def extract_question(row: dict[str, Any], prompt_key: str) -> str:
    extra_info = normalize_obj(row.get("extra_info") or {})
    if isinstance(extra_info, dict) and extra_info.get("question"):
        return str(extra_info["question"])

    prompt_value = normalize_obj(row.get(prompt_key))
    if isinstance(prompt_value, list):
        for message in prompt_value:
            if isinstance(message, dict) and message.get("role") == "user":
                text = extract_content_text(message.get("content", ""))
                if text:
                    return text
        if prompt_value:
            first = prompt_value[0]
            if isinstance(first, dict):
                return extract_content_text(first.get("content", ""))
    if prompt_value is not None:
        return str(prompt_value)
    return ""


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def score_to_acc(score_payload: Any, numeric_score: float) -> float:
    if isinstance(score_payload, dict) and "acc" in score_payload:
        return 1.0 if bool(score_payload["acc"]) else 0.0
    return 1.0 if numeric_score > 0 else 0.0


def classify_difficulty(avg_score: float, too_easy_threshold: float, too_hard_threshold: float) -> str:
    if avg_score >= too_easy_threshold:
        return "too_easy"
    if avg_score <= too_hard_threshold:
        return "too_hard"
    return "keep"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default))
            handle.write("\n")


def parse_dtype(dtype_name: str | None):
    if dtype_name is None or dtype_name == "auto":
        return None
    import torch  # type: ignore

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def resolve_hf_target_device(model):
    import torch  # type: ignore

    device_attr = getattr(model, "device", None)
    if device_attr is not None:
        return torch.device(device_attr)

    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        first = next(iter(hf_device_map.values()))
        if isinstance(first, (list, tuple)):
            first = first[0]
        if isinstance(first, str):
            return torch.device(first)
        if isinstance(first, int):
            return torch.device(f"cuda:{first}")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def determine_backend(backend: str) -> str:
    if backend != "auto":
        return backend

    try:
        import vllm  # noqa: F401

        return "vllm"
    except Exception:
        pass

    try:
        import transformers  # noqa: F401

        return "hf"
    except Exception:
        pass

    raise RuntimeError("Could not detect an available backend. Install `vllm` or `transformers` in the runtime env.")


@dataclass
class PreparedSample:
    sample_index: int
    uid: str
    prompt_text: str
    prompt_token_ids: list[int]
    prompt_tokens: int
    question: str
    ground_truth: Any
    style: Any
    data_source: str
    extra_info: dict[str, Any]
    reward_model: dict[str, Any]
    raw_sample: dict[str, Any]


def prepare_samples(records: list[dict[str, Any]], tokenizer, args) -> tuple[list[PreparedSample], list[dict[str, Any]]]:
    prepared: list[PreparedSample] = []
    skipped: list[dict[str, Any]] = []

    for sample_index, raw_row in enumerate(
        tqdm(records, desc="Preparing samples", unit="sample", dynamic_ncols=True), start=0
    ):
        row = normalize_obj(raw_row)
        prompt_text, prompt_token_ids = render_prompt(tokenizer, row.get(args.prompt_key))
        prompt_tokens = len(prompt_token_ids)
        uid = str(row.get(args.uid_key) or uuid.uuid4())
        question = extract_question(row, args.prompt_key)

        if prompt_tokens > args.max_prompt_length and args.filter_overlong_prompts:
            skipped.append(
                {
                    "sample_index": sample_index,
                    "uid": uid,
                    "question": question,
                    "prompt_tokens": prompt_tokens,
                    "max_prompt_length": args.max_prompt_length,
                    "reason": "prompt_too_long",
                    "raw_sample": row,
                }
            )
            continue

        if prompt_tokens > args.max_prompt_length and not args.filter_overlong_prompts:
            if args.truncation == "left":
                prompt_token_ids = prompt_token_ids[-args.max_prompt_length :]
            else:
                prompt_token_ids = prompt_token_ids[: args.max_prompt_length]
            prompt_tokens = len(prompt_token_ids)
            prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=True)

        reward_model = normalize_obj(row.get(args.reward_model_key) or {})
        extra_info = normalize_obj(row.get(args.extra_info_key) or {})
        style = reward_model.get("style") if isinstance(reward_model, dict) else None
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        data_source = str(row.get(args.data_source_key) or "unknown")

        prepared.append(
            PreparedSample(
                sample_index=sample_index,
                uid=uid,
                prompt_text=prompt_text,
                prompt_token_ids=prompt_token_ids,
                prompt_tokens=prompt_tokens,
                question=question,
                ground_truth=ground_truth,
                style=style,
                data_source=data_source,
                extra_info=extra_info if isinstance(extra_info, dict) else {},
                reward_model=reward_model if isinstance(reward_model, dict) else {},
                raw_sample=row,
            )
        )

    return prepared, skipped


def generate_with_vllm_compat(llm, batch: list[PreparedSample], sampling_params):
    prompt_inputs = [
        {
            "prompt_token_ids": sample.prompt_token_ids,
            "prompt": sample.prompt_text,
        }
        for sample in batch
    ]
    prompt_token_ids = [sample.prompt_token_ids for sample in batch]
    prompt_texts = [sample.prompt_text for sample in batch]

    base_kwargs_list = [
        {"sampling_params": sampling_params, "use_tqdm": False},
        {"sampling_params": sampling_params},
    ]

    attempts: list[tuple[str, Any]] = [
        (
            "prompt_inputs_positional",
            lambda kwargs: llm.generate(prompt_inputs, **kwargs),
        ),
        (
            "prompt_inputs_keyword",
            lambda kwargs: llm.generate(prompts=prompt_inputs, **kwargs),
        ),
        (
            "legacy_prompt_token_ids_only",
            lambda kwargs: llm.generate(prompt_token_ids=prompt_token_ids, **kwargs),
        ),
        (
            "legacy_prompts_and_prompt_token_ids",
            lambda kwargs: llm.generate(
                prompts=prompt_texts,
                prompt_token_ids=prompt_token_ids,
                **kwargs,
            ),
        ),
    ]

    errors: list[str] = []
    for attempt_name, attempt_fn in attempts:
        for call_kwargs in base_kwargs_list:
            suffix = "+use_tqdm" if "use_tqdm" in call_kwargs else "-use_tqdm"
            try:
                return attempt_fn(call_kwargs)
            except TypeError as exc:
                errors.append(f"{attempt_name}{suffix}: {exc}")
                continue
            except ValueError as exc:
                errors.append(f"{attempt_name}{suffix}: {exc}")
                continue

    joined_errors = "\n".join(errors)
    raise RuntimeError(
        "Unable to call vLLM LLM.generate() with any supported prompt format.\n"
        "Tried new prompt-input API and legacy prompt_token_ids API.\n"
        f"Errors:\n{joined_errors}"
    )


def _dedupe_vllm_candidates(candidates: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    deduped: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for name, config in candidates:
        fingerprint = tuple(sorted(config.items()))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append((name, config))
    return deduped


def build_vllm_init_candidates(base_kwargs: dict[str, Any], rollout_n: int) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = [("requested", dict(base_kwargs))]

    no_budget = dict(base_kwargs)
    no_budget.pop("max_num_batched_tokens", None)
    candidates.append(("drop_token_budget", no_budget))

    current_budget = base_kwargs.get("max_num_batched_tokens")
    if isinstance(current_budget, int):
        for budget in (65536, 32768, 16384):
            if current_budget > budget:
                candidate = dict(base_kwargs)
                candidate["max_num_batched_tokens"] = budget
                candidates.append((f"token_budget_{budget}", candidate))

                candidate_no_budget = dict(candidate)
                candidate_no_budget.pop("max_num_batched_tokens", None)
                candidates.append((f"token_budget_{budget}_drop_budget", candidate_no_budget))

    current_seqs = int(base_kwargs.get("max_num_seqs", rollout_n))
    for seqs in (256, 192, 128, 96, 64):
        if seqs < rollout_n:
            continue
        if current_seqs > seqs:
            candidate = dict(no_budget)
            candidate["max_num_seqs"] = seqs
            candidates.append((f"max_num_seqs_{seqs}", candidate))

    current_gpu_util = float(base_kwargs.get("gpu_memory_utilization", 0.0))
    for gpu_util in (0.82, 0.80, 0.75):
        if current_gpu_util > gpu_util:
            candidate = dict(no_budget)
            candidate["gpu_memory_utilization"] = gpu_util
            candidates.append((f"gpu_util_{gpu_util:.2f}", candidate))

            seqs = min(int(candidate.get("max_num_seqs", rollout_n)), 128)
            if seqs >= rollout_n:
                candidate_combo = dict(candidate)
                candidate_combo["max_num_seqs"] = seqs
                candidates.append((f"gpu_util_{gpu_util:.2f}_seqs_{seqs}", candidate_combo))

    return _dedupe_vllm_candidates(candidates)


def init_vllm_with_fallback(base_kwargs: dict[str, Any], rollout_n: int, requested_batch_size: int):
    from vllm import LLM  # type: ignore

    candidates = build_vllm_init_candidates(base_kwargs, rollout_n)
    errors: list[str] = []

    for attempt_name, candidate_kwargs in candidates:
        effective_max_num_seqs = int(candidate_kwargs.get("max_num_seqs", rollout_n))
        effective_batch_size = max(1, min(requested_batch_size, effective_max_num_seqs // rollout_n))
        candidate_summary = {
            "gpu_memory_utilization": candidate_kwargs.get("gpu_memory_utilization"),
            "max_num_seqs": candidate_kwargs.get("max_num_seqs"),
            "max_num_batched_tokens": candidate_kwargs.get("max_num_batched_tokens", "default"),
            "effective_batch_size": effective_batch_size,
        }
        print(f"[vllm:init] trying {attempt_name}: {candidate_summary}", flush=True)
        try:
            llm = LLM(**filter_supported_kwargs(LLM.__init__, candidate_kwargs))
            print(f"[vllm:init] success with {attempt_name}", flush=True)
            return llm, effective_batch_size, candidate_kwargs
        except Exception as exc:
            errors.append(f"{attempt_name}: {type(exc).__name__}: {exc}")
            print(f"[vllm:init] failed with {attempt_name}: {type(exc).__name__}: {exc}", flush=True)

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"Failed to initialize vLLM after trying fallback configs.\n{joined_errors}")


def generate_with_vllm(samples: list[PreparedSample], args) -> list[list[str]]:
    from vllm import LLM, SamplingParams  # type: ignore

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enable_prefix_caching": args.enable_prefix_caching,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "load_format": args.load_format,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.dtype and args.dtype != "auto":
        llm_kwargs["dtype"] = args.dtype

    llm, effective_batch_size, resolved_llm_kwargs = init_vllm_with_fallback(
        llm_kwargs,
        rollout_n=args.rollout_n,
        requested_batch_size=args.batch_size,
    )

    sampling_kwargs: dict[str, Any] = {
        "n": args.rollout_n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_response_length,
    }
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed
    sampling_params = SamplingParams(**filter_supported_kwargs(SamplingParams.__init__, sampling_kwargs))

    all_outputs: list[list[str]] = []
    total = len(samples)
    num_batches = math.ceil(total / effective_batch_size) if total else 0

    batch_iter = tqdm(range(num_batches), desc="Generating with vLLM", unit="batch", dynamic_ncols=True)
    for batch_idx in batch_iter:
        start = batch_idx * effective_batch_size
        end = min(total, start + effective_batch_size)
        batch = samples[start:end]
        batch_iter.set_postfix(
            samples=len(batch),
            eff_batch=effective_batch_size,
            max_num_seqs=resolved_llm_kwargs.get("max_num_seqs"),
            refresh=False,
        )
        request_outputs = generate_with_vllm_compat(llm, batch, sampling_params)

        for request_output in request_outputs:
            request_texts = [candidate.text for candidate in request_output.outputs]
            if len(request_texts) != args.rollout_n:
                raise RuntimeError(
                    f"Expected {args.rollout_n} rollouts per sample, but got {len(request_texts)} from vLLM."
                )
            all_outputs.append(request_texts)

    return all_outputs


def generate_with_hf(samples: list[PreparedSample], args) -> list[list[str]]:
    import torch  # type: ignore
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    dtype = parse_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model_cls = AutoModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = model_cls.from_pretrained(args.model, **model_kwargs)
    if not args.device_map:
        model.to(torch.device(args.device))
    model.eval()

    target_device = resolve_hf_target_device(model)
    all_outputs: list[list[str]] = []
    total = len(samples)
    num_batches = math.ceil(total / args.batch_size) if total else 0

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_response_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
        if args.top_k >= 0:
            generation_kwargs["top_k"] = args.top_k
    else:
        generation_kwargs["do_sample"] = False

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    batch_iter = tqdm(range(num_batches), desc="Generating with HF", unit="batch", dynamic_ncols=True)
    for batch_idx in batch_iter:
        start = batch_idx * args.batch_size
        end = min(total, start + args.batch_size)
        batch = samples[start:end]
        batch_iter.set_postfix(samples=len(batch), refresh=False)

        repeated_prompt_ids: list[list[int]] = []
        for sample in batch:
            for _ in range(args.rollout_n):
                repeated_prompt_ids.append(sample.prompt_token_ids)

        encoded = tokenizer.pad(
            {"input_ids": repeated_prompt_ids},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(target_device) for key, value in encoded.items()}

        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_kwargs)

        generated_ids = output_ids
        if not getattr(config, "is_encoder_decoder", False):
            prompt_width = encoded["input_ids"].shape[1]
            generated_ids = output_ids[:, prompt_width:]

        flat_outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        if len(flat_outputs) != len(batch) * args.rollout_n:
            raise RuntimeError(
                f"Expected {len(batch) * args.rollout_n} generated outputs, got {len(flat_outputs)} from HF."
            )

        for sample_offset in range(len(batch)):
            first = sample_offset * args.rollout_n
            last = first + args.rollout_n
            all_outputs.append(flat_outputs[first:last])

    return all_outputs


def normalize_score_payload(score_payload: Any) -> tuple[float, dict[str, Any]]:
    if isinstance(score_payload, dict):
        payload = normalize_obj(score_payload)
        if "score" not in payload:
            raise ValueError("Reward payload dict must contain a 'score' key.")
        return float(payload["score"]), payload
    return float(score_payload), {}


def call_reward_fn(reward_fn, sample: PreparedSample, output_text: str) -> Any:
    kwargs = {
        "solution_str": output_text,
        "ground_truth": sample.ground_truth,
        "style": sample.style,
        "data_source": sample.data_source,
        "extra_info": sample.extra_info,
    }
    try:
        return reward_fn(**kwargs)
    except TypeError:
        return reward_fn(sample.data_source, output_text, sample.ground_truth)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single pre-training rollout pass, score each rollout, and bucket samples by difficulty."
    )
    parser.add_argument("--model", required=True, help="Model path or HF identifier.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path. Defaults to --model.")
    parser.add_argument("--data-path", required=True, help="Dataset path (.parquet/.jsonl/.json).")
    parser.add_argument("--output-dir", required=True, help="Directory to write rollout outputs and summaries.")
    parser.add_argument("--reward-path", required=True, help="Python file that defines the reward function.")
    parser.add_argument("--reward-name", required=True, help="Reward function name in --reward-path.")
    parser.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enable-chunked-prefill", type=str2bool, default=True)
    parser.add_argument("--enable-prefix-caching", type=str2bool, default=True)
    parser.add_argument("--enforce-eager", type=str2bool, default=False)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default=None, help="HF-only device_map; e.g. auto.")
    parser.add_argument("--device", default="cuda", help="HF-only device when --device-map is unset.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rollout-n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick probing.")
    parser.add_argument("--filter-overlong-prompts", type=str2bool, default=True)
    parser.add_argument("--truncation", choices=["left", "right"], default="left")
    parser.add_argument("--too-easy-threshold", type=float, default=0.75)
    parser.add_argument("--too-hard-threshold", type=float, default=-0.75)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--reward-model-key", default="reward_model")
    parser.add_argument("--data-source-key", default="data_source")
    parser.add_argument("--extra-info-key", default="extra_info")
    parser.add_argument("--uid-key", default="uid")
    return parser.parse_args()


def main() -> None:
    total_start_time = time.perf_counter()
    args = parse_args()

    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    reward_path = Path(args.reward_path).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    backend = determine_backend(args.backend)
    args.backend = backend
    if args.max_model_len is None:
        args.max_model_len = args.max_prompt_length + args.max_response_length

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading dataset from {data_path}", flush=True)
    data_load_start_time = time.perf_counter()
    records = load_records(data_path)
    if args.max_samples is not None and args.max_samples > 0:
        records = records[: args.max_samples]
    data_load_seconds = time.perf_counter() - data_load_start_time
    print(f"Loaded {len(records)} raw samples", flush=True)

    prepare_start_time = time.perf_counter()
    prepared_samples, skipped_samples = prepare_samples(records, tokenizer, args)
    prepare_seconds = time.perf_counter() - prepare_start_time
    print(
        f"Prepared {len(prepared_samples)} samples for rollout; skipped {len(skipped_samples)} overlong samples",
        flush=True,
    )

    if skipped_samples:
        write_jsonl(output_dir / "skipped_overlong.jsonl", skipped_samples)

    if not prepared_samples:
        summary = {
            "backend": backend,
            "model": args.model,
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "total_raw_samples": len(records),
            "prepared_samples": 0,
            "skipped_overlong": len(skipped_samples),
            "message": "No samples left after prompt filtering.",
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=json_default)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    reward_fn = load_reward_fn(reward_path, args.reward_name)

    generation_start_time = time.perf_counter()
    if backend == "vllm":
        grouped_outputs = generate_with_vllm(prepared_samples, args)
    else:
        grouped_outputs = generate_with_hf(prepared_samples, args)
    generation_seconds = time.perf_counter() - generation_start_time

    if len(grouped_outputs) != len(prepared_samples):
        raise RuntimeError(
            f"Generated outputs mismatch: expected {len(prepared_samples)} samples, got {len(grouped_outputs)}."
        )

    rollout_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    scoring_start_time = time.perf_counter()
    scoring_iter = tqdm(
        zip(prepared_samples, grouped_outputs, strict=True),
        total=len(prepared_samples),
        desc="Scoring rollouts",
        unit="sample",
        dynamic_ncols=True,
    )
    for sample, outputs in scoring_iter:
        if len(outputs) != args.rollout_n:
            raise RuntimeError(
                f"Sample {sample.sample_index} expected {args.rollout_n} outputs, but received {len(outputs)}."
            )

        per_rollout_rows: list[dict[str, Any]] = []
        rollout_scores: list[float] = []
        rollout_accs: list[float] = []

        for rollout_index, output_text in enumerate(outputs):
            score_payload = call_reward_fn(reward_fn, sample, output_text)
            numeric_score, extra_payload = normalize_score_payload(score_payload)
            rollout_scores.append(numeric_score)
            rollout_accs.append(score_to_acc(score_payload, numeric_score))

            row = {
                "sample_index": sample.sample_index,
                "rollout_index": rollout_index,
                "uid": sample.uid,
                "question": sample.question,
                "input": sample.prompt_text,
                "output": output_text,
                "gts": sample.ground_truth,
                "score": numeric_score,
                "reward": numeric_score,
                "step": 0,
                "data_source": sample.data_source,
                "prompt_tokens": sample.prompt_tokens,
            }
            for key, value in extra_payload.items():
                if key == "score":
                    continue
                row[key] = normalize_obj(value)
            per_rollout_rows.append(row)

        avg_score = safe_mean(rollout_scores)
        avg_acc = safe_mean(rollout_accs)
        difficulty = classify_difficulty(avg_score, args.too_easy_threshold, args.too_hard_threshold)

        for row in per_rollout_rows:
            row["sample_avg_score"] = avg_score
            row["sample_avg_acc"] = avg_acc
            row["difficulty"] = difficulty
            rollout_rows.append(row)

        sample_rows.append(
            {
                "sample_index": sample.sample_index,
                "uid": sample.uid,
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "data_source": sample.data_source,
                "prompt_tokens": sample.prompt_tokens,
                "rollout_n": args.rollout_n,
                "rollout_scores": rollout_scores,
                "rollout_accs": rollout_accs,
                "avg_score": avg_score,
                "avg_acc": avg_acc,
                "difficulty": difficulty,
                "raw_sample": sample.raw_sample,
            }
        )
        scoring_iter.set_postfix(avg_score=f"{avg_score:.3f}", difficulty=difficulty, refresh=False)
    scoring_seconds = time.perf_counter() - scoring_start_time

    too_easy_rows = [row for row in sample_rows if row["difficulty"] == "too_easy"]
    too_hard_rows = [row for row in sample_rows if row["difficulty"] == "too_hard"]
    keep_rows = [row for row in sample_rows if row["difficulty"] == "keep"]

    write_jsonl(output_dir / "0.jsonl", rollout_rows)
    write_jsonl(output_dir / "sample_summary.jsonl", sample_rows)
    write_jsonl(output_dir / "too_easy.jsonl", too_easy_rows)
    write_jsonl(output_dir / "too_hard.jsonl", too_hard_rows)
    write_jsonl(output_dir / "keep.jsonl", keep_rows)
    write_jsonl(output_dir / "too_easy.dataset.jsonl", [row["raw_sample"] for row in too_easy_rows])
    write_jsonl(output_dir / "too_hard.dataset.jsonl", [row["raw_sample"] for row in too_hard_rows])
    write_jsonl(output_dir / "keep.dataset.jsonl", [row["raw_sample"] for row in keep_rows])

    avg_scores = [row["avg_score"] for row in sample_rows]
    avg_accs = [row["avg_acc"] for row in sample_rows]
    total_seconds = time.perf_counter() - total_start_time
    summary = {
        "backend": backend,
        "model": args.model,
        "data_path": str(data_path),
        "reward_path": str(reward_path),
        "reward_name": args.reward_name,
        "output_dir": str(output_dir),
        "total_raw_samples": len(records),
        "prepared_samples": len(prepared_samples),
        "skipped_overlong": len(skipped_samples),
        "rollout_n": args.rollout_n,
        "total_rollouts": len(rollout_rows),
        "mean_avg_score": safe_mean(avg_scores),
        "median_avg_score": float(statistics.median(avg_scores)),
        "mean_avg_acc": safe_mean(avg_accs),
        "median_avg_acc": float(statistics.median(avg_accs)),
        "too_easy_threshold": args.too_easy_threshold,
        "too_hard_threshold": args.too_hard_threshold,
        "too_easy_count": len(too_easy_rows),
        "too_hard_count": len(too_hard_rows),
        "keep_count": len(keep_rows),
        "timing": {
            "data_load_seconds": data_load_seconds,
            "prepare_seconds": prepare_seconds,
            "generation_seconds": generation_seconds,
            "scoring_seconds": scoring_seconds,
            "total_seconds": total_seconds,
        },
        "throughput": {
            "prepared_samples_per_second": (len(prepared_samples) / total_seconds) if total_seconds > 0 else None,
            "rollouts_per_second": (len(rollout_rows) / generation_seconds) if generation_seconds > 0 else None,
            "scored_samples_per_second": (len(sample_rows) / scoring_seconds) if scoring_seconds > 0 else None,
        },
        "files": {
            "rollout_jsonl": str(output_dir / "0.jsonl"),
            "sample_summary_jsonl": str(output_dir / "sample_summary.jsonl"),
            "too_easy_jsonl": str(output_dir / "too_easy.jsonl"),
            "too_hard_jsonl": str(output_dir / "too_hard.jsonl"),
            "keep_jsonl": str(output_dir / "keep.jsonl"),
            "too_easy_dataset_jsonl": str(output_dir / "too_easy.dataset.jsonl"),
            "too_hard_dataset_jsonl": str(output_dir / "too_hard.dataset.jsonl"),
            "keep_dataset_jsonl": str(output_dir / "keep.dataset.jsonl"),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=json_default)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise
