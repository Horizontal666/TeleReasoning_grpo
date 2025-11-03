from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

WORKING_GROUP_KEY = "WORKING GROUP"
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
BOX_PATTERN = re.compile(r"\\boxed\s*\{([^{}]+)\}")
CHOICE_PATTERN = re.compile(r"C\d+", flags=re.IGNORECASE)
LETTER_PATTERN = re.compile(r"\b([A-Z])\b")


@dataclass
class EvalItem:
    id: str
    prompt: str
    label: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


def _default_extras(item: EvalItem, pred: Any) -> Dict[str, Any]:
    return {}


@dataclass
class EvalSpec:
    name: str
    items: List[EvalItem]
    parse_prediction: Callable[[str, EvalItem], Any]
    format_label: Callable[[EvalItem], str]
    format_pred: Callable[[Any, EvalItem], Optional[str]]
    is_correct: Callable[[EvalItem, Any], bool]
    per_class: bool = True
    extras: Callable[[EvalItem, Any], Dict[str, Any]] = _default_extras


def load_eval_spec(path: Path) -> EvalSpec:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return _load_parquet_spec(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            sample_keys = set(payload[0].keys())
            if {"instruction", "input", "output"} <= sample_keys:
                return _build_3gpp_spec(payload)
            if {"question", "answer", "tags"} <= sample_keys:
                return _build_telemath_spec(payload)
            if {"question", "answer"} <= sample_keys:
                return _build_telelogs_troubleshooting_spec(payload)
        raise ValueError("Unsupported list-based dataset schema.")
    if isinstance(payload, dict):
        return _build_teleqna_spec(payload)
    raise ValueError(f"Unsupported dataset format for {path}.")


# -------- Shared helpers --------

def _strip_code_fence(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[^\n]*\n", "", content)
        content = re.sub(r"\n```$", "", content)
    return content.strip()


def _load_json_fragment(text: str) -> Optional[Dict[str, Any]]:
    fragment = _strip_code_fence(text)
    if not fragment:
        return None
    if not fragment.startswith("{"):
        match = re.search(r"\{[^{}]*\}", fragment)
        fragment = match.group(0) if match else fragment
    try:
        parsed = json.loads(fragment)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _render_chat_messages(messages: Any) -> str:
    if isinstance(messages, list):
        lines: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user")).strip().capitalize()
            content = str(msg.get("content", "")).strip()
            lines.append(f"{role}: {content}" if content else role)
        return "\n\n".join(lines).strip()
    return str(messages)


def extract_working_group(text: str, valid_labels: Optional[Sequence[str]] = None) -> Optional[str]:
    if not isinstance(text, str):
        return None

    parsed = _load_json_fragment(text)
    if parsed:
        for key, value in parsed.items():
            if isinstance(value, str) and key.replace("_", " ").lower() == WORKING_GROUP_KEY.lower():
                return value.strip()

    normalized = _strip_code_fence(text).replace("\n", " ")
    if valid_labels:
        lower_norm = normalized.lower()
        for label in valid_labels:
            if label.lower() in lower_norm:
                return label

    match = re.search(r"WORKING GROUP[^A-Za-z0-9]+([A-Za-z0-9_]+)", normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_option_choice(text: str, options: Sequence[Dict[str, str]]) -> Optional[str]:
    normalized = _strip_code_fence(text)
    lower = normalized.lower()
    stripped = lower.strip()
    option_ids = [opt["id"] for opt in options]
    option_texts = [opt["text"] for opt in options]

    simple_ids = {opt_id.split()[-1] for opt_id in option_ids}
    if stripped in simple_ids:
        return f"option {stripped}"
    if stripped.startswith("option"):
        match = re.search(r"option\s*(\d+)", stripped)
        if match:
            candidate = f"option {match.group(1)}"
            if candidate in option_ids:
                return candidate

    for opt_id in option_ids:
        if opt_id in lower:
            return opt_id

    for opt_id, opt_text in zip(option_ids, option_texts):
        if opt_text.lower() and opt_text.lower() in lower:
            return opt_id

    match = re.search(r"option\s*(\d+)", lower)
    if match:
        candidate = f"option {match.group(1)}"
        if candidate in option_ids:
            return candidate

    first_line = stripped.splitlines()[0] if stripped else ""
    if first_line in simple_ids:
        return f"option {first_line}"
    return None


def extract_numeric_answer(text: str) -> Optional[float]:
    normalized = _strip_code_fence(text).replace(",", "")
    matches = NUMBER_PATTERN.findall(normalized)
    for token in reversed(matches):
        try:
            return float(token)
        except ValueError:
            continue
    return None


def extract_boxed_values(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [match.strip() for match in BOX_PATTERN.findall(text)]


def extract_after_marker(text: str, marker: str = "####") -> Optional[str]:
    if not isinstance(text, str):
        return None
    idx = text.rfind(marker)
    if idx == -1:
        return None
    candidate = text[idx + len(marker) :].strip()
    return candidate or None


def last_non_empty_line(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def normalize_expression(expr: str) -> str:
    return re.sub(r"\s+", "", expr)


def extract_troubleshooting_codes(text: str, valid_codes: Sequence[str]) -> List[str]:
    if not isinstance(text, str):
        return []
    matches: List[str] = []

    for boxed in extract_boxed_values(text):
        candidate = boxed.upper().replace(" ", "")
        if candidate in valid_codes:
            matches.append(candidate)

    for raw in CHOICE_PATTERN.findall(text):
        candidate = raw.upper().replace(" ", "")
        if candidate in valid_codes:
            matches.append(candidate)
    return matches


def extract_letter_choice(text: str, valid_letters: Sequence[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    for boxed in extract_boxed_values(text):
        candidate = boxed.strip().upper()
        if candidate in valid_letters:
            return candidate
    upper_text = text.upper()
    for letter in valid_letters:
        if re.search(rf"\bOPTION\s*{letter}\b", upper_text):
            return letter
    for letter in valid_letters:
        if re.search(rf"\bANSWER[:\s]*{letter}\b", upper_text):
            return letter
    for match in LETTER_PATTERN.findall(upper_text):
        if match in valid_letters:
            return match
    return None


def summarize(results: List[Dict[str, Any]], is_interim: bool = False, per_class: bool = True) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    acc = round(correct / total, 4) if total else 0.0
    latencies = [r["latency"] for r in results if r.get("latency") is not None]
    avg_latency = round(statistics.mean(latencies), 4) if latencies else None

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "avg_latency_sec": avg_latency,
    }

    if not is_interim and per_class:
        per_class_scores: Dict[str, List[bool]] = {}
        for r in results:
            label = r.get("label")
            if not label:
                continue
            per_class_scores.setdefault(label, []).append(bool(r.get("correct")))
        per_class_acc = {
            label: round(sum(flags) / len(flags), 4) if flags else 0.0
            for label, flags in sorted(per_class_scores.items(), key=lambda kv: kv[0].lower())
        }
        summary["per_class_accuracy"] = per_class_acc
    return summary


# -------- Dataset loaders --------

def _load_parquet_spec(path: Path) -> EvalSpec:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("Parquet datasets require pandas + pyarrow.") from exc

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        if "Parquet magic bytes not found" in str(exc):
            raise RuntimeError(
                f"{path} does not contain parquet bytes. If this repository uses Git LFS, run 'git lfs pull' first."
            ) from exc
        raise

    columns = set(df.columns)
    if {"prompt", "reward_model"}.issubset(columns):
        return _build_wireless_mathbench_spec(path, df.to_dict("records"))
    if "correct_answer" in columns and "prompt" in columns:
        return _build_wireless_mathbench_xl_spec(path, df.to_dict("records"))
    raise ValueError(f"Unrecognized parquet schema for {path}.")


# -------- 3GPP classification --------

def _normalize_3gpp_records(raw: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for idx, sample in enumerate(raw):
        if not isinstance(sample, dict):
            raise ValueError(f"3GPP sample {idx} is not an object.")
        records.append(
            {
                "id": sample.get("file_name") or sample.get("id") or f"sample_{idx:05d}",
                "instruction": str(sample.get("instruction", "")),
                "input": str(sample.get("input", "")),
                "output": str(sample.get("output", "")),
            }
        )
    return records


def _build_3gpp_prompt(sample: Dict[str, str]) -> str:
    instruction = sample["instruction"].strip()
    text = sample["input"].strip()
    if instruction and text:
        return f"{instruction}\n\n{text}"
    return instruction or text


def _build_3gpp_spec(raw: List[Dict[str, Any]]) -> EvalSpec:
    records = _normalize_3gpp_records(raw)
    label_candidates = [extract_working_group(sample["output"]) for sample in records]
    valid_labels = sorted({label for label in label_candidates if label}, key=str.lower)

    items: List[EvalItem] = []
    for sample in records:
        label = extract_working_group(sample["output"], valid_labels) or ""
        metadata = {
            "reference_output": sample["output"],
            "dataset": "3gpp_classification",
        }
        items.append(
            EvalItem(
                id=sample["id"],
                prompt=_build_3gpp_prompt(sample),
                label=label,
                metadata=metadata,
            )
        )

    def parse_prediction(text: str, _: EvalItem) -> Optional[str]:
        return extract_working_group(text, valid_labels)

    def format_label(item: EvalItem) -> str:
        return item.label or ""

    def format_pred(pred: Optional[str], _: EvalItem) -> Optional[str]:
        return pred

    def is_correct(item: EvalItem, pred: Optional[str]) -> bool:
        return bool(item.label) and pred == item.label

    return EvalSpec(
        name="3gpp_classification",
        items=items,
        parse_prediction=parse_prediction,
        format_label=format_label,
        format_pred=format_pred,
        is_correct=is_correct,
        per_class=True,
    )


# -------- TeleQnA multiple-choice --------

def _canonical_option_id(option_key: str) -> str:
    match = re.search(r"\d+", option_key)
    if match:
        return f"option {match.group(0)}"
    return option_key.strip().lower()


def _build_teleqna_spec(raw: Dict[str, Any]) -> EvalSpec:
    items: List[EvalItem] = []
    for idx, (key, sample) in enumerate(sorted(raw.items(), key=lambda kv: kv[0])):
        if not isinstance(sample, dict):
            raise ValueError(f"TeleQnA entry '{key}' is not an object.")
        options: List[Dict[str, str]] = []
        raw_options = [
            (opt_key, sample[opt_key])
            for opt_key in sample.keys()
            if opt_key.lower().startswith("option")
        ]
        raw_options.sort(key=lambda kv: int(re.search(r"\d+", kv[0]).group(0)) if re.search(r"\d+", kv[0]) else kv[0])
        for opt_key, opt_value in raw_options:
            canonical_id = _canonical_option_id(opt_key)
            options.append({"id": canonical_id, "text": str(opt_value).strip()})

        answer_text = str(sample.get("answer", "")).strip()
        answer_key = extract_option_choice(answer_text, options) if answer_text else None
        if not answer_key and ":" in answer_text:
            head = answer_text.split(":", 1)[0]
            answer_key = _canonical_option_id(head)
            if answer_key not in [opt["id"] for opt in options]:
                answer_key = None

        question_text = str(sample.get("question", "")).strip()
        prompt_lines = [
            "You are a telecom domain expert. Select the correct option id for the question.",
            "",
            f"Question: {question_text}",
            "",
            "Options:",
        ]
        for opt in options:
            prompt_lines.append(f"{opt['id']}: {opt['text']}")
        prompt_lines.extend(
            [
                "",
                "Reply with the option id only, e.g., 'option 2'.",
            ]
        )
        metadata = {
            "options": options,
            "answer_text": answer_text,
            "category": sample.get("category"),
            "explanation": sample.get("explanation"),
            "dataset": "teleqna",
        }
        items.append(
            EvalItem(
                id=sample.get("id") or key or f"teleqna_{idx:05d}",
                prompt="\n".join(prompt_lines),
                label=answer_key or "",
                metadata=metadata,
            )
        )

    def parse_prediction(text: str, item: EvalItem) -> Optional[str]:
        return extract_option_choice(text, item.metadata.get("options", []))

    def format_label(item: EvalItem) -> str:
        return item.label or ""

    def format_pred(pred: Optional[str], _: EvalItem) -> Optional[str]:
        return pred

    def is_correct(item: EvalItem, pred: Optional[str]) -> bool:
        if not item.label:
            return False
        return pred == item.label

    def extras(item: EvalItem, pred: Optional[str]) -> Dict[str, Any]:
        data = {
            "options": item.metadata.get("options"),
            "answer_explanation": item.metadata.get("explanation"),
            "category": item.metadata.get("category"),
        }
        if pred:
            for opt in item.metadata.get("options", []):
                if opt["id"] == pred:
                    data["selected_option_text"] = opt["text"]
                    break
        return data

    return EvalSpec(
        name="teleqna",
        items=items,
        parse_prediction=parse_prediction,
        format_label=format_label,
        format_pred=format_pred,
        is_correct=is_correct,
        per_class=False,
        extras=extras,
    )


# -------- TeleMath numeric QA --------

def _format_numeric(value: float) -> str:
    return format(value, ".12g")


def _build_telemath_spec(raw: List[Dict[str, Any]]) -> EvalSpec:
    items: List[EvalItem] = []
    for idx, sample in enumerate(raw):
        if not isinstance(sample, dict):
            raise ValueError(f"TeleMath sample {idx} is not an object.")
        question = str(sample.get("question", "")).strip()
        prompt = (
            "You are a telecom mathematics assistant. Solve the problem and respond with only the final numeric answer.\n\n"
            f"Question:\n{question}"
        )
        raw_answer = sample.get("answer")
        target_float: Optional[float] = None
        if isinstance(raw_answer, (int, float)):
            target_float = float(raw_answer)
        elif isinstance(raw_answer, str):
            try:
                target_float = float(raw_answer)
            except ValueError:
                target_float = None

        metadata = {
            "target_float": target_float,
            "raw_answer": raw_answer,
            "category": sample.get("category"),
            "tags": sample.get("tags"),
            "difficulty": sample.get("difficulty"),
            "dataset": "telemath",
        }
        label_str = _format_numeric(target_float) if target_float is not None else str(raw_answer)
        items.append(
            EvalItem(
                id=sample.get("id") or f"telemath_{idx:05d}",
                prompt=prompt,
                label=label_str,
                metadata=metadata,
            )
        )

    def parse_prediction(text: str, _: EvalItem) -> Optional[float]:
        return extract_numeric_answer(text)

    def format_label(item: EvalItem) -> str:
        return item.label or ""

    def format_pred(pred: Optional[float], _: EvalItem) -> Optional[str]:
        if pred is None:
            return None
        return _format_numeric(pred)

    def is_correct(item: EvalItem, pred: Optional[float]) -> bool:
        target = item.metadata.get("target_float")
        if target is None or pred is None:
            return False
        return math.isclose(pred, target, rel_tol=1e-4, abs_tol=1e-6)

    def extras(item: EvalItem, pred: Optional[float]) -> Dict[str, Any]:
        data = {
            "category": item.metadata.get("category"),
            "tags": item.metadata.get("tags"),
            "difficulty": item.metadata.get("difficulty"),
            "target_float": item.metadata.get("target_float"),
        }
        if pred is not None:
            data["pred_float"] = pred
        return data

    return EvalSpec(
        name="telemath",
        items=items,
        parse_prediction=parse_prediction,
        format_label=format_label,
        format_pred=format_pred,
        is_correct=is_correct,
        per_class=False,
        extras=extras,
    )


# -------- TeleLogs troubleshooting classification --------

def _build_telelogs_troubleshooting_spec(raw: List[Dict[str, Any]]) -> EvalSpec:
    valid_codes = sorted(
        {str(sample.get("answer", "")).strip().upper() for sample in raw if sample.get("answer")},
        key=lambda c: (c[0], int(c[1:]) if c[1:].isdigit() else c[1:])
    )

    items: List[EvalItem] = []
    for idx, sample in enumerate(raw):
        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip().upper()
        items.append(
            EvalItem(
                id=sample.get("id") or f"telelogs_{idx:05d}",
                prompt=question,
                label=answer,
                metadata={
                    "dataset": "telelogs_troubleshooting",
                    "valid_codes": valid_codes,
                },
            )
        )

    def parse_prediction(text: str, _: EvalItem) -> Optional[str]:
        matches = extract_troubleshooting_codes(text, valid_codes)
        return matches[0] if matches else None

    def format_label(item: EvalItem) -> str:
        return f"\\boxed{{{item.label}}}" if item.label else ""

    def format_pred(pred: Optional[str], _: EvalItem) -> Optional[str]:
        if pred:
            return f"\\boxed{{{pred}}}"
        return None

    def is_correct(item: EvalItem, pred: Optional[str]) -> bool:
        return bool(item.label) and pred == item.label

    def extras(item: EvalItem, pred: Optional[str]) -> Dict[str, Any]:
        return {
            "valid_codes": item.metadata.get("valid_codes"),
            "predicted_code": pred,
        }

    return EvalSpec(
        name="telelogs_troubleshooting",
        items=items,
        parse_prediction=parse_prediction,
        format_label=format_label,
        format_pred=format_pred,
        is_correct=is_correct,
        per_class=True,
        extras=extras,
    )


# -------- WirelessMathBench (Verl conversion) --------

def _build_wireless_mathbench_spec(path: Path, records: Iterable[Dict[str, Any]]) -> EvalSpec:
    items: List[EvalItem] = []
    for idx, sample in enumerate(records):
        prompt_text = _render_chat_messages(sample.get("prompt"))
        reward_model = sample.get("reward_model") or {}
        ground_truth_raw = str(reward_model.get("ground_truth", "")).strip()
        ground_tokens = extract_boxed_values(ground_truth_raw)
        if not ground_tokens and ground_truth_raw:
            ground_tokens = [ground_truth_raw]
        normalized_tokens = [normalize_expression(tok) for tok in ground_tokens if tok]

        item_id = (
            sample.get("id")
            or (sample.get("extra_info") or {}).get("id")
            or f"wireless_mathbench_{idx:05d}"
        )
        ability = sample.get("ability")
        items.append(
            EvalItem(
                id=str(item_id),
                prompt=prompt_text,
                label=normalized_tokens,
                metadata={
                    "dataset": "wireless_mathbench",
                    "ground_truth_raw": ground_truth_raw,
                    "ground_truth_tokens": ground_tokens,
                    "ability": ability,
                    "style": reward_model.get("style"),
                    "extra_info": sample.get("extra_info"),
                },
            )
        )

    def parse_prediction(text: str, _: EvalItem) -> Dict[str, Any]:
        raw_tokens = extract_boxed_values(text)
        if not raw_tokens:
            marker_value = extract_after_marker(text)
            if marker_value:
                raw_tokens = [marker_value]
        if not raw_tokens:
            last_line = last_non_empty_line(text)
            if last_line:
                raw_tokens = [last_line]
        normalized = [normalize_expression(tok) for tok in raw_tokens if tok]
        return {"raw": raw_tokens, "tokens": normalized}

    def format_label(item: EvalItem) -> str:
        return item.metadata.get("ground_truth_raw", "")

    def format_pred(pred: Dict[str, Any], _: EvalItem) -> Optional[str]:
        if not pred:
            return None
        raw = pred.get("raw") or []
        if not raw:
            return None
        return " | ".join(raw)

    def is_correct(item: EvalItem, pred: Dict[str, Any]) -> bool:
        expected = item.label
        if not expected:
            return False
        tokens = pred.get("tokens") if pred else []
        return bool(tokens) and set(tokens) == set(expected)

    def extras(item: EvalItem, pred: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "expected_tokens_normalized": item.label,
            "pred_tokens_normalized": pred.get("tokens") if pred else None,
            "ability": item.metadata.get("ability"),
            "style": item.metadata.get("style"),
        }

    return EvalSpec(
        name="wireless_mathbench",
        items=items,
        parse_prediction=parse_prediction,
        format_label=format_label,
        format_pred=format_pred,
        is_correct=is_correct,
        per_class=False,
        extras=extras,
    )


# -------- WirelessMATHBench-XL --------

def _extract_letter_from_answer(answer: str) -> str:
    if not answer:
        return ""
    boxed = extract_boxed_values(answer)
    if boxed:
        answer = boxed[-1]
    answer = answer.strip().upper()
    if len(answer) == 1 and answer.isalpha():
        return answer
    match = re.search(r"[A-Z]", answer)
    return match.group(0) if match else answer


def _build_wireless_mathbench_xl_spec(path: Path, records: Iterable[Dict[str, Any]]) -> EvalSpec:
    items: List[EvalItem] = []
    for idx, sample in enumerate(records):
        prompt_text = _render_chat_messages(sample.get("prompt"))
        correct_answer_raw = str(sample.get("correct_answer", "")).strip()
        options = sample.get("options")
        item_id = str(
            sample.get("id")
            or (sample.get("extra_info") or {}).get("id")
            or f"wireless_mathbench_xl_{idx:05d}"
        )

        metadata = {
            "dataset": "wireless_mathbench_xl",
            "type": sample.get("type"),
            "options": options,
            "extra_info": sample.get("extra_info"),
            "paper_id": sample.get("paper_id"),
        }

        if isinstance(options, dict) and options:
            valid_letters = sorted(options.keys())
            label_letter = _extract_letter_from_answer(correct_answer_raw)
            items.append(
                EvalItem(
                    id=item_id,
                    prompt=prompt_text,
                    label=label_letter,
                    metadata={**metadata, "correct_answer_raw": correct_answer_raw, "valid_letters": valid_letters},
                )
            )
        else:
            gt_tokens = extract_boxed_values(correct_answer_raw)
            if not gt_tokens and correct_answer_raw:
                gt_tokens = [correct_answer_raw]
            normalized_tokens = [normalize_expression(tok) for tok in gt_tokens if tok]
            items.append(
                EvalItem(
                    id=item_id,
                    prompt=prompt_text,
                    label=normalized_tokens,
                    metadata={
                        **metadata,
                        "correct_answer_raw": correct_answer_raw,
                        "ground_truth_tokens": gt_tokens,
                    },
                )
            )

    def parse_prediction(text: str, item: EvalItem):
        options = item.metadata.get("options")
        if isinstance(options, dict) and options:
            valid_letters = [opt.upper() for opt in item.metadata.get("valid_letters", options.keys())]
            return extract_letter_choice(text, valid_letters)
        raw_tokens = extract_boxed_values(text)
        if not raw_tokens:
            marker_value = extract_after_marker(text)
            if marker_value:
                raw_tokens = [marker_value]
        if not raw_tokens:
            last_line = last_non_empty_line(text)
            if last_line:
                raw_tokens = [last_line]
        normalized = [normalize_expression(tok) for tok in raw_tokens if tok]
        return {"raw": raw_tokens, "tokens": normalized}

    def format_label(item: EvalItem) -> str:
        options = item.metadata.get("options")
        if isinstance(options, dict) and options:
            return f"\\boxed{{{item.label}}}"
        return item.metadata.get("correct_answer_raw", "")

    def format_pred(pred: Any, item: EvalItem) -> Optional[str]:
        options = item.metadata.get("options")
        if isinstance(options, dict) and options:
            if pred:
                return f"\\boxed{{{pred}}}"
            return None
        if not pred:
            return None
        raw = pred.get("raw") or []
        if not raw:
            return None
        return " | ".join(raw)

    def is_correct(item: EvalItem, pred: Any) -> bool:
        options = item.metadata.get("options")
        if isinstance(options, dict) and options:
            return bool(item.label) and pred == item.label
        expected = item.label
        if not expected:
            return False
        tokens = pred.get("tokens") if pred else []
        return bool(tokens) and set(tokens) == set(expected)

    def extras(item: EvalItem, pred: Any) -> Dict[str, Any]:
        options = item.metadata.get("options")
        if isinstance(options, dict) and options:
            return {
                "options": options,
                "predicted_letter": pred,
                "correct_answer_raw": item.metadata.get("correct_answer_raw"),
            }
        return {
            "ground_truth_tokens": item.metadata.get("ground_truth_tokens"),
            "pred_tokens_normalized": pred.get("tokens") if pred else None,
            "correct_answer_raw": item.metadata.get("correct_answer_raw"),
        }

    return EvalSpec(
        name="wireless_mathbench_xl",
        items=items,
        parse_prediction=parse_prediction,
        format_label=format_label,
        format_pred=format_pred,
        is_correct=is_correct,
        per_class=False,
        extras=extras,
    )
