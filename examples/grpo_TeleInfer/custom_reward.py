"""TeleInfer custom reward functions.

This module is designed to be loaded through VERL's
`custom_reward_function.path/name` configuration.

The intended call chain is:
1. reward_manager calls `default_reward_function`
2. `default_reward_function` extracts the current sample fields
3. a task-specific reward function is selected and executed

The symbolic verifier is adapted from the Hendrycks MATH style equivalence
logic the user provided.
"""

from __future__ import annotations

import math
import multiprocessing
import re
from typing import Any, Callable, Union

try:
    import regex  # type: ignore[import]
except Exception:
    regex = re

try:
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr
except Exception:
    N = None
    simplify = None
    parse_latex = None
    parse_expr = None

try:
    from latex2sympy2 import latex2sympy  # type: ignore[import]
except Exception:
    try:
        from latex2sympy import latex2sympy  # type: ignore[import]
    except Exception:
        latex2sympy = None


AnswerValue = Union[bool, float, str]

_BOXED_TOKEN = r"\boxed"
_FINAL_ANSWER_MARKERS = (
    "### Final Result",
    "### Final Answer",
)
def choice_answer_clean(pred: str) -> str:
    pred = str(pred).strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp[-1]
    else:
        pred = pred.strip().strip(".")
    return pred.rstrip(".").rstrip("/")


def parse_digits(num: Any) -> float | None:
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                return None
    return None


def is_digit(num: Any) -> bool:
    return parse_digits(num) is not None


def str_to_pmatrix(input_str: str) -> str:
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for matrix in matrix_str:
        matrix = matrix.strip("{}")
        pmatrix = r"\begin{pmatrix}" + matrix.replace(",", "\\\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def numeric_equal(prediction: float, reference: float) -> bool:
    return math.isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a: Any, b: Any) -> bool:
    def _parse(expr: Any) -> Any:
        expr = str(expr)
        for parser_fn in (parse_latex, parse_expr, latex2sympy):
            if parser_fn is None:
                continue
            try:
                return parser_fn(expr.replace("\\\\", "\\"))
            except Exception:
                try:
                    return parser_fn(expr)
                except Exception:
                    continue
        return expr

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    if simplify is not None:
        try:
            if a.equals(b) or simplify(a - b) == 0:
                return True
        except Exception:
            pass

    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    if N is not None:
        try:
            if numeric_equal(float(N(a)), float(N(b))):
                return True
        except Exception:
            pass

    try:
        if a.shape == b.shape:
            rounded_a = a.applyfunc(lambda x: round(x, 3))
            rounded_b = b.applyfunc(lambda x: round(x, 3))
            if rounded_a.equals(rounded_b):
                return True
    except Exception:
        pass

    return False


def symbolic_equal_process(a: Any, b: Any, output_queue: multiprocessing.Queue) -> None:
    output_queue.put(symbolic_equal(a, b))


def call_with_timeout(func: Callable[..., None], *args: Any, timeout: float = 1, **kwargs: Any) -> bool:
    output_queue: multiprocessing.Queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    if output_queue.empty():
        return False

    return bool(output_queue.get())


def math_equal(
    prediction: AnswerValue,
    reference: AnswerValue,
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """Compare math answers by numeric and symbolic equivalence."""
    if prediction is None or reference is None:
        return False

    prediction_text = str(prediction).strip()
    reference_text = str(reference).strip()
    if not prediction_text or not reference_text:
        return False

    if prediction_text.lower() == reference_text.lower():
        return True

    if reference_text in ["A", "B", "C", "D", "E"] and choice_answer_clean(prediction_text) == reference_text:
        return True

    try:
        if is_digit(prediction_text) and is_digit(reference_text):
            prediction_num = parse_digits(prediction_text)
            reference_num = parse_digits(reference_text)
            if prediction_num is not None and reference_num is not None:
                candidates = [reference_num / 100, reference_num, reference_num * 100] if include_percentage else [reference_num]
                for item in candidates:
                    try:
                        if is_close:
                            if numeric_equal(prediction_num, item):
                                return True
                        elif item == prediction_num:
                            return True
                    except Exception:
                        continue
                return False
    except Exception:
        pass

    if "pmatrix" in prediction_text and "pmatrix" not in reference_text:
        reference_text = str_to_pmatrix(reference_text)

    pred_str, ref_str = prediction_text, reference_text
    if (
        prediction_text.startswith("[")
        and prediction_text.endswith("]")
        and not reference_text.startswith("(")
    ) or (
        prediction_text.startswith("(")
        and prediction_text.endswith(")")
        and not reference_text.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")

    for token in ["{", "}", "(", ")"]:
        pred_str = pred_str.replace(token, "")
        ref_str = ref_str.replace(token, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction_text) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference_text) is not None
    ):
        pred_parts = prediction_text[1:-1].split(",")
        ref_parts = reference_text[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_parts[idx], ref_parts[idx], include_percentage, is_close)
                for idx in range(len(pred_parts))
            ):
                return True

    if (
        (
            prediction_text.startswith("\\begin{pmatrix}")
            or prediction_text.startswith("\\begin{bmatrix}")
        )
        and (
            prediction_text.endswith("\\end{pmatrix}")
            or prediction_text.endswith("\\end{bmatrix}")
        )
        and (
            reference_text.startswith("\\begin{pmatrix}")
            or reference_text.startswith("\\begin{bmatrix}")
        )
        and (
            reference_text.endswith("\\end{pmatrix}")
            or reference_text.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction_text[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference_text[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        if len(pred_lines) == len(ref_lines):
            matched = True
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) != len(ref_parts):
                    matched = False
                    break
                if not all(
                    math_equal(pred_parts[idx], ref_parts[idx], include_percentage, is_close)
                    for idx in range(len(pred_parts))
                ):
                    matched = False
                    break
            if matched:
                return True

    if prediction_text.count("=") == 1 and reference_text.count("=") == 1:
        pred = prediction_text.split("=")
        ref = reference_text.split("=")
        pred_expr = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref_expr = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred_expr, ref_expr) or symbolic_equal(f"-({pred_expr})", ref_expr):
            return True
    elif (
        prediction_text.count("=") == 1
        and len(prediction_text.split("=")[0].strip()) <= 2
        and "=" not in reference_text
    ):
        if math_equal(prediction_text.split("=")[1], reference_text, include_percentage, is_close):
            return True
    elif (
        reference_text.count("=") == 1
        and len(reference_text.split("=")[0].strip()) <= 2
        and "=" not in prediction_text
    ):
        if math_equal(prediction_text, reference_text.split("=")[1], include_percentage, is_close):
            return True

    if timeout:
        return call_with_timeout(symbolic_equal_process, prediction_text, reference_text)

    return symbolic_equal(prediction_text, reference_text)


def _get_first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _extract_data_from_args_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args and len(args) >= 3 and isinstance(args[0], str):
        return None
    if args and args[0] is not None:
        return args[0]
    for key in ("data", "data_proto", "batch"):
        if key in kwargs and kwargs[key] is not None:
            return kwargs[key]
    return None


def _extract_non_tensor(batch_like: Any) -> dict[str, Any]:
    if batch_like is None:
        return {}

    non_tensor = getattr(batch_like, "non_tensor_batch", None)
    if non_tensor is not None:
        return non_tensor

    if isinstance(batch_like, dict):
        if "non_tensor_batch" in batch_like and isinstance(batch_like["non_tensor_batch"], dict):
            return batch_like["non_tensor_batch"]
        return batch_like

    return {}


def _extract_response(non_tensor: dict[str, Any]) -> str | None:
    response_keys = (
        "response_str",
        "response",
        "generation",
        "output",
        "text",
        "pred",
        "answer",
        "model_answer",
        "chosen",
        "sample",
    )

    for key in response_keys:
        value = non_tensor.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
            return value[0]

    for key in ("responses", "generations"):
        value = non_tensor.get(key)
        if isinstance(value, (list, tuple)) and value:
            if isinstance(value[0], str):
                return value[0]
            if isinstance(value[0], dict):
                return _get_first_not_none(
                    value[0].get("text"),
                    value[0].get("response"),
                    value[0].get("generation"),
                )

    return None


def _normalize_answer_string(expr: Any) -> str | None:
    if expr is None:
        return None

    expr = str(expr).strip()
    if not expr:
        return None

    expr = expr.replace(r"\left", "")
    expr = expr.replace(r"\right", "")
    expr = expr.replace(r"\!", "")
    expr = expr.replace(r"\%", "%")
    expr = expr.replace(r"\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("\n", " ").replace("\t", " ")

    if expr.startswith("{") and expr.endswith("}"):
        expr = expr[1:-1]

    return expr.strip() or None


def _strip_answer_region(solution_str: str | None) -> str:
    if not solution_str:
        return ""
    if "</think>" in solution_str:
        return solution_str.split("</think>")[-1]
    return solution_str


def _extract_hash_answer(solution_str: str | None) -> str | None:
    if not solution_str:
        return None
    matches = re.findall(r"####\s*([^\n\r]+)", solution_str)
    if not matches:
        return None
    answer = _normalize_answer_string(matches[-1])
    return answer.strip() if answer else None


def _extract_last_boxed_answer(solution_str: str | None) -> str | None:
    if not solution_str:
        return None

    idx = solution_str.rfind(_BOXED_TOKEN)
    if idx < 0:
        return None

    cursor = idx
    while cursor < len(solution_str) and solution_str[cursor] != "{":
        cursor += 1
    if cursor >= len(solution_str):
        return None

    opened = 0
    right_idx = None
    while cursor < len(solution_str):
        if solution_str[cursor] == "{":
            opened += 1
        elif solution_str[cursor] == "}":
            opened -= 1
            if opened == 0:
                right_idx = cursor
                break
        cursor += 1

    if right_idx is None:
        return None

    boxed = solution_str[idx : right_idx + 1]
    if boxed.startswith(r"\boxed{") and boxed.endswith("}"):
        return boxed[len(r"\boxed{") : -1].strip() or None
    return None


def _extract_final_answer(solution_str: str | None) -> tuple[str | None, bool, str]:
    if not solution_str:
        return None, False, "missing"

    answer_region = _strip_answer_region(solution_str)

    last_header_idx = -1
    for marker in _FINAL_ANSWER_MARKERS:
        idx = answer_region.rfind(marker)
        if idx > last_header_idx:
            last_header_idx = idx
    if last_header_idx >= 0:
        answer_region = answer_region[last_header_idx:]

    boxed_answer = _extract_last_boxed_answer(answer_region)
    if boxed_answer is not None:
        return boxed_answer, True, "boxed"

    hash_answer = _extract_hash_answer(answer_region)
    if hash_answer is not None:
        return hash_answer, True, "hash"

    return None, False, "missing"


def _extract_reward_fields(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) >= 3 and isinstance(args[0], str):
        extra_info = kwargs.get("extra_info", {})
        style = ""
        if isinstance(extra_info, dict):
            style = str(extra_info.get("style") or "")
        return {
            "data": None,
            "non_tensor": {},
            "data_source": args[0],
            "solution_str": args[1],
            "ground_truth": args[2],
            "extra_info": extra_info if isinstance(extra_info, dict) else {},
            "style": style,
        }

    data = _extract_data_from_args_kwargs(args, kwargs)
    non_tensor = _extract_non_tensor(data)
    reward_cfg = non_tensor.get("reward_model", {})
    extra_info = kwargs.get("extra_info") or non_tensor.get("extra_info") or {}
    if not isinstance(extra_info, dict):
        extra_info = {}

    data_source = _get_first_not_none(
        kwargs.get("data_source"),
        non_tensor.get("data_source"),
        "",
    )
    solution_str = _get_first_not_none(
        kwargs.get("solution_str"),
        _extract_response(non_tensor),
        "",
    )
    ground_truth = _get_first_not_none(
        kwargs.get("ground_truth"),
        reward_cfg.get("ground_truth"),
        non_tensor.get("ground_truth"),
        "",
    )
    style = _get_first_not_none(
        kwargs.get("style"),
        reward_cfg.get("style"),
        non_tensor.get("style"),
        extra_info.get("style"),
        "",
    )

    return {
        "data": data,
        "non_tensor": non_tensor,
        "data_source": str(data_source or ""),
        "solution_str": str(solution_str or ""),
        "ground_truth": ground_truth,
        "extra_info": extra_info,
        "style": str(style or ""),
    }


def _build_result(
    *,
    score: float,
    acc: bool,
    pred: str | None,
    reward_name: str,
    data_source: str,
    style: str,
    format_correct: bool | None = None,
    format_mode: str | None = None,
) -> dict[str, Any]:
    result = {
        "score": float(score),
        "acc": bool(acc),
        "pred": pred,
        "reward_name": reward_name,
        "data_source": data_source,
        "style": style,
    }
    if format_correct is not None:
        result["format_correct"] = bool(format_correct)
    if format_mode is not None:
        result["format_mode"] = format_mode
    return result


def _score_multiple_choice(fields: dict[str, Any]) -> dict[str, Any]:
    answer, format_correct, format_mode = _extract_final_answer(fields["solution_str"])
    candidate_pred = answer if answer is not None else fields["solution_str"]
    pred = choice_answer_clean(candidate_pred) if candidate_pred else None
    gt = str(fields["ground_truth"]).strip() if fields["ground_truth"] is not None else ""
    acc = bool(pred and gt and math_equal(pred, gt, include_percentage=False, is_close=False))
    return _build_result(
        score=1.0 if acc else -1.0,
        acc=acc,
        pred=pred,
        reward_name="multiple_choice_reward",
        data_source=fields["data_source"],
        style=fields["style"],
        format_correct=format_correct,
        format_mode=format_mode,
    )


def _score_fill_in_blank(fields: dict[str, Any]) -> dict[str, Any]:
    answer, format_correct, format_mode = _extract_final_answer(fields["solution_str"])
    candidate_pred = answer if answer is not None else fields["solution_str"]
    candidate_pred = _normalize_answer_string(candidate_pred)
    ground_truth = _normalize_answer_string(fields["ground_truth"])
    acc = bool(candidate_pred and ground_truth and math_equal(candidate_pred, ground_truth, timeout=True))
    return _build_result(
        score=1.0 if acc else -1.0,
        acc=acc,
        pred=candidate_pred,
        reward_name="fill_in_blank_reward",
        data_source=fields["data_source"],
        style=fields["style"],
        format_correct=format_correct,
        format_mode=format_mode,
    )


def _score_rule_reward(fields: dict[str, Any]) -> dict[str, Any]:
    pred, format_correct, format_mode = _extract_final_answer(fields["solution_str"])
    ground_truth = _normalize_answer_string(fields["ground_truth"])
    pred = _normalize_answer_string(pred)
    acc = bool(format_correct and pred and ground_truth and math_equal(pred, ground_truth, timeout=True))
    return _build_result(
        score=1.0 if acc else -1.0,
        acc=acc,
        pred=pred,
        reward_name="rule_reward",
        data_source=fields["data_source"],
        style="rule",
        format_correct=format_correct,
        format_mode=format_mode,
    )


def default_reward_function(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Default TeleInfer reward entrypoint used by the reward manager.

    TeleInfer data is treated as rule-style symbolic verification, so we do not
    dispatch by style or data_source here.
    """
    fields = _extract_reward_fields(args, kwargs)
    return _score_rule_reward(fields)


def my_math_reward_fn(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible alias for older TeleInfer configs."""
    return default_reward_function(*args, **kwargs)


def my_math_reward_fn_deepmath_boxed(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible alias for the prior default symbolic TeleInfer reward."""
    return default_reward_function(*args, **kwargs)


__all__ = [
    "default_reward_function",
    "my_math_reward_fn",
    "my_math_reward_fn_deepmath_boxed",
    "math_equal",
]
