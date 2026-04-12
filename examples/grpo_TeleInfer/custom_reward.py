# custom_reward.py

import math
import multiprocessing
import re
from typing import Union

try:
    import regex  # type: ignore[import]
except Exception:
    regex = re

try:
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr
    _HAS_SYMPY = True
except Exception:
    N = None
    simplify = None
    parse_latex = None
    parse_expr = None
    _HAS_SYMPY = False

try:
    from latex2sympy2 import latex2sympy  # type: ignore[import]
except Exception:
    try:
        from latex2sympy import latex2sympy  # type: ignore[import]
    except Exception:
        latex2sympy = None

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    _HAS_MATH_VERIFY = True
except Exception:
    mv_parse = None
    mv_verify = None
    _HAS_MATH_VERIFY = False

BOXED_RE = re.compile(r"\\boxed\{(.+?)\}")
MATH_ENV_RE = re.compile(r"^\$+|\\\(|\\\)|\)$")
LATEX_FRAC_RE = re.compile(
    r"^\\(?:dfrac|tfrac|frac)\s*\{?\s*([+-]?\d+(?:\.\d+)?)\s*\}?\s*\{?\s*([+-]?\d+(?:\.\d+)?)\s*\}?\s*$"
)

def _strip_boxed(s: str) -> str:
    m = BOXED_RE.search(s)
    if m:
        return m.group(1)
    return s

def _cleanup_tex(s: str) -> str:
    s = s.strip()
    s = _strip_boxed(s)
    # 去掉最外层 $...$ 或 \( ... \) 之类
    s = re.sub(r"^\s*(?:\$\$?|\\\()\s*", "", s)
    s = re.sub(r"\s*(?:\$\$?|\\\))\s*$", "", s)
    # 去掉花括号包裹的一层
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s

def _try_parse_number(s: str):
    """尽量把字符串解析为数值（支持 a/b、sqrt()、√x），失败返回 None。"""
    s = s.strip()
    s = _cleanup_tex(s)
    # 去掉多余空格
    s = s.replace(" ", "")
    try:
        # 分数 a/b
        if "/" in s and all(part.strip() for part in s.split("/", 1)):
            a, b = s.split("/", 1)
            return float(a) / float(b)
        # 根号
        if s.startswith("√"):
            return math.sqrt(float(s[1:]))
        if s.lower().startswith("sqrt(") and s.endswith(")"):
            return math.sqrt(float(s[5:-1]))
        # 直接 float
        return float(s)
    except Exception:
        return None

def _normalize_expr(s: str) -> str:
    s = _cleanup_tex(s)
    # 去空格、小写；去掉无关括号
    s = s.replace(" ", "").lower()
    # 去掉外层括号
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return s

def _get_first_not_none(*candidates):
    for x in candidates:
        if x is not None:
            return x
    return None


def _extract_direct_reward_fields(args, kwargs):
    """Extract direct call kwargs used by current VERL reward manager."""
    data = _extract_data_from_args_kwargs(args, kwargs)
    non_tensor = _extract_non_tensor(data)

    reward_cfg = non_tensor.get("reward_model", {})
    extra_info = (
        kwargs.get("extra_info")
        or non_tensor.get("extra_info")
        or {}
    )

    solution_str = _get_first_not_none(
        kwargs.get("solution_str"),
        _extract_response(non_tensor),
    )
    ground_truth = _get_first_not_none(
        kwargs.get("ground_truth"),
        reward_cfg.get("ground_truth"),
        non_tensor.get("ground_truth"),
    )
    style = _get_first_not_none(
        kwargs.get("style"),
        reward_cfg.get("style"),
        non_tensor.get("style"),
        extra_info.get("style") if isinstance(extra_info, dict) else None,
        "",
    )
    data_source = _get_first_not_none(
        kwargs.get("data_source"),
        non_tensor.get("data_source"),
        "",
    )

    return {
        "data": data,
        "non_tensor": non_tensor,
        "reward_cfg": reward_cfg,
        "extra_info": extra_info if isinstance(extra_info, dict) else {},
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "style": style,
        "data_source": data_source,
    }


def _extract_hash_answer(solution_str: str):
    if not solution_str:
        return None
    matches = re.findall(r"####\s*([^\n\r]+)", solution_str)
    if not matches:
        return None
    return _cleanup_tex(matches[-1]).strip() or None


def _extract_last_boxed_answer(solution_str: str):
    if not solution_str:
        return None

    idx = solution_str.rfind(r"\boxed")
    if idx < 0:
        return None

    i = idx
    while i < len(solution_str) and solution_str[i] != "{":
        i += 1
    if i >= len(solution_str):
        return None

    open_braces = 0
    right_idx = None
    while i < len(solution_str):
        if solution_str[i] == "{":
            open_braces += 1
        elif solution_str[i] == "}":
            open_braces -= 1
            if open_braces == 0:
                right_idx = i
                break
        i += 1

    if right_idx is None:
        return None

    boxed = solution_str[idx : right_idx + 1]
    left = r"\boxed{"
    if boxed.startswith(left) and boxed.endswith("}"):
        return boxed[len(left) : -1].strip() or None
    return None


def _extract_final_answer(solution_str: str):
    if not solution_str:
        return None, False, "missing"

    answer_region = solution_str
    if "</think>" in answer_region:
        answer_region = answer_region.split("</think>")[-1]

    boxed_answer = _extract_last_boxed_answer(answer_region)
    if boxed_answer is not None:
        return boxed_answer, True, "boxed"

    legacy_hash_answer = _extract_hash_answer(answer_region)
    if legacy_hash_answer is not None:
        return legacy_hash_answer, True, "hash"

    return None, False, "missing"


def _normalize_answer_string(expr: str):
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
    expr = expr.replace("%", "")
    expr = expr.replace(" ", "")
    expr = expr.replace("\n", "")
    expr = expr.replace("\t", "")

    if expr.startswith("{") and expr.endswith("}"):
        expr = expr[1:-1]

    return expr.strip() or None


def _parse_number_like(expr: str):
    if expr is None:
        return None

    s = _normalize_answer_string(expr)
    if s is None:
        return None

    frac_match = LATEX_FRAC_RE.match(s)
    if frac_match:
        try:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            if denominator == 0:
                return None
            return numerator / denominator
        except Exception:
            return None

    if s.lower().startswith("sqrt(") and s.endswith(")"):
        try:
            return math.sqrt(float(s[5:-1]))
        except Exception:
            return None

    if s.startswith("√"):
        try:
            return math.sqrt(float(s[1:]))
        except Exception:
            return None

    if "/" in s:
        left, right = s.split("/", 1)
        try:
            denom = float(right)
            if denom == 0:
                return None
            return float(left) / denom
        except Exception:
            return None

    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def choice_answer_clean(pred: str):
    pred = str(pred).strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
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
                pass
    return None


def is_digit(num):
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = str(input_str).strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def numeric_equal(prediction: float, reference: float):
    if reference == 0:
        return math.isclose(reference, prediction, abs_tol=1e-2)
    return math.isclose(reference, prediction, rel_tol=1e-2)


def symbolic_equal(a, b):
    if not _HAS_SYMPY:
        return False

    def _parse(s):
        s = str(s)
        parse_fns = [fn for fn in (parse_latex, parse_expr, latex2sympy) if fn is not None]
        for fn in parse_fns:
            try:
                return fn(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return fn(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if abs(a.lhs - a.rhs).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
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
    return output_queue.get()


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    if prediction is None or reference is None:
        return False

    prediction_text = str(prediction).strip()
    reference_text = str(reference).strip()
    if prediction_text.lower() == reference_text.lower():
        return True

    if reference_text in ["A", "B", "C", "D", "E"] and choice_answer_clean(prediction_text) == reference_text:
        return True

    try:
        if is_digit(prediction_text) and is_digit(reference_text):
            prediction_num = parse_digits(prediction_text)
            reference_num = parse_digits(reference_text)
            if include_percentage:
                gt_result = [reference_num / 100, reference_num, reference_num * 100]
            else:
                gt_result = [reference_num]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction_num, item):
                            return True
                    else:
                        if item == prediction_num:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference_text = str(reference).strip()
    prediction_text = str(prediction).strip()

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
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
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
                [
                    math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                    for i in range(len(pred_parts))
                ]
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
        pred_body = prediction_text
        ref_body = reference_text
        for begin_tag, end_tag in (
            ("\\begin{pmatrix}", "\\end{pmatrix}"),
            ("\\begin{bmatrix}", "\\end{bmatrix}"),
        ):
            if pred_body.startswith(begin_tag) and pred_body.endswith(end_tag):
                pred_body = pred_body[len(begin_tag) : -len(end_tag)]
            if ref_body.startswith(begin_tag) and ref_body.endswith(end_tag):
                ref_body = ref_body[len(begin_tag) : -len(end_tag)]

        pred_lines = [line.strip() for line in pred_body.split("\\\\") if line.strip()]
        ref_lines = [line.strip() for line in ref_body.split("\\\\") if line.strip()]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction_text.count("=") == 1 and reference_text.count("=") == 1:
        pred = prediction_text.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference_text.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
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
        if call_with_timeout(symbolic_equal_process, prediction_text, reference_text):
            return True
    else:
        if symbolic_equal(prediction_text, reference_text):
            return True

    return False


def _symbolic_equal_like(prediction: str, reference: str, tolerance: float = 1e-4):
    del tolerance
    return symbolic_equal(prediction, reference)


def _math_equal_like(prediction, reference, include_percentage: bool = True, tolerance: float = 1e-4):
    del tolerance
    return math_equal(prediction, reference, include_percentage=include_percentage, timeout=True)


def _math_verify_first(expr: str):
    if not _HAS_MATH_VERIFY or mv_parse is None or expr is None:
        return None

    try:
        try:
            parsed = mv_parse(pred=expr, parsing_timeout=None)
        except TypeError:
            parsed = mv_parse(expr, parsing_timeout=None)
    except Exception:
        return None

    if isinstance(parsed, (list, tuple)):
        for item in parsed:
            if item is not None:
                return item
        return None
    return parsed


def _math_verify_equal(solution_str: str, ground_truth) -> bool:
    if not _HAS_MATH_VERIFY or mv_verify is None:
        return False

    pred_obj = None
    gold_obj = None
    try:
        pred_obj = _math_verify_first(solution_str)
        gold_obj = _math_verify_first(f"\\boxed{{${ground_truth}$}}")
        if pred_obj is None or gold_obj is None:
            return False
        return bool(
            mv_verify(
                gold_obj,
                pred_obj,
                float_rounding=6,
                numeric_precision=15,
                strict=True,
                timeout_seconds=3,
            )
        )
    except TypeError:
        try:
            return bool(mv_verify(gold_obj, pred_obj))
        except Exception:
            return False
    except Exception:
        return False


def _legacy_choice_or_blank_score(solution_str, ground_truth, style, eps=1e-6):
    if not solution_str or not ground_truth:
        return {
            "score": -1.0,
            "acc": False,
            "pred": None,
            "legacy_mode": style,
        }

    sol = str(solution_str).strip()
    gt = str(ground_truth).strip()
    extracted_pred, format_correct, format_mode = _extract_final_answer(sol)
    candidate_pred = extracted_pred if extracted_pred is not None else sol

    if style == "multiple_choice":
        pred = choice_answer_clean(candidate_pred)
        acc = math_equal(pred, gt, include_percentage=False, is_close=False)
        return {
            "score": 1.0 if acc else -1.0,
            "acc": acc,
            "pred": pred,
            "legacy_mode": style,
            "format_correct": format_correct,
            "format_mode": format_mode,
        }

    acc = math_equal(candidate_pred, gt, timeout=True)
    return {
        "score": 1.0 if acc else -1.0,
        "acc": acc,
        "pred": candidate_pred,
        "legacy_mode": style,
        "format_correct": format_correct,
        "format_mode": format_mode,
    }

def _extract_data_from_args_kwargs(args, kwargs):
    """
    兼容多种调用方式，尽量拿到 data/DataProto：
    - 位置参数优先：args[0]
    - 关键字参数里找：data / data_proto / batch
    """
    if args and args[0] is not None:
        return args[0]
    for k in ("data", "data_proto", "batch"):
        if k in kwargs and kwargs[k] is not None:
            return kwargs[k]
    return None

def _extract_non_tensor(batch_like):
    """
    从 DataProto 或 dict 中拿 non_tensor 部分。
    """
    if batch_like is None:
        return {}
    # DataProto: 有 non_tensor_batch
    nt = getattr(batch_like, "non_tensor_batch", None)
    if nt is not None:
        return nt
    # 如果本身就是 dict
    if isinstance(batch_like, dict):
        # 有的路径会把 non_tensor_batch 封在这个键里
        if "non_tensor_batch" in batch_like and isinstance(batch_like["non_tensor_batch"], dict):
            return batch_like["non_tensor_batch"]
        return batch_like
    return {}

def _extract_response(non_tensor):
    """
    模型输出可能在不同的 key：做个兜底查找。
    """
    # 常见命名
    resp_keys = [
        "response_str", "response", "generation", "output", "text",
        "pred", "answer", "model_answer", "chosen", "sample"
    ]
    for k in resp_keys:
        v = non_tensor.get(k)
        if isinstance(v, str) and v.strip():
            return v
        # 有些会是 list[str]
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            return v[0]
    # 有的会把多候选放在 "responses" 或 "generations"
    for k in ["responses", "generations"]:
        v = non_tensor.get(k)
        if isinstance(v, (list, tuple)) and v:
            # 取第一个字符串或其中的字段
            if isinstance(v[0], str):
                return v[0]
            if isinstance(v[0], dict):
                return _get_first_not_none(
                    v[0].get("text"),
                    v[0].get("response"),
                    v[0].get("generation"),
                )
    return None

def my_math_reward_fn(*args, **kwargs):
    """
    自定义奖励函数：支持 multiple_choice 和 fill_in_the_blank。
    兼容 VERL 不同调用路径：位置参数/关键字参数传入 data。
    允许额外 kwargs（例如 return_dict、tokenizer 等）无副作用。
    """
    eps = kwargs.get("eps", 1e-6)

    # 1) 拿到 data / DataProto
    data = _extract_data_from_args_kwargs(args, kwargs)

    # 2) 拿 non-tensor 部分
    non_tensor = _extract_non_tensor(data)

    # 3) reward 配置通常在 reward_model 节点
    reward_cfg = non_tensor.get("reward_model", non_tensor)
    style = reward_cfg.get("style") or non_tensor.get("style") or ""
    ground_truth = reward_cfg.get("ground_truth") or non_tensor.get("ground_truth") or ""

    # 4) 拿模型输出
    solution_str = _extract_response(non_tensor)

    extracted_pred, _, _ = _extract_final_answer(solution_str) if solution_str else (None, False, "missing")
    candidate_pred = extracted_pred if extracted_pred is not None else solution_str

    # 5) 计算 reward
    reward = 0.0

    if style == "multiple_choice":
        if candidate_pred and ground_truth:
            reward = 1.0 if math_equal(candidate_pred, ground_truth, include_percentage=False, is_close=False) else 0.0
        else:
            reward = 0.0

    elif style in {"fill_in_the_blank", "rule"}:
        if candidate_pred and ground_truth:
            reward = 1.0 if math_equal(candidate_pred, ground_truth, timeout=True) else 0.0
        else:
            reward = 0.0

    else:
        # 未知样式，保守给 0
        reward = 0.0

    # 允许 reward manager 传多余的 kwargs（例如 return_dict），这里直接忽略
    return float(reward)


def my_math_reward_fn_deepmath_boxed(*args, **kwargs):
    """
    DeepMath-like reward:
    - Prefer final answer in \\boxed{...}
    - Accept legacy #### fallback so existing TeleInfer prompts do not collapse
    - Validate with math equivalence instead of pure string matching
    - Keep multiple-choice / fill-in-the-blank legacy behavior available
    """
    reward_fields = _extract_direct_reward_fields(args, kwargs)
    solution_str = reward_fields["solution_str"]
    ground_truth = reward_fields["ground_truth"]
    style = str(reward_fields["style"] or "").strip()
    extra_info = reward_fields["extra_info"]

    if style in {"multiple_choice", "fill_in_the_blank"}:
        return _legacy_choice_or_blank_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            style=style,
            eps=kwargs.get("eps", 1e-6),
        )

    pred, format_correct, format_mode = _extract_final_answer(solution_str)

    omi_correct = False
    mathv_correct = False
    if format_correct and pred is not None and ground_truth is not None:
        omi_correct = math_equal(pred, ground_truth, timeout=True)
        answer_region = solution_str.split("</think>")[-1] if solution_str and "</think>" in solution_str else solution_str
        mathv_correct = _math_verify_equal(answer_region, ground_truth)

    acc = bool(format_correct and omi_correct)
    score = 1.0 if acc else -1.0

    return {
        "score": score,
        "acc": acc,
        "pred": pred,
        "format_correct": format_correct,
        "format_mode": format_mode,
        "omi_correct": omi_correct,
        "mathv_correct": mathv_correct,
        "scoring_backend": "math_equal",
        "style": style or (extra_info.get("style") if isinstance(extra_info, dict) else ""),
    }
