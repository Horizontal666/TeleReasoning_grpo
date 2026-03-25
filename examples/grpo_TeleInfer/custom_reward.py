# custom_reward.py

import math
import re

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


def _symbolic_equal_like(prediction: str, reference: str, tolerance: float = 1e-4):
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr
    except Exception:
        return False

    def _parse(expr: str):
        expr = expr.replace("^", "**")
        for parser in (parse_expr, parse_latex):
            try:
                return parser(expr)
            except Exception:
                continue
        return None

    pred_obj = _parse(prediction)
    ref_obj = _parse(reference)
    if pred_obj is None or ref_obj is None:
        return False

    try:
        if sympy.simplify(pred_obj - ref_obj) == 0:
            return True
    except Exception:
        pass

    try:
        pred_val = float(sympy.N(pred_obj))
        ref_val = float(sympy.N(ref_obj))
        return math.isclose(pred_val, ref_val, rel_tol=tolerance, abs_tol=tolerance)
    except Exception:
        return False


def _math_equal_like(prediction, reference, include_percentage: bool = True, tolerance: float = 1e-4):
    pred = _normalize_answer_string(prediction)
    ref = _normalize_answer_string(reference)
    if pred is None or ref is None:
        return False

    if pred.lower() == ref.lower():
        return True

    pred_num = _parse_number_like(pred)
    ref_num = _parse_number_like(ref)
    if pred_num is not None and ref_num is not None:
        candidates = [ref_num]
        if include_percentage:
            candidates.extend([ref_num / 100.0, ref_num * 100.0])
        for cand in candidates:
            if math.isclose(pred_num, cand, rel_tol=tolerance, abs_tol=tolerance):
                return True
        return False

    if "," in pred and "," in ref:
        pred_parts = [part.strip() for part in pred.split(",")]
        ref_parts = [part.strip() for part in ref.split(",")]
        if len(pred_parts) == len(ref_parts):
            return all(
                _math_equal_like(p, r, include_percentage=include_percentage, tolerance=tolerance)
                for p, r in zip(pred_parts, ref_parts)
            )

    return _symbolic_equal_like(pred, ref, tolerance=tolerance)


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

    if style == "multiple_choice":
        match = re.search(r"####\s*([A-D])\b", sol, re.IGNORECASE)
        pred = match.group(1).upper() if match else sol.upper()
        acc = pred == gt.upper()
        return {
            "score": 1.0 if acc else -1.0,
            "acc": acc,
            "pred": pred,
            "legacy_mode": style,
        }

    norm_sol = _normalize_expr(sol)
    norm_gt = _normalize_expr(gt)
    if norm_sol == norm_gt:
        return {
            "score": 1.0,
            "acc": True,
            "pred": sol,
            "legacy_mode": style,
        }

    sol_val = _try_parse_number(sol)
    gt_val = _try_parse_number(gt)
    acc = (
        sol_val is not None
        and gt_val is not None
        and abs(sol_val - gt_val) < eps
    )
    return {
        "score": 1.0 if acc else -1.0,
        "acc": acc,
        "pred": sol,
        "legacy_mode": style,
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

    # 5) 计算 reward
    reward = 0.0

    if style == "multiple_choice":
        # 只要选项字符串一致就给 1.0（可以再加更鲁棒的 A/B/C/D 提取）
        if solution_str and ground_truth:
            sol = solution_str.strip()
            gt = ground_truth.strip()
            # 从输出里抽末尾 "#### X" 的选项也可增强鲁棒性
            m = re.search(r"####\s*([A-D])\b", sol, re.IGNORECASE)
            if m:
                sol = m.group(1).upper()
                gt  = gt.upper()
                reward = 1.0 if sol == gt else 0.0
            else:
                reward = 1.0 if sol == gt else 0.0
        else:
            reward = 0.0

    elif style == "fill_in_the_blank":
        if solution_str and ground_truth:
            sol = solution_str.strip()
            gt  = ground_truth.strip()

            # 优先做表达式规范化的字符串精确匹配
            if _normalize_expr(sol) == _normalize_expr(gt):
                reward = 1.0
            else:
                # 再尝试数值比较
                sol_val = _try_parse_number(sol)
                gt_val  = _try_parse_number(gt)
                if sol_val is not None and gt_val is not None and abs(sol_val - gt_val) < eps:
                    reward = 1.0
                else:
                    reward = 0.0
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
        omi_correct = _math_equal_like(pred, ground_truth, tolerance=kwargs.get("tolerance", 1e-4))
        answer_region = solution_str.split("</think>")[-1] if solution_str and "</think>" in solution_str else solution_str
        mathv_correct = _math_verify_equal(answer_region, ground_truth)

    acc = bool(format_correct and (omi_correct or mathv_correct))
    score = 1.0 if acc else -1.0

    return {
        "score": score,
        "acc": acc,
        "pred": pred,
        "format_correct": format_correct,
        "format_mode": format_mode,
        "omi_correct": omi_correct,
        "mathv_correct": mathv_correct,
        "style": style or (extra_info.get("style") if isinstance(extra_info, dict) else ""),
    }
