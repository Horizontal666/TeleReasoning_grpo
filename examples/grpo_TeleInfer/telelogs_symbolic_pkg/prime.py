"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
"""

import math
import re
import regex
import multiprocessing
from math import isclose
from typing import Any, Optional, Union
from collections import defaultdict

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy

# from .parser import choice_answer_clean, strip_string
# from parser import choice_answer_clean


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


# ---------------------------------------------------------------------------
# Numeric-expression fast path (synced from scripts/regrade_telemath_attempts.py)
# ---------------------------------------------------------------------------

def _strip_latex_wrappers(s: Any) -> str:
    s = str(s).strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    for wrapper in (r"\left", r"\right"):
        s = s.replace(wrapper, "")
    s = s.replace("−", "-").replace("×", r"\times")
    s = s.replace(r"\,", "").replace(r"\!", "").replace(r"\ ", "")
    s = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{[^{}]*\}", "", s)
    return s.strip()


def _single_numeric_token(s: str) -> Optional[float]:
    token = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    matches = re.findall(token, s)
    if len(matches) != 1:
        return None
    try:
        return float(matches[0])
    except Exception:
        return None


def _eval_log_with_base(s: str, *, assume_log_base2: bool = False) -> Optional[float]:
    text = _strip_latex_wrappers(s)
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace("{", "").replace("}", "")

    m = re.fullmatch(
        r"\\?log_\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\(?\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)?",
        text,
    )
    if m:
        base = float(m.group(1))
        arg = float(m.group(2))
        if base > 0 and base != 1 and arg > 0:
            return math.log(arg, base)
        return None

    if assume_log_base2:
        m = re.fullmatch(
            r"\\?log\s*\(?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)?",
            text,
        )
        if m:
            arg = float(m.group(1))
            if arg > 0:
                return math.log2(arg)
    return None


def _normalize_for_parse_expr(s: str) -> str:
    text = _strip_latex_wrappers(s)
    text = text.replace("^", "**")
    text = re.sub(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\\times10\*\*\{?([+-]?\d+)\}?",
        r"\1*10**(\2)",
        text,
    )
    text = re.sub(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\\times10\^\{?([+-]?\d+)\}?",
        r"\1*10**(\2)",
        text,
    )
    text = text.replace(r"\times", "*")
    return text


def _looks_numeric_expression(s: str) -> bool:
    text = _strip_latex_wrappers(s)
    if not re.search(r"\d", text):
        return False
    text = re.sub(
        r"\\(?:frac|dfrac|tfrac|sqrt|log|ln|exp|sin|cos|tan|left|right|times|cdot|pi)\b",
        "",
        text,
    )
    text = text.replace("\\", "")
    text = re.sub(r"\b(?:log|ln|exp|sqrt|sin|cos|tan|pi|e)\b", "", text)
    return re.search(r"[A-Za-z]", text) is None


def eval_expr_to_float(value: Any, *, assume_log_base2: bool = False) -> Optional[float]:
    if value is None:
        return None

    digit = parse_digits(value)
    if digit is not None:
        return digit

    original = str(value).strip()
    if not original:
        return None

    log_val = _eval_log_with_base(original, assume_log_base2=assume_log_base2)
    if log_val is not None and math.isfinite(log_val):
        return log_val

    cleaned = _strip_latex_wrappers(original)
    if any(marker in cleaned for marker in (r"\approx", "≈")):
        approx_tail = re.split(r"\\approx|≈", cleaned)[-1].strip()
        approx_val = eval_expr_to_float(approx_tail, assume_log_base2=assume_log_base2)
        if approx_val is not None:
            return approx_val

    if "=" in cleaned and cleaned.count("=") == 1:
        left, right = cleaned.split("=", 1)
        if len(left.strip()) <= 12:
            right_val = eval_expr_to_float(right, assume_log_base2=assume_log_base2)
            if right_val is not None:
                return right_val

    normalized = _normalize_for_parse_expr(original)
    for candidate in (cleaned, normalized):
        one = _single_numeric_token(candidate)
        if one is not None:
            return one

    if not _looks_numeric_expression(original):
        return None

    for parser in (parse_latex, parse_expr, latex2sympy):
        for candidate in (cleaned, normalized, original):
            try:
                expr = parser(candidate.replace("\\\\", "\\"))
            except Exception:
                try:
                    expr = parser(candidate)
                except Exception:
                    continue
            try:
                if getattr(expr, "free_symbols", None):
                    continue
                val = float(N(expr))
                if math.isfinite(val):
                    return val
            except Exception:
                continue

    return None


def numeric_expression_equal(
    prediction: Any,
    reference: Any,
    *,
    include_percentage: bool = True,
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-4,
    assume_log_base2: bool = False,
    max_numeric_expr_chars: int = 240,
) -> bool:
    if max(len(str(prediction)), len(str(reference))) > max_numeric_expr_chars:
        return False
    pred_num = eval_expr_to_float(prediction, assume_log_base2=assume_log_base2)
    ref_num = eval_expr_to_float(reference, assume_log_base2=assume_log_base2)
    if pred_num is None or ref_num is None:
        return False

    candidates = [ref_num]
    if include_percentage:
        candidates = [ref_num / 100, ref_num, ref_num * 100]
    return any(
        numeric_equal(pred_num, item, rel_tol=rel_tol, abs_tol=abs_tol)
        for item in candidates
    )


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
    *,
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-4,
    timeout_seconds: float = 2.0,
    assume_log_base2: bool = False,
    max_numeric_expr_chars: int = 240,
    max_symbolic_chars: int = 240,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item, rel_tol=rel_tol, abs_tol=abs_tol):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 1.5 numeric-expression fast path: handles unit-suffixed GT
    # ("-26.0206\\,\\mathrm{dBW}") and loose engineering precision before
    # falling through to sympy.
    if numeric_expression_equal(
        prediction,
        reference,
        include_percentage=include_percentage,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        assume_log_base2=assume_log_base2,
        max_numeric_expr_chars=max_numeric_expr_chars,
    ):
        return True

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
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

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref, rel_tol=rel_tol, abs_tol=abs_tol) or symbolic_equal(
            f"-({pred})", ref, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    # bail out before sympy if either side is too long: prevents pathological
    # symbolic expressions from blocking the reward batch.
    if max(len(prediction), len(reference)) > max_symbolic_chars:
        return False

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(
            symbolic_equal_process,
            prediction,
            reference,
            rel_tol,
            abs_tol,
            timeout=timeout_seconds,
        ):
            return True
    else:
        if symbolic_equal(prediction, reference, rel_tol=rel_tol, abs_tol=abs_tol):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float, *, rel_tol: float = 1e-2, abs_tol: float = 1e-4) -> bool:
    return isclose(reference, prediction, rel_tol=rel_tol, abs_tol=abs_tol)


def symbolic_equal(a, b, *, rel_tol: float = 1e-2, abs_tol: float = 1e-4):
    def _parse(s):
        text = _strip_latex_wrappers(str(s))
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(text.replace("\\\\", "\\"))
            except:
                try:
                    return f(text)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b)), rel_tol=rel_tol, abs_tol=abs_tol):
            return True
    except:
        pass

    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, rel_tol, abs_tol, output_queue):
    result = symbolic_equal(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=2.0, **kwargs):
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        ctx = multiprocessing.get_context("spawn")
    output_queue = ctx.Queue()
    process_args = args + (output_queue,)
    process = ctx.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    if output_queue.empty():
        return False
    return bool(output_queue.get())
import re

def extract_boxed_content(text):
    # Return the LAST NON-EMPTY \boxed{...} content in the text, with brace
    # nesting support. Returns None only when there is no \boxed{ at all,
    # and "" only when every \boxed{...} is empty.
    #
    # History: the original implementation returned the FIRST boxed group
    # (via text.find), but rollout analysis of TeleMath responses showed
    # ~35% of completions emit an EMPTY \boxed{} early as a placeholder
    # ("\boxed{}", "\boxed{answer}", "\boxed{...}") and then put the real
    # answer in a later \boxed{...}. The first-match behaviour graded ~24%
    # of TeleMath samples as wrong even when the actual final answer was
    # correct. True "first wrong value, later corrected" cases (the kind
    # this change could regress on) were ~0.5% in the same sample, so
    # taking the last non-empty answer is a strict improvement.
    if not text:
        return None

    marker = r"\boxed{"
    mlen = len(marker)
    last_non_empty = None
    last_any = None  # remember an empty boxed so we still return "" if only empty ones exist
    n = len(text)
    pos = 0
    while True:
        start = text.find(marker, pos)
        if start == -1:
            break
        i = start + mlen
        brace_count = 1
        content = []
        while i < n and brace_count > 0:
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            if brace_count > 0:
                content.append(text[i])
            i += 1
        # If we ran off the end without closing the brace, treat what we have
        # as the candidate (preserves prior best-effort behaviour).
        cand = "".join(content).strip()
        last_any = cand
        if cand:
            last_non_empty = cand
        pos = i  # continue scanning past this match

    if last_non_empty is not None:
        return last_non_empty
    return last_any  # "" if every boxed was empty
def _test_math_equal():
    # print(math_equal("0.0833333333333333", "\\frac{1}{12}"))
    # print(math_equal("(1,4.5)", "(1,\\frac{9}{2})"))
    # print(math_equal("\\frac{x}{7}+\\frac{2}{7}", "\\frac{x+2}{7}", timeout=True))
    # print(math_equal("\\sec^2(y)", "\\tan^2(y)+1", timeout=True))
    # print(math_equal("\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\end{pmatrix}", "(\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\\\\\end{pmatrix})", timeout=True))

    # pred = '\\begin{pmatrix}\\frac{1}{3x^{2/3}}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\end{pmatrix}'
    # gt = '(\\begin{pmatrix}\\frac{1}{3\\sqrt[3]{x}^2}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\\\\\end{pmatrix})'

    # pred= '-\\frac{8x^2}{9(x^2-2)^{5/3}}+\\frac{2}{3(x^2-2)^{2/3}}'
    # gt= '-\\frac{2(x^2+6)}{9(x^2-2)\\sqrt[3]{x^2-2}^2}'

    # pred =  '-34x-45y+20z-100=0'
    # gt = '34x+45y-20z+100=0'

    # pred = '\\frac{100}{3}'
    # gt = '33.3'

    # pred = '\\begin{pmatrix}0.290243531202435\\\\0.196008371385084\\\\-0.186381278538813\\end{pmatrix}'
    # gt = '(\\begin{pmatrix}0.29\\\\0.196\\\\-0.186\\\\\\end{pmatrix})'

    # pred = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{2\\sqrt{33}+15}'
    # gt = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{15+2\\sqrt{33}}'

    # pred = '(+5)(b+2)'
    # gt = '(a+5)(b+2)'

    # pred = '\\frac{1+\\sqrt{5}}{2}'
    # gt = '2'

    # pred = '\\frac{34}{16}+\\frac{\\sqrt{1358}}{16}', gt = '4'
    # pred = '1', gt = '1\\\\sqrt{19}'

    # pred = "(0.6,2.6667]"
    # gt = "(\\frac{3}{5},\\frac{8}{3}]"

    gt = "asdasd\\boxed{"+str("x+1")+"}"
    pred = "aslkdj \\boxed{1+x} asdasd"
    gt=extract_boxed_content(gt)
    pred=extract_boxed_content(pred)
    print(gt)
    print(pred)
    print(math_equal(pred, gt, timeout=True))


if __name__ == "__main__":
    _test_math_equal()
