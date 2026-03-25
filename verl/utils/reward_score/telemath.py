import math
import re

# Only look at the tail of the solution when extracting numbers / answers.
_SOLUTION_CLIP_CHARS = 300

# Try to import Math-Verify for robust LaTeX / expression parsing.
# If it's not installed, the code still works, but LaTeX parsing will be more limited.
try:
    from math_verify import parse as mv_parse  # type: ignore[import]
    _HAS_MATH_VERIFY = True
except Exception:  # ImportError or any env-related failure
    mv_parse = None
    _HAS_MATH_VERIFY = False

# Simple pattern for common LaTeX fractions: \frac{a}{b}, \dfrac{a}{b}, \tfrac{a}{b}
_FRAC_PATTERN = re.compile(
    r"^\\(?:dfrac|tfrac|frac)\s*\{?\s*([+-]?\d+)\s*\}?\s*\{?\s*([+-]?\d+)\s*\}?\s*$"
)


def extract_solution(solution_str: str, method: str = "strict"):
    """
    GSM8K-style extraction:
      - method='strict': look for lines like '#### 0.75'
      - method='flexible': take the last numeric token in the tail.
    """
    assert method in ["strict", "flexible"]

    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    else:  # method == "flexible"
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            pass
        else:
            invalid_str = ["", "."]
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break

    return final_answer


def _last_boxed_substring(s: str):
    """
    Find the last '\\boxed{...}' substring, with a simple brace-matching
    algorithm so that nested braces inside the box still work.
    Returns the full substring (e.g. '\\boxed{\\dfrac{3}{4}}') or None.
    """
    if not s:
        return None

    idx = s.rfind(r"\boxed")
    if idx < 0:
        return None

    # Move from '\boxed' to the first '{'
    i = idx
    while i < len(s) and s[i] != "{":
        i += 1

    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(s):
        if s[i] == "{":
            num_left_braces_open += 1
        elif s[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    return s[idx : right_brace_idx + 1]


def _strip_boxed(boxed: str):
    """
    Remove the leading '\\boxed' wrapper from a substring like '\\boxed{...}'.
    """
    if boxed is None:
        return None

    s = boxed.strip()
    if s.startswith(r"\boxed "):
        # Rare style: '\boxed x'
        return s[len(r"\boxed ") :]
    if s.startswith(r"\boxed{") and s.endswith("}"):
        return s[len(r"\boxed{") : -1]
    return s


def extract_boxed_answer(solution_str: str):
    """
    Extract the final LaTeX answer from the model output.

    Heuristic:
      1. Search from the end for '### Final Result' or '### Final Answer'.
         If found, only consider the substring from that header to the end.
      2. In that region, take the last '\\boxed{...}'.
      3. If no header is found, fall back to the last '\\boxed{...}' in the tail.
    The content inside the last '\\boxed{...}' is returned.
    """
    if not solution_str:
        return None

    search_region = solution_str

    # From back to front, locate the last marker like '### Final Result' or '### Final Answer'.
    last_header_idx = -1
    for marker in ("### Final Result", "### Final Answer"):
        idx = search_region.rfind(marker)
        if idx != -1 and idx > last_header_idx:
            last_header_idx = idx

    if last_header_idx != -1:
        search_region = search_region[last_header_idx:]
    elif len(search_region) > _SOLUTION_CLIP_CHARS:
        search_region = search_region[-_SOLUTION_CLIP_CHARS:]

    boxed = _last_boxed_substring(search_region)
    if boxed is None:
        return None

    content = _strip_boxed(boxed)
    if not content:
        return None

    content = content.strip()

    # Strip simple $...$ if the expression is wrapped again
    if content.startswith("$") and content.endswith("$") and len(content) >= 2:
        content = content[1:-1].strip()

    return content or None


def _to_float(value):
    """
    Best-effort conversion of a plain numeric string to float.
    Handles commas and simple '$...$' wrappers.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    s = s.replace(",", "")

    # Strip surrounding $ for simple cases like '$0.75$'
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()

    try:
        return float(s)
    except Exception:
        return None


def _parse_simple_fraction(expr: str):
    """
    Handle very simple fractions:
      - LaTeX: \\frac{3}{4}, \\dfrac{3}{4}, \\tfrac{3}{4}
      - Plain: 3/4
    Returns float or None.
    """
    if expr is None:
        return None

    s = str(expr).strip()
    if not s:
        return None

    # LaTeX-style fraction
    m = _FRAC_PATTERN.match(s)
    if m:
        try:
            num = float(m.group(1))
            den = float(m.group(2))
            if den == 0:
                return None
            return num / den
        except Exception:
            return None

    # Plain "a/b"
    if "/" in s:
        num_s, den_s = s.split("/", 1)
        num_s = num_s.strip()
        den_s = den_s.strip()
        if num_s and den_s:
            try:
                num = float(num_s)
                den = float(den_s)
                if den == 0:
                    return None
                return num / den
            except Exception:
                return None

    return None


def _eval_expr_to_float(expr_str: str):
    """
    Convert an expression string to a float, using (in order):
      1. Math-Verify's parse(), if available (robust LaTeX / expression support).
      2. Simple fraction parsing (LaTeX and plain 'a/b').
      3. Bare float parsing.
    """
    if expr_str is None:
        return None

    s = str(expr_str).strip()
    if not s:
        return None

    # 1) Best effort: Math-Verify, if installed.
    if _HAS_MATH_VERIFY and mv_parse is not None:
        try:
            # Signature differences across versions:
            # - Some use parse(pred=..., parsing_timeout=...)
            # - Others use parse(text, parsing_timeout=...)
            try:
                parsed = mv_parse(pred=s, parsing_timeout=None)
            except TypeError:
                parsed = mv_parse(s, parsing_timeout=None)

            parsed_expr = None
            if isinstance(parsed, (list, tuple)):
                # Take the first non-None candidate if a list is returned.
                for item in parsed:
                    if item is not None:
                        parsed_expr = item
                        break
            else:
                parsed_expr = parsed

            if parsed_expr is not None:
                try:
                    # SymPy-style object
                    val = float(parsed_expr.evalf())  # type: ignore[attr-defined]
                except Exception:
                    # Fall back to direct float conversion
                    val = float(parsed_expr)
                if math.isfinite(val):
                    return val
        except Exception:
            # Any Math-Verify parsing failure just falls back to simpler logic.
            pass

    # 2) Simple fraction parsing (works even without Math-Verify).
    frac_val = _parse_simple_fraction(s)
    if frac_val is not None:
        return frac_val

    # 3) Final fallback: plain float.
    return _to_float(s)


def own_score(
    solution_str: str,
    ground_truth,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-4,
) -> float:
    """
    GSM8K-style numeric scorer with tolerance.

    Uses extract_solution (looking for '####') to get a numeric string, then
    checks if the prediction is within abs_tol / rel_tol of the ground truth.
    """
    answer_str = extract_solution(solution_str=solution_str, method=method)
    if answer_str is None:
        return 0.0

    pred = _to_float(answer_str)
    gt = _to_float(ground_truth)

    if pred is None or gt is None:
        # Cannot parse as numbers at all → 0 reward.
        return 0.0

    abs_err = abs(pred - gt)
    rel_err = abs_err / max(1.0, abs(gt))

    if abs_err <= abs_tol or rel_err <= rel_tol:
        return float(score)
    else:
        return float(format_score)


def math_score(
    solution_str: str,
    ground_truth,
    score: float = 1.0,
    format_score: float = 0.0,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-4,
) -> float:
    """
    LaTeX-based scorer using the final '\\boxed{...}' answer.

    Steps:
      1. Extract the last '\\boxed{...}' after '### Final Result' / '### Final Answer'
         (or anywhere in the tail if no header is found).
      2. Convert both predicted and ground-truth answers to floats using
         `_eval_expr_to_float`, which can use Math-Verify if installed.
      3. Compute absolute and relative error:
           abs_err = |pred - gt|
           rel_err = abs_err / max(1, |gt|)
         If abs_err <= abs_tol OR rel_err <= rel_tol, return `score`,
         otherwise return `format_score`.
    """
    boxed_expr = extract_boxed_answer(solution_str)
    if boxed_expr is None:
        return 0.0

    pred = _eval_expr_to_float(boxed_expr)
    gt = _eval_expr_to_float(str(ground_truth))

    if pred is None or gt is None:
        # Could not interpret as numeric expressions.
        return 0.0

    abs_err = abs(pred - gt)
    rel_err = abs_err / max(1.0, abs(gt))

    # This is exactly the "relative_error < tau" logic you described,
    # combined with a small absolute tolerance for very small ground truths.
    if abs_err <= abs_tol or rel_err <= rel_tol:
        return float(score)
    else:
        return float(format_score)


def compute_score(
    solution_str: str,
    ground_truth,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-4,
    use_math_verify: bool = True,
) -> float:
    """
    The main scoring entry point for VeRL.

    Behavior:
      - Computes a GSM8K-style tolerant numeric score via `own_score` (####).
      - Computes a LaTeX-based tolerant numeric score via `math_score` (\\boxed{}),
        which can leverage Math-Verify when installed.
      - Returns the maximum of the two scores (so adding Math-Verify cannot
        reduce the reward; it only provides more opportunities to match).

    This keeps the interface compatible with:
        from . import telemath
        return telemath.compute_score(solution_str, ground_truth)
    """
    base = own_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=score,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

    mv = 0.0
    if use_math_verify:
        mv = math_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            score=score,
            format_score=format_score,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )

    final = max(base, mv)
    return final
