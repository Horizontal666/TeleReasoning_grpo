import math
import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    """Extract the final numeric answer from the solution string.

    Mirrors the GSM8K reward function parsing logic, using the '####' convention
    for the final answer.
    """
    assert method in ["strict", "flexible"]

    # Focus on the tail of the solution where the final answer usually appears
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # This also tests the formatting of the model
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # Take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # No reward if there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # Find the last number that is not just '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break

    return final_answer


def _to_float(value):
    """Best-effort conversion of a value to a finite float; returns None on failure."""
    # Already numeric
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
        return None

    # Expect ground_truth usually as a string in VERL parquet files
    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None

    # Remove common formatting characters
    s = s.replace(",", "").replace("$", "")

    try:
        result = float(s)
    except ValueError:
        return None

    if not math.isfinite(result):
        return None

    return result


def compute_score(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.0,
    score=1.0,
    abs_tol=1e-4,
    rel_tol=1e-4,
):
    """The scoring function for telemath-style numeric tasks.

    Extends the GSM8K-style scorer with numeric tolerance, so answers that are
    numerically close (within abs_tol / rel_tol) are treated as correct.

    Args:
        solution_str: the solution text.
        ground_truth: the ground truth numeric answer (string or number).
        method: the method to extract the solution, choices are 'strict' and 'flexible'.
        format_score: the score when the format is correct (a numeric answer can be
            parsed) but the value is outside tolerance.
        score: the score for a numerically correct answer.
        abs_tol: absolute tolerance for numeric comparison.
        rel_tol: relative tolerance for numeric comparison, scaled by
            max(1.0, abs(ground_truth)).

    Returns:
        A scalar reward:
            * 0.0 if no answer or unparsable / non-finite numeric values
            * `score` if the numeric answer is within tolerance
            * `format_score` otherwise (format OK, value incorrect)
    """
    # Reuse GSM8K-style extraction
    answer = extract_solution(solution_str=solution_str, method=method)

    # No parsed answer -> no reward
    if answer is None:
        return 0.0

    # Convert both prediction and ground truth to floats
    pred_val = _to_float(answer)
    gt_val = _to_float(ground_truth)

    # If either cannot be parsed as a finite float, treat as no valid answer
    if pred_val is None or gt_val is None:
        return 0.0

    # Guard against negative tolerances
    if abs_tol < 0.0:
        abs_tol = 0.0
    if rel_tol < 0.0:
        rel_tol = 0.0

    abs_err = abs(pred_val - gt_val)

    # Check absolute tolerance first
    if abs_err <= abs_tol:
        return score

    # Then check relative tolerance w.r.t. true answer magnitude
    denom = max(1.0, abs(gt_val))
    rel_err = abs_err / denom
    if rel_err <= rel_tol:
        return score

    # Parsed a number, but it is outside tolerance -> format-only reward
    return format_score
