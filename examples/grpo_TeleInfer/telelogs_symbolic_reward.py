"""Custom-reward entry for verl 0.8 that mirrors v0.7's telelogs_symbolic wiring.

verl 0.8 loads a single function via:
    reward.custom_reward_function.path=examples/grpo_TeleInfer/telelogs_symbolic_reward.py
    reward.custom_reward_function.name=compute_score_batched
    reward.reward_manager.name=batch          # BatchRewardManager

`BatchRewardManager` calls it as:
    compute_score(data_sources, solution_strs, ground_truths, extra_infos, **reward_kwargs)

This mirrors `recipe/grpo_symbolic/dapo_reward_entry.py` in the v0.7 fork
(myverl0.7.0_telelogReward) — same dispatch, with a TeleMath fallback bound
locally so we don't need verl's internal _default_compute_score plumbing.

Dispatch (inside telelogs_symbolic.compute_score_batched, keyed on
parquet `data_source`):
    telelogs / telelogs_troubleshooting -> R1+R2+R3 deterministic rule reward
    3gpp_working_group                  -> exact working-group label match
    teleqna / oranbench / srsran /      -> ANSWER:$LETTER exact match
       teletable
    telemath / aime* / math500 / ...    -> fallback (prime.math_equal)
"""
from __future__ import annotations

import os
import sys
from typing import Any

# Ensure the sibling package is importable when verl loads us via
# importlib.util.spec_from_file_location (which doesn't add our dir to sys.path).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from telelogs_symbolic_pkg import telelogs_symbolic  # noqa: E402
from telelogs_symbolic_pkg import prime  # noqa: E402

_PRIME_MATH_SOURCES = {
    "real/deepmath-codeverifier",
    "aime24",
    "aime25",
    "math500",
    "telemath",
}


def _normalize_ds(data_source: Any) -> str:
    return str(data_source).strip().lower() if data_source is not None else ""


def _telemath_fallback(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **_kwargs,
) -> float:
    """Scalar TeleMath / math fallback used when telelogs_symbolic doesn't
    recognise a sample's data_source. Returns float in {0.0, 1.0}.
    """
    ds = _normalize_ds(data_source)
    if ds not in _PRIME_MATH_SOURCES:
        return 0.0
    boxed = prime.extract_boxed_content(solution_str or "")
    try:
        ok = prime.math_equal(
            boxed,
            str(ground_truth),
            timeout=True,
            timeout_seconds=2.0,
            rel_tol=1e-2,
            abs_tol=1e-4,
        )
    except Exception:
        ok = False
    return 1.0 if ok else 0.0


def compute_score_batched(*args, **kwargs):
    """Telelogs-symbolic batched scorer with TeleMath fallback bound.

    Equivalent to v0.7's recipe.grpo_symbolic.reward.py wiring:
        partial(telelogs_symbolic.compute_score_batched,
                fallback_score_fn=default_compute_score)

    Use this with the legacy BatchRewardManager (verl/workers/reward_manager/batch.py).
    For verl 0.8's experimental reward_loop managers (naive/dapo/...), use the
    per-sample `compute_score` below instead.
    """
    kwargs.setdefault("fallback_score_fn", _telemath_fallback)
    data_sources = kwargs.get("data_sources", args[0] if args else None)
    results = telelogs_symbolic.compute_score_batched(*args, **kwargs)
    if data_sources is not None and len(results) == len(data_sources):
        for r, ds in zip(results, data_sources):
            if isinstance(r, dict):
                r["data_source"] = str(ds) if ds is not None else ""
    return results


# Union of keys across all dispatch branches. Every per-sample result must
# emit *all* of these keys (with 0.0 default) so that verl 0.8's async
# postprocess can stack them into non_tensor_batch arrays without KeyError.
#   telelogs/3gpp -> {acc, gt_rule_id, pred_choice_parsed, pred_rule_matched, r1, r2, r3, score}
#   mcq (binary)  -> {acc, pred_letter_parsed, pred_value_parsed, score}
#   mcq (v2)      -> {acc, r1, r2, r3, pred_letter_parsed, pred_value_parsed, score}
#                    Active when MCQ_REWARD_MODE=v2 (see telelogs_symbolic.py docstring
#                    and failure_analysis.md §12.5 GRPO-A). The legacy binary branch
#                    still emits r1/r2/r3 = 0.0 via the dispatcher fill below, so the
#                    schema is identical across modes.
#   telemath (bin)-> {acc, score}                                  (legacy path)
#   telemath (v2) -> {acc, r1, r2, r3, r_unit_credit, score, pred_value_parsed}
#                    Active when TELEMATH_REWARD_MODE=v2 (see
#                    telelogs_symbolic.compute_telemath_score docstring +
#                    telemath_failure_analysis.md Part II §17). Legacy binary
#                    branch still emits r1/r2/r3/r_unit_credit = 0.0 via the
#                    dispatcher fill below, so the schema is identical across
#                    modes.
_NUMERIC_KEYS = (
    "score", "acc",
    "r1", "r2", "r3",
    "r_unit_credit",
    "pred_choice_parsed", "pred_rule_matched", "gt_rule_id",
    "pred_letter_parsed", "pred_value_parsed",
)


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **kwargs,
):
    """Per-sample dispatcher for verl 0.8's experimental reward_loop managers
    (naive / dapo / ...). Same dispatch logic as compute_score_batched, but
    on a single sample. Always returns a dict containing the union of keys
    across all branches; per-branch missing keys default to 0.0.
    """
    src = _normalize_ds(data_source)

    if src in {"telelogs", "telelogs_troubleshooting"}:
        r = telelogs_symbolic.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
    elif src in telelogs_symbolic._3GPP_SOURCES:
        r = telelogs_symbolic.compute_3gpp_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
        )
    elif src in telelogs_symbolic._MCQ_SOURCES:
        r = telelogs_symbolic.compute_mcq_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    elif src in telelogs_symbolic._TELEMATH_SOURCES:
        # Explicit TeleMath / math branch. compute_telemath_score reads
        # TELEMATH_REWARD_MODE on every call: binary (default / unset / any
        # value != "v2") is byte-identical to _telemath_fallback below;
        # v2 activates the multi-component shaping reward. The
        # _telemath_fallback path is preserved (it still serves as the
        # ultimate safety net via compute_score_batched's fallback_score_fn).
        r = telelogs_symbolic.compute_telemath_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    else:
        r = _telemath_fallback(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    if not isinstance(r, dict):
        r = {"score": float(r), "acc": float(r)}
    r.setdefault("score", 0.0)
    r.setdefault("acc", r["score"])

    # Rebuild in fixed key order. verl 0.8's agent_loop._postprocess takes
    # `list(reward_extra_infos[0].keys())` from the first sample only, and
    # later DataProto.concat asserts list equality across workers. Branch-
    # dependent insertion order would make workers disagree on the order
    # even when the set is identical.
    out = {k: r.get(k, 0.0) for k in _NUMERIC_KEYS}
    out["data_source"] = str(data_source) if data_source is not None else ""
    return out
