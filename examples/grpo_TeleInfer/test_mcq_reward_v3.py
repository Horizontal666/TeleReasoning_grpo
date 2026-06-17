#!/usr/bin/env python
"""Smoke test for MCQ reward v3_teleqna (teleqna_failure_analysis.md §8).

Verifies:
  1. MCQ_REWARD_MODE unset / "binary" → byte-identical to legacy for both
     teleqna and non-teleqna inputs (rollback path preserved).
  2. MCQ_REWARD_MODE=v2 → identical to v2 for both teleqna and non-teleqna
     inputs (no leakage of v3-only keys when v3 is not selected).
  3. MCQ_REWARD_MODE=v3_teleqna AND data_source=teleqna:
       - Correct + no fabrication + no hedging → same as v2 (r4=r5=0)
       - Wrong + "I found a snippet from a study guide" → r4 == -0.30
       - Correct + "answer key:" → r4 == -0.15
       - Wrong + 4 distinct hedging phrases → r5 == -0.15
       - Wrong + 8 hedging phrases → r5 == -0.20 (capped)
       - Correct + 8 hedging phrases → r5 == 0.0 (no penalty on correct)
  4. MCQ_REWARD_MODE=v3_teleqna with non-teleqna data_source (oranbench) →
     v3 penalties do NOT fire (matches v2).

Run:
    /dpc/kuin0100/conda_env/grpo_py311/bin/python \\
        examples/grpo_TeleInfer/test_mcq_reward_v3.py

Stdlib-only; no GPU / heavy deps required.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running both as a script and as a module.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Ensure clean env at import time. compute_mcq_score re-reads the env per call,
# but being explicit avoids surprises if a prior shell exported it.
os.environ.pop("MCQ_REWARD_MODE", None)

from telelogs_symbolic_pkg.telelogs_symbolic import (  # noqa: E402
    compute_mcq_score,
    compute_mcq_score_v2,
    compute_mcq_score_v3_teleqna,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(x):
    if isinstance(x, float):
        return f"{x:+.2f}"
    return str(x)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _call(mode: str, solution_str: str, gt: str, data_source: str | None):
    """Invoke compute_mcq_score with the given env mode + data_source. Cleans
    up the env on return so tests don't leak state."""
    if mode is None:
        os.environ.pop("MCQ_REWARD_MODE", None)
    else:
        os.environ["MCQ_REWARD_MODE"] = mode
    try:
        return compute_mcq_score(
            solution_str=solution_str,
            ground_truth=gt,
            data_source=data_source,
        )
    finally:
        os.environ.pop("MCQ_REWARD_MODE", None)


# ---------------------------------------------------------------------------
# Test cases for v3_teleqna behaviour (data_source = teleqna)
# ---------------------------------------------------------------------------

# Each case is a dict with: name, solution, gt, expected r4, r5, score, acc.
# We'll also compare r1/r2/r3 to the v2 reference (they must be identical to
# v2 for the same input — v3 only layers on top).

V3_TELEQNA_CASES: list[dict] = [
    {
        "name": "correct + clean, no fabrication, no hedging",
        "solution": "Reasoning briefly about options.\n\nANSWER: B",
        "gt": "B",
        # v2: r1=1, r2=0.30, r3=0 → score=1.0
        # v3: r4=0, r5=0 → score still clipped to 1.0
        "expect_r4": 0.0,
        "expect_r5": 0.0,
        "expect_score": 1.0,
        "expect_acc": 1.0,
    },
    {
        "name": "wrong + 'I found a snippet from a study guide' → r4=-0.30",
        # Two fabrication phrases present, but r4 is a single fixed penalty.
        # "I found a snippet" → matches "I found a snippet"
        # "study guide"       → matches "study guide"
        "solution": (
            "I found a snippet from a study guide that says option A.\n"
            "ANSWER: A"
        ),
        "gt": "B",
        # v2: r1=0, r2=0.30 (clean ANSWER:, short), r3=0 (no spec citations)
        # v3: r4=-0.30
        #     r5: count hedging tokens — none of "wait/actually/hmm/..." present
        # score = clip(0 + 0.30 + 0 + (-0.30) + 0, 0, 1) = 0.0
        "expect_r4": -0.30,
        "expect_r5": 0.0,
        "expect_score": 0.0,
        "expect_acc": 0.0,
    },
    {
        "name": "correct + 'answer key:' → r4=-0.15",
        "solution": "The answer key: confirms option C.\n\nANSWER: C",
        "gt": "C",
        # v2: r1=1, r2=0.30, r3=0 (correct → no r3 even if hallucinations)
        # v3: r4=-0.15 (phrase present, but correct), r5=0 (correct)
        # score = clip(1 + 0.30 + 0 + (-0.15) + 0, 0, 1) = 1.0
        "expect_r4": -0.15,
        "expect_r5": 0.0,
        "expect_score": 1.0,
        "expect_acc": 1.0,
    },
    {
        "name": "wrong + 4 hedging phrases (wait, actually, hmm, reconsider) → r5=-0.15",
        # Each phrase appears exactly once. Total occurrences = 4, extras = 3,
        # penalty = 3 * -0.05 = -0.15. Above the cap (-0.20).
        "solution": (
            "Wait, the choice could be A. Actually, hmm, let me look again.\n"
            "I should reconsider this carefully.\n"
            "ANSWER: A"
        ),
        "gt": "B",
        # v2: r1=0, r2=0.30, r3=0
        # v3: r4=0 (no fabrication phrases), r5=-0.15
        # score = clip(0 + 0.30 + 0 + 0 + (-0.15), 0, 1) = 0.15
        "expect_r4": 0.0,
        "expect_r5": -0.15,
        "expect_score": 0.15,
        "expect_acc": 0.0,
    },
    {
        "name": "wrong + 8 hedging phrases → r5=-0.20 (capped)",
        # 8 distinct phrases each appearing once → 8 total, extras=7,
        # raw penalty -0.35, capped at -0.20.
        # NOTE: phrases must be word-boundary-isolated. We avoid "let me
        # reconsider" (which would also match "reconsider"), and we use
        # "but wait" / "but actually" carefully — each contains "wait" /
        # "actually" as substrings but with word-boundaries, both the longer
        # and the shorter phrase will match the same token span, so each
        # "but wait" contributes BOTH a "but wait" match AND a "wait" match.
        # To keep the count exactly 8 (one per phrase listed), we use unique
        # tokens and avoid nesting.
        "solution": (
            "Wait. Actually, I am not sure. Hmm.\n"
            "I should reconsider and re-evaluate this question carefully.\n"
            "Let me reconsider. On second thought, the option might be different.\n"
            "But wait — but actually, the answer is A.\n"
            "ANSWER: A"
        ),
        "gt": "B",
        # Counting (the cap saves us from off-by-one ambiguity):
        #   wait              : "Wait" + "but wait"-wait  ≥ 2
        #   actually          : "Actually" + "but actually"-actually ≥ 2
        #   hmm               : 1
        #   reconsider        : "reconsider" + "let me reconsider"-reconsider ≥ 2
        #   re-evaluate       : 1
        #   let me reconsider : 1
        #   but actually      : 1
        #   but wait          : 1
        #   on second thought : 1
        # Total ≥ 12, extras ≥ 11, penalty (uncapped) ≤ -0.55 → capped at -0.20.
        # v2: r1=0, r2=0.30, r3=0
        # v3: r4=0, r5=-0.20
        # score = clip(0 + 0.30 + 0 + 0 + (-0.20), 0, 1) = 0.10
        "expect_r4": 0.0,
        "expect_r5": -0.20,
        "expect_score": 0.10,
        "expect_acc": 0.0,
    },
    {
        "name": "correct + 8 hedging phrases → r5=0.0 (no penalty on correct)",
        "solution": (
            "Wait. Actually, hmm, let me reconsider this. I should re-evaluate.\n"
            "On second thought, but wait — but actually, after careful "
            "reasoning, the answer is C.\n"
            "ANSWER: C"
        ),
        "gt": "C",
        # v2: r1=1, r2=0.30, r3=0
        # v3: r4=0, r5=0 (correct path — no hedging penalty)
        # score = clip(1 + 0.30 + 0 + 0 + 0, 0, 1) = 1.0
        "expect_r4": 0.0,
        "expect_r5": 0.0,
        "expect_score": 1.0,
        "expect_acc": 1.0,
    },
]


# ---------------------------------------------------------------------------
# Cross-mode no-regression cases (run under binary, v2, and v3_teleqna with
# data_source=teleqna). For binary/v2 they MUST match the existing behaviour
# byte-for-byte.
# ---------------------------------------------------------------------------

ROLLBACK_CASES: list[dict] = [
    {
        "name": "correct teleqna case",
        "solution": "ANSWER: A",
        "gt": "A",
        "data_source": "teleqna",
    },
    {
        "name": "wrong teleqna case",
        "solution": "ANSWER: B",
        "gt": "A",
        "data_source": "teleqna",
    },
    {
        "name": "correct oranbench case",
        "solution": "ANSWER: D",
        "gt": "D",
        "data_source": "oranbench",
    },
    {
        "name": "wrong oranbench case (no fabrication phrases)",
        "solution": "Reasoning shortly.\nANSWER: A",
        "gt": "B",
        "data_source": "oranbench",
    },
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run() -> int:
    failures: list[str] = []

    # ------------------------------------------------------------------
    # 1. Rollback / no-regression checks
    # ------------------------------------------------------------------
    print("=" * 100)
    print(" v3_teleqna no-regression checks")
    print("=" * 100)

    legacy_expected_keys = {"score", "acc", "pred_letter_parsed", "pred_value_parsed"}
    v2_expected_keys = {
        "score", "acc", "r1", "r2", "r3",
        "pred_letter_parsed", "pred_value_parsed",
    }

    for c in ROLLBACK_CASES:
        # ---- binary mode (unset and explicit) ----
        for mode_name in (None, "binary"):
            out = _call(mode_name, c["solution"], c["gt"], c["data_source"])
            keys = set(out.keys())
            label = "unset" if mode_name is None else mode_name
            if keys != legacy_expected_keys:
                failures.append(
                    f"[binary/{label}] {c['name']}: keys={sorted(keys)} "
                    f"!= legacy expected {sorted(legacy_expected_keys)}"
                )
                print(f"  FAIL [binary/{label}] {c['name']}: keys drift")
            else:
                print(
                    f"  OK   [binary/{label}] {c['name']}: "
                    f"score={out['score']:+.2f} acc={out['acc']:+.2f}"
                )

        # ---- v2 mode: must match direct compute_mcq_score_v2 ----
        v2_via_toggle = _call("v2", c["solution"], c["gt"], c["data_source"])
        v2_direct = compute_mcq_score_v2(c["solution"], c["gt"])
        keys = set(v2_via_toggle.keys())
        if keys != v2_expected_keys:
            failures.append(
                f"[v2] {c['name']}: keys={sorted(keys)} "
                f"!= v2 expected {sorted(v2_expected_keys)}"
            )
            print(f"  FAIL [v2] {c['name']}: keys drift")
        else:
            ok = True
            for k in v2_expected_keys:
                if not _close(v2_via_toggle.get(k, 0.0), v2_direct.get(k, 0.0)):
                    failures.append(
                        f"[v2] {c['name']}: {k}={v2_via_toggle.get(k)!r} "
                        f"!= direct v2.{k}={v2_direct.get(k)!r}"
                    )
                    ok = False
            print(
                f"  {'OK' if ok else 'FAIL'}   [v2] {c['name']}: "
                f"score={v2_via_toggle['score']:+.2f}"
            )

        # ---- v3_teleqna mode, non-teleqna data_source → must match v2 ----
        if c["data_source"] != "teleqna":
            v3_out = _call("v3_teleqna", c["solution"], c["gt"], c["data_source"])
            keys = set(v3_out.keys())
            if keys != v2_expected_keys:
                failures.append(
                    f"[v3/non-teleqna] {c['name']}: keys={sorted(keys)} "
                    f"!= v2 expected (no r4/r5 should appear)"
                )
                print(f"  FAIL [v3/non-teleqna] {c['name']}: keys drift")
                continue
            ok = True
            for k in v2_expected_keys:
                if not _close(v3_out.get(k, 0.0), v2_direct.get(k, 0.0)):
                    failures.append(
                        f"[v3/non-teleqna] {c['name']}: {k}={v3_out.get(k)!r} "
                        f"!= direct v2.{k}={v2_direct.get(k)!r}"
                    )
                    ok = False
            print(
                f"  {'OK' if ok else 'FAIL'}   [v3/non-teleqna] {c['name']}: "
                f"score={v3_out['score']:+.2f}"
            )

    # ------------------------------------------------------------------
    # 2. v3_teleqna semantics on teleqna data
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print(" v3_teleqna semantic checks (data_source=teleqna)")
    print("=" * 100)
    v3_expected_keys = {
        "score", "acc", "r1", "r2", "r3", "r4", "r5",
        "pred_letter_parsed", "pred_value_parsed",
    }
    header = (
        f"{'#':>2} {'name':<60} {'r1':>5} {'r2':>5} {'r3':>6} "
        f"{'r4':>6} {'r5':>6} {'score':>6} {'acc':>5}"
    )
    print(header)
    print("-" * 100)

    for i, c in enumerate(V3_TELEQNA_CASES, 1):
        out = _call("v3_teleqna", c["solution"], c["gt"], "teleqna")
        v2_ref = compute_mcq_score_v2(c["solution"], c["gt"])

        keys = set(out.keys())
        if keys != v3_expected_keys:
            failures.append(
                f"[#{i} {c['name']}] keys={sorted(keys)} != v3 expected "
                f"{sorted(v3_expected_keys)}"
            )

        print(
            f"{i:>2} {c['name']:<60} "
            f"{_fmt(out.get('r1', 0.0)):>5} {_fmt(out.get('r2', 0.0)):>5} "
            f"{_fmt(out.get('r3', 0.0)):>6} {_fmt(out.get('r4', 0.0)):>6} "
            f"{_fmt(out.get('r5', 0.0)):>6} {_fmt(out['score']):>6} "
            f"{_fmt(out['acc']):>5}"
        )

        # r1/r2/r3 must equal v2's values exactly (v3 layers, doesn't replace).
        for k in ("r1", "r2", "r3"):
            if not _close(out.get(k, 0.0), v2_ref.get(k, 0.0)):
                failures.append(
                    f"[#{i} {c['name']}] v3.{k}={out.get(k)!r} != "
                    f"v2.{k}={v2_ref.get(k)!r} (v3 must not alter v2 components)"
                )

        for k, expected in (
            ("r4", c["expect_r4"]),
            ("r5", c["expect_r5"]),
            ("score", c["expect_score"]),
            ("acc", c["expect_acc"]),
        ):
            if k not in out:
                failures.append(f"[#{i} {c['name']}] missing key {k!r}")
                continue
            if not _close(out[k], expected):
                failures.append(
                    f"[#{i} {c['name']}] {k}={out[k]!r} != expected {expected!r}"
                )

        # Direct-call sanity: bypassing the env toggle must give the same result.
        direct = compute_mcq_score_v3_teleqna(c["solution"], c["gt"])
        for k in ("score", "acc", "r1", "r2", "r3", "r4", "r5"):
            if not _close(out.get(k, 0.0), direct.get(k, 0.0)):
                failures.append(
                    f"[#{i} {c['name']}] env-toggled v3.{k}={out.get(k)!r} "
                    f"!= direct v3.{k}={direct.get(k)!r}"
                )

    print("-" * 100)

    # ------------------------------------------------------------------
    # 3. v3_teleqna with non-teleqna data_source — penalties MUST NOT fire
    # ------------------------------------------------------------------
    print()
    print("Explicit non-teleqna-under-v3 check:")
    case = {
        "solution": (
            "I found a snippet from a study guide. Wait, actually hmm reconsider.\n"
            "ANSWER: A"
        ),
        "gt": "B",
    }
    # If v3 were (incorrectly) firing for oranbench, the score would drop by r4
    # and r5. We want it to match v2 exactly.
    v3_oran = _call("v3_teleqna", case["solution"], case["gt"], "oranbench")
    v2_oran = compute_mcq_score_v2(case["solution"], case["gt"])
    if set(v3_oran.keys()) != v2_expected_keys:
        failures.append(
            f"[v3/oranbench] keys={sorted(v3_oran.keys())} "
            f"!= v2 expected (no r4/r5 should appear for non-teleqna)"
        )
        print(f"  FAIL: v3 on oranbench leaked r4/r5 keys")
    else:
        ok = True
        for k in v2_expected_keys:
            if not _close(v3_oran.get(k, 0.0), v2_oran.get(k, 0.0)):
                failures.append(
                    f"[v3/oranbench] {k}={v3_oran.get(k)!r} != "
                    f"v2.{k}={v2_oran.get(k)!r}"
                )
                ok = False
        print(
            f"  {'OK' if ok else 'FAIL'}: v3/oranbench score={v3_oran['score']:+.2f} "
            f"matches v2 score={v2_oran['score']:+.2f}"
        )

    # Case-insensitivity sanity for data_source string.
    out_upper = _call("v3_teleqna", "ANSWER: A", "A", "TELEQNA")
    if set(out_upper.keys()) != v3_expected_keys:
        failures.append(
            f"data_source case-insensitivity broken: TELEQNA → "
            f"keys={sorted(out_upper.keys())}"
        )
        print(f"  FAIL: data_source='TELEQNA' did not select v3")
    else:
        print(f"  OK: data_source='TELEQNA' (case-insensitive) selects v3")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if failures:
        print("=" * 100)
        print(f"FAILED: {len(failures)} assertion(s)")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("=" * 100)
    total = len(V3_TELEQNA_CASES) + len(ROLLBACK_CASES)
    print(f"PASSED: {total} cases across binary/v2/v3_teleqna + cross-mode checks")
    return 0


if __name__ == "__main__":
    sys.exit(run())
