#!/usr/bin/env python
"""Smoke test for MCQ reward v2 (failure_analysis.md §12.5 GRPO-A).

Verifies:
  1. With MCQ_REWARD_MODE unset (default), compute_mcq_score behaves
     BYTE-IDENTICALLY to the legacy binary scorer for all test cases — this
     is the core safety guarantee for the no-regression rollout.
  2. With MCQ_REWARD_MODE=v2, the auxiliary r1/r2/r3 components fire as
     expected on hand-crafted cases covering:
       - correct + clean format
       - correct + verbose (long output, no runaway)
       - wrong + terse, no citations
       - wrong + 1-2 citation-shaped fragments
       - wrong + 3+ citation-shaped fragments
       - runaway-repetition loop
       - empty output
       - near-miss (correct letter but verbose with citations - still correct)

Run:
    /dpc/kuin0100/conda_env/grpo_py311/bin/python \\
        examples/grpo_TeleInfer/test_mcq_reward_v2.py

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

# Force binary mode for the initial import; we'll flip it per-case via os.environ
# below since compute_mcq_score re-reads the env on every call.
os.environ.pop("MCQ_REWARD_MODE", None)

from telelogs_symbolic_pkg.telelogs_symbolic import (  # noqa: E402
    compute_mcq_score,
    compute_mcq_score_v2,
)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def _runaway_solution(letter: str = "A") -> str:
    # 50 repetitions of a single short token -> uniq/total ratio ~ 1/50 < 0.15
    return ("Item " * 400) + f"\nANSWER: {letter}"


def _verbose_with_n_citations(n: int, letter: str = "A") -> str:
    """Build a verbose wrong answer with n distinct citation fragments."""
    cites = [
        "O-RAN.WG4.MP.0-v09.00",
        "srsran/phy/upper/channel_coding/ldpc.cpp",
        "Section 6.2.2.3 of the spec",
        "RFC 3261",
        "3GPP TS 38.331",
        "O-RAN.WG9.XTRP-v02.0",
    ][:n]
    body = "Based on the specification, "
    body += " and ".join(cites) if cites else "no spec citations are used here"
    body += " applies to this question.\n"
    body += "Therefore the answer is the most likely option.\n\n"
    body += f"ANSWER: {letter}"
    return body


TEST_CASES: list[dict] = [
    {
        "name": "correct + clean format",
        "solution": "Reasoning briefly about options, B looks right.\n\nANSWER: B",
        "gt": "B",
        # legacy: 1.0 (correct)
        # v2: r1=1, r2=0.3, r3=0 → score=1.0 (clipped)
        "expect_legacy_score": 1.0,
        "expect_v2_r1": 1.0,
        "expect_v2_r2": 0.30,
        "expect_v2_r3": 0.0,
        "expect_v2_score": 1.0,
        "expect_v2_acc": 1.0,
    },
    {
        "name": "correct + verbose (citations, but correct so no penalty)",
        "solution": _verbose_with_n_citations(4, letter="C"),
        "gt": "C",
        "expect_legacy_score": 1.0,
        "expect_v2_r1": 1.0,
        "expect_v2_r2": 0.30,    # short < 4000 chars, clean trailing ANSWER: C
        "expect_v2_r3": 0.0,     # correct -> no hallucination penalty
        "expect_v2_score": 1.0,
        "expect_v2_acc": 1.0,
    },
    {
        "name": "wrong + terse, no citations",
        "solution": "Let me think...\nANSWER: A",
        "gt": "B",
        "expect_legacy_score": 0.0,
        "expect_v2_r1": 0.0,
        "expect_v2_r2": 0.30,    # clean trailing ANSWER, short
        "expect_v2_r3": 0.0,     # no citations
        "expect_v2_score": 0.30,
        "expect_v2_acc": 0.0,
    },
    {
        "name": "wrong + 2 fake citations (light penalty)",
        "solution": _verbose_with_n_citations(2, letter="A"),
        "gt": "B",
        "expect_legacy_score": 0.0,
        "expect_v2_r1": 0.0,
        "expect_v2_r2": 0.30,
        "expect_v2_r3": -0.10,
        "expect_v2_score": 0.20,
        "expect_v2_acc": 0.0,
    },
    {
        "name": "wrong + 3+ fake citations (heavy penalty)",
        "solution": _verbose_with_n_citations(5, letter="A"),
        "gt": "B",
        "expect_legacy_score": 0.0,
        "expect_v2_r1": 0.0,
        "expect_v2_r2": 0.30,
        "expect_v2_r3": -0.30,
        # score = clip(0 + 0.30 + (-0.30), 0, 1) = 0.0
        "expect_v2_score": 0.0,
        "expect_v2_acc": 0.0,
    },
    {
        "name": "runaway-repetition loop (wrong, no citations)",
        "solution": _runaway_solution("A"),
        "gt": "B",
        "expect_legacy_score": 0.0,
        "expect_v2_r1": 0.0,
        "expect_v2_r2": 0.0,     # runaway tail -> 0
        "expect_v2_r3": 0.0,
        "expect_v2_score": 0.0,
        "expect_v2_acc": 0.0,
    },
    {
        "name": "empty output",
        "solution": "",
        "gt": "B",
        "expect_legacy_score": 0.0,
        "expect_v2_r1": 0.0,
        "expect_v2_r2": 0.0,     # empty -> 0
        "expect_v2_r3": 0.0,
        "expect_v2_score": 0.0,
        "expect_v2_acc": 0.0,
    },
    {
        "name": "near-miss: correct letter but with 3+ citations (still correct)",
        "solution": _verbose_with_n_citations(3, letter="D"),
        "gt": "D",
        "expect_legacy_score": 1.0,
        "expect_v2_r1": 1.0,
        "expect_v2_r2": 0.30,
        "expect_v2_r3": 0.0,     # correct -> no penalty even with citations
        "expect_v2_score": 1.0,
        "expect_v2_acc": 1.0,
    },
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _fmt(x):
    if isinstance(x, float):
        return f"{x:+.2f}"
    return str(x)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def run() -> int:
    failures: list[str] = []
    print("=" * 100)
    print(" MCQ reward smoke test — legacy vs v2 side-by-side")
    print("=" * 100)
    print(f"{'#':>2} {'name':<55} {'gt':>3} {'leg.score':>10} "
          f"{'v2.r1':>6} {'v2.r2':>6} {'v2.r3':>7} {'v2.score':>9} {'v2.acc':>7}")
    print("-" * 100)

    for i, c in enumerate(TEST_CASES, 1):
        # ---- Legacy mode (env unset) ----
        os.environ.pop("MCQ_REWARD_MODE", None)
        legacy = compute_mcq_score(c["solution"], c["gt"])

        # ---- v2 mode ----
        os.environ["MCQ_REWARD_MODE"] = "v2"
        try:
            v2 = compute_mcq_score(c["solution"], c["gt"])
        finally:
            os.environ.pop("MCQ_REWARD_MODE", None)

        # Direct v2 call (bypassing toggle) should equal the toggled result.
        v2_direct = compute_mcq_score_v2(c["solution"], c["gt"])

        print(f"{i:>2} {c['name']:<55} {c['gt']:>3} "
              f"{_fmt(legacy['score']):>10} "
              f"{_fmt(v2['r1']):>6} {_fmt(v2['r2']):>6} {_fmt(v2['r3']):>7} "
              f"{_fmt(v2['score']):>9} {_fmt(v2['acc']):>7}")

        # ---- Assertions ----
        # 1. Legacy must match expectations.
        if not _close(legacy["score"], c["expect_legacy_score"]):
            failures.append(
                f"[#{i} {c['name']}] legacy.score={legacy['score']!r} "
                f"!= expected {c['expect_legacy_score']!r}"
            )

        # 2. Legacy must NOT contain r1/r2/r3 keys (byte-identical to original
        #    behaviour where these keys were absent and back-filled by the
        #    dispatcher with 0.0).
        for k in ("r1", "r2", "r3"):
            if k in legacy:
                failures.append(
                    f"[#{i} {c['name']}] legacy mode leaked key {k!r} "
                    f"(broken byte-identity guarantee)"
                )

        # 3. v2 must populate r1/r2/r3 explicitly with expected values.
        for k, expected in (
            ("r1", c["expect_v2_r1"]),
            ("r2", c["expect_v2_r2"]),
            ("r3", c["expect_v2_r3"]),
            ("score", c["expect_v2_score"]),
            ("acc", c["expect_v2_acc"]),
        ):
            if k not in v2:
                failures.append(f"[#{i} {c['name']}] v2 missing key {k!r}")
                continue
            if not _close(v2[k], expected):
                failures.append(
                    f"[#{i} {c['name']}] v2.{k}={v2[k]!r} != expected {expected!r}"
                )

        # 4. v2 via env-toggle must equal v2 via direct call (no caching bug).
        for k in ("score", "acc", "r1", "r2", "r3"):
            if not _close(v2.get(k, 0.0), v2_direct.get(k, 0.0)):
                failures.append(
                    f"[#{i} {c['name']}] env-toggled v2.{k}={v2.get(k)!r} "
                    f"!= direct v2.{k}={v2_direct.get(k)!r}"
                )

    print("-" * 100)

    # ---- Cross-cutting check: legacy default truly is byte-identical ----
    print()
    print("Byte-identity check (legacy default vs untouched compute_mcq_score):")
    # The legacy dict for a correct sample must contain EXACTLY these keys.
    os.environ.pop("MCQ_REWARD_MODE", None)
    sample = compute_mcq_score("ANSWER: A", "A")
    expected_keys = {"score", "acc", "pred_letter_parsed", "pred_value_parsed"}
    actual_keys = set(sample.keys())
    if actual_keys != expected_keys:
        failures.append(
            f"legacy default key set drifted: got {sorted(actual_keys)}, "
            f"expected {sorted(expected_keys)}"
        )
        print(f"  FAIL: keys={sorted(actual_keys)} vs expected={sorted(expected_keys)}")
    else:
        print(f"  OK: legacy key set = {sorted(actual_keys)}")

    # Toggle that ISN'T "v2" must also fall through to legacy.
    os.environ["MCQ_REWARD_MODE"] = "binary"
    bin_sample = compute_mcq_score("ANSWER: A", "A")
    os.environ.pop("MCQ_REWARD_MODE", None)
    if set(bin_sample.keys()) != expected_keys:
        failures.append(
            f"MCQ_REWARD_MODE=binary did not select legacy: keys={sorted(bin_sample.keys())}"
        )
        print(f"  FAIL: MCQ_REWARD_MODE=binary -> keys={sorted(bin_sample.keys())}")
    else:
        print(f"  OK: MCQ_REWARD_MODE=binary -> legacy keys")

    # Unknown mode values must also fall through.
    os.environ["MCQ_REWARD_MODE"] = "something_else"
    unk_sample = compute_mcq_score("ANSWER: A", "A")
    os.environ.pop("MCQ_REWARD_MODE", None)
    if set(unk_sample.keys()) != expected_keys:
        failures.append(
            f"MCQ_REWARD_MODE=something_else did not select legacy: "
            f"keys={sorted(unk_sample.keys())}"
        )
        print(f"  FAIL: MCQ_REWARD_MODE=unknown -> keys={sorted(unk_sample.keys())}")
    else:
        print(f"  OK: MCQ_REWARD_MODE=unknown -> legacy keys")

    print()
    if failures:
        print("=" * 100)
        print(f"FAILED: {len(failures)} assertion(s)")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("=" * 100)
    print(f"PASSED: all {len(TEST_CASES)} test cases + 3 byte-identity checks")
    return 0


if __name__ == "__main__":
    sys.exit(run())
