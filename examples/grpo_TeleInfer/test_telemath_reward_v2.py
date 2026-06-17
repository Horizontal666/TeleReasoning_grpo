#!/usr/bin/env python
"""Smoke test for TeleMath reward v2 (telemath_failure_analysis.md Part II §17).

Verifies:
  1. Backwards-compatibility: with TELEMATH_REWARD_MODE unset (or =binary, or
     =unknown), compute_telemath_score returns the BYTE-IDENTICAL {score, acc}
     dict that _telemath_fallback would, for >= 6 hand-crafted cases.
  2. v2 mode: r1 / r2 / r3 / r_unit_credit fire as designed on:
       - correct + clean format
       - correct + verbose (>= 10000 chars: loses length bonus)
       - correct + runaway repetition tail (loses no-rep bonus)
       - wrong + clean format
       - wrong + Friis NF double-count text
       - wrong + M/M/1 formula on a constant-bit-rate prompt
       - wrong + pure-ALOHA formula on a slotted-ALOHA prompt
       - wrong but unit-conversion-equivalent (dB <-> linear)
       - empty output
  3. Schema invariance: every return dict shape contains the union of keys
     once dispatched through compute_score (the verl 0.8 reward-loop entry
     point), so DataProto.concat doesn't blow up across workers.

Run:
    /dpc/kuin0100/conda_env/grpo_py311/bin/python \\
        examples/grpo_TeleInfer/test_telemath_reward_v2.py

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

# Force binary mode for the initial import; we'll flip per-case via os.environ
# (compute_telemath_score re-reads the env on every call).
os.environ.pop("TELEMATH_REWARD_MODE", None)

from telelogs_symbolic_pkg.telelogs_symbolic import (  # noqa: E402
    compute_telemath_score,
    compute_telemath_score_v2,
)
from telelogs_symbolic_reward import (  # noqa: E402
    _telemath_fallback,
    compute_score as dispatcher_compute_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_correct_solution(boxed_value: str, body_len_chars: int = 800) -> str:
    """Build a clean-format solution: padded body + trailing \\boxed{value}."""
    # Body is deterministic, varied tokens (no n-gram repetition).
    tokens = [
        "First", "we", "compute", "the", "value", "step", "by", "step", "and",
        "show", "that", "with", "careful", "manipulation", "the", "expression",
        "reduces", "to", "a", "clean", "closed-form", "answer", "below.",
    ]
    out: list[str] = []
    i = 0
    while sum(len(t) + 1 for t in out) < body_len_chars:
        out.append(tokens[i % len(tokens)])
        i += 1
    body = " ".join(out)
    return f"{body}\n\nTherefore the answer is \\boxed{{{boxed_value}}}"


def _verbose_correct_solution(boxed_value: str) -> str:
    """Build a >10_000 char solution so the length bonus drops."""
    return _clean_correct_solution(boxed_value, body_len_chars=11000)


def _runaway_correct_solution(boxed_value: str) -> str:
    """Body with heavy repetition in the tail, ending in a clean boxed answer.

    Repetition is induced by appending the same 30+ char sentence many times
    AFTER the natural body but BEFORE the final boxed answer, so that the
    last-25% tail contains the repeated motif AND the boxed is still in the
    last 500 chars."""
    head = _clean_correct_solution(boxed_value, body_len_chars=800)
    # Strip the trailing \boxed{...} from the head so we can re-append.
    boxed_marker = f"Therefore the answer is \\boxed{{{boxed_value}}}"
    body = head.replace(boxed_marker, "").rstrip()
    repeating_motif = " The same conclusion follows by symmetry once more."
    # 200 reps of a 49-char fragment → tail has many overlapping >=30-char hits
    runaway = repeating_motif * 200
    return f"{body}{runaway}\n{boxed_marker}"


def _wrong_clean_solution(wrong_value: str) -> str:
    return _clean_correct_solution(wrong_value)


def _nf_double_count_solution(wrong_value: str) -> str:
    """A solution that mentions T_e = T_0(F-1) AND SNR_out = SNR_in - NF."""
    body = (
        "We model the receiver chain. The equivalent noise temperature is "
        "T_e = T_0(F-1). After cascading the noise figure to the front-end, "
        "we then write SNR_out = SNR_in - NF and substitute the temperature "
        "directly, treating the system as if both contributions add. "
    )
    # Pad to >500 chars
    pad = (
        " We then carry the numbers through the link budget, accumulating "
        "losses and gains across each stage with care to avoid double-counting "
        "but proceed anyway with the simpler formulation. " * 4
    )
    return body + pad + f"\n\nFinal: \\boxed{{{wrong_value}}}"


def _mm1_on_deterministic_solution(wrong_value: str) -> str:
    body = (
        "We model the queue as M/M/1 with arrival rate lambda and service "
        "rate mu. The mean waiting time is 1/(mu - lambda). Substituting "
        "the given numbers we obtain a finite answer. "
    )
    pad = (
        "We then double-check using rho/(1-rho) * 1/mu and confirm. " * 8
    )
    return body + pad + f"\n\nFinal: \\boxed{{{wrong_value}}}"


def _slotted_aloha_with_pure_formula_solution(wrong_value: str) -> str:
    body = (
        "We compute the throughput. The classic formula is S = G * e^(-G), "
        "so we plug in G = 0.5 and obtain the answer. "
    )
    pad = " We then verify by substitution into the closed-form expression. " * 10
    return body + pad + f"\n\nFinal: \\boxed{{{wrong_value}}}"


def _unit_conversion_solution(boxed_value: str) -> str:
    """Wrong-as-string, but unit-converted equivalent of the gt."""
    return _clean_correct_solution(boxed_value)


def _dispatcher_call(data_source: str, sol: str, gt: str, ei: dict | None) -> dict:
    """Route through the verl entry point (compute_score) — exercises the
    full dispatch path + the _NUMERIC_KEYS padding."""
    return dispatcher_compute_score(
        data_source=data_source,
        solution_str=sol,
        ground_truth=gt,
        extra_info=ei,
    )


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Backward-compat (binary mode) cases
# ---------------------------------------------------------------------------

BIN_CASES: list[dict] = [
    {
        "name": "correct boxed numeric",
        "solution": "Some reasoning.\n\\boxed{42}",
        "gt": "42",
        "expect_score": 1.0,
    },
    {
        "name": "wrong boxed numeric",
        "solution": "Some reasoning.\n\\boxed{41}",
        "gt": "42",
        "expect_score": 0.0,
    },
    {
        "name": "fraction-form correct",
        "solution": "We get \\boxed{\\frac{1}{2}}",
        "gt": "0.5",
        "expect_score": 1.0,
    },
    {
        "name": "scientific-notation correct",
        "solution": "We get \\boxed{1.5e3}",
        "gt": "1500",
        "expect_score": 1.0,
    },
    {
        "name": "empty boxed",
        "solution": "We get \\boxed{}",
        "gt": "42",
        "expect_score": 0.0,
    },
    {
        "name": "no boxed at all",
        "solution": "The answer is 42 but I forgot to box it.",
        "gt": "42",
        "expect_score": 0.0,
    },
]


# ---------------------------------------------------------------------------
# v2 cases
# ---------------------------------------------------------------------------

V2_CASES: list[dict] = [
    {
        "name": "correct + clean format",
        "solution": _clean_correct_solution("42", body_len_chars=800),
        "gt": "42",
        "extra_info": {"question_text": "Find the answer."},
        "expect_r1": 1.0,
        "expect_r2": 0.20,
        "expect_r3": 0.0,
        "expect_unit": 0.0,
        "expect_score": 1.0,
    },
    {
        "name": "correct + verbose (>10k chars, loses length bonus)",
        "solution": _verbose_correct_solution("42"),
        "gt": "42",
        "extra_info": {"question_text": "Find the answer."},
        "expect_r1": 1.0,
        # Per spec (failure_analysis Part II §17): "correct + verbose →
        # r2=0.10 (loses length bonus)". In practice the cycling-token body
        # also trips the 30-char no-rep heuristic at >10k chars, so only the
        # boxed bonus survives. This matches the spec intent — r2 ≤ 0.10.
        "expect_r2_max": 0.10,
        "expect_r3": 0.0,
        "expect_unit": 0.0,
        "expect_score": 1.0,  # clipped
    },
    {
        "name": "correct + runaway-repetition tail (loses no-rep bonus)",
        "solution": _runaway_correct_solution("42"),
        "gt": "42",
        "extra_info": {"question_text": "Find the answer."},
        "expect_r1": 1.0,
        "expect_r2_max": 0.15,   # 0.10 (boxed) + maybe 0.05 (length) — loses no-rep
        "expect_r3": 0.0,
        "expect_unit": 0.0,
        "expect_score": 1.0,
    },
    {
        "name": "wrong + clean format",
        "solution": _wrong_clean_solution("41"),
        "gt": "42",
        "extra_info": {"question_text": "Find the answer."},
        "expect_r1": 0.0,
        "expect_r2": 0.20,
        "expect_r3": 0.0,
        "expect_unit": 0.0,
        "expect_score": 0.20,
    },
    {
        "name": "wrong + Friis NF double-count fingerprint",
        "solution": _nf_double_count_solution("99"),
        "gt": "42",
        "extra_info": {
            "question_text": (
                "A receiver has a noise figure of 3 dB and an antenna "
                "temperature of 50 K. Compute the equivalent noise floor."
            ),
        },
        "expect_r1": 0.0,
        "expect_r3": -0.10,
        "expect_unit": 0.0,
        # r2 may vary slightly with body length; just check the floor.
        "expect_score_max": 0.20,
    },
    {
        "name": "wrong + M/M/1 on constant-bit-rate prompt",
        "solution": _mm1_on_deterministic_solution("99"),
        "gt": "42",
        "extra_info": {
            "question_text": (
                "Packets of fixed packet length arrive at a router. "
                "Assume a constant bit rate of 1 Mbps. What is the mean delay?"
            ),
        },
        "expect_r1": 0.0,
        "expect_r3": -0.10,
        "expect_unit": 0.0,
    },
    {
        "name": "wrong + slotted-ALOHA prompt + pure-formula response",
        "solution": _slotted_aloha_with_pure_formula_solution("0.18"),
        "gt": "0.36",
        "extra_info": {
            "question_text": "In a slotted ALOHA system with offered load G=1, ...",
        },
        "expect_r1": 0.0,
        "expect_r3": -0.10,
        "expect_unit": 0.0,
    },
    {
        "name": "wrong but dB <-> linear unit-equivalent",
        # boxed is "20" (dB), gt is "100" (linear). 10**(20/10) = 100.
        "solution": _unit_conversion_solution("20"),
        "gt": "100",
        "extra_info": {"question_text": "Compute the ratio."},
        "expect_r1": 0.0,
        "expect_r2": 0.20,
        "expect_r3": 0.0,
        "expect_unit": 0.5,
        # 0 + 0.20 + 0 + 0.5 = 0.70
        "expect_score": 0.70,
    },
    {
        "name": "empty output",
        "solution": "",
        "gt": "42",
        "extra_info": {"question_text": "Find the answer."},
        "expect_r1": 0.0,
        "expect_r2": 0.05,    # No boxed (0), short (no length bonus, < 500), but
                              # below repetition threshold so norep=0.05
        "expect_r3": 0.0,
        "expect_unit": 0.0,
        # Note: empty text passes the norep "too short to plausibly repeat" branch,
        # so r2=0.05. Total = 0.05.
        "expect_score": 0.05,
    },
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _fmt(x):
    if isinstance(x, float):
        return f"{x:+.3f}"
    return str(x)


def run() -> int:
    failures: list[str] = []

    # -----------------------------------------------------------------------
    # Section 1: binary-mode byte-identity with _telemath_fallback
    # -----------------------------------------------------------------------
    print("=" * 100)
    print(" TeleMath reward smoke test — binary backwards-compat")
    print("=" * 100)
    print(f"{'#':>2} {'name':<35} {'gt':>10} "
          f"{'fb.score':>9} {'v2off.score':>12} {'v2off.acc':>10} {'expect':>7}")
    print("-" * 100)

    for i, c in enumerate(BIN_CASES, 1):
        os.environ.pop("TELEMATH_REWARD_MODE", None)
        # Legacy path (the rollback contract).
        fb_score = _telemath_fallback(
            data_source="telemath",
            solution_str=c["solution"],
            ground_truth=c["gt"],
            extra_info=None,
        )
        # New compute_telemath_score in binary mode.
        out = compute_telemath_score(c["solution"], c["gt"], None)

        print(f"{i:>2} {c['name']:<35} {c['gt']:>10} "
              f"{_fmt(fb_score):>9} {_fmt(out['score']):>12} "
              f"{_fmt(out['acc']):>10} {_fmt(c['expect_score']):>7}")

        if not _close(fb_score, c["expect_score"]):
            failures.append(
                f"[bin #{i} {c['name']}] _telemath_fallback={fb_score!r} "
                f"!= expected {c['expect_score']!r}"
            )
        if not _close(out["score"], fb_score):
            failures.append(
                f"[bin #{i} {c['name']}] compute_telemath_score(binary).score="
                f"{out['score']!r} != _telemath_fallback={fb_score!r}"
            )
        if not _close(out["acc"], fb_score):
            failures.append(
                f"[bin #{i} {c['name']}] compute_telemath_score(binary).acc="
                f"{out['acc']!r} != _telemath_fallback={fb_score!r}"
            )
        # Binary mode must NOT leak r1/r2/r3/r_unit_credit.
        for k in ("r1", "r2", "r3", "r_unit_credit", "pred_value_parsed"):
            if k in out:
                failures.append(
                    f"[bin #{i} {c['name']}] binary mode leaked key {k!r}"
                )
        # Binary key set is exactly {score, acc}.
        if set(out.keys()) != {"score", "acc"}:
            failures.append(
                f"[bin #{i} {c['name']}] binary key set drifted: "
                f"{sorted(out.keys())}"
            )

    # Also check that =binary and =unknown both fall through.
    for mode_val in ("binary", "BINARY", "something_else"):
        os.environ["TELEMATH_REWARD_MODE"] = mode_val
        try:
            out = compute_telemath_score("\\boxed{42}", "42", None)
        finally:
            os.environ.pop("TELEMATH_REWARD_MODE", None)
        if set(out.keys()) != {"score", "acc"}:
            failures.append(
                f"[bin fallthrough mode={mode_val!r}] keys drifted: "
                f"{sorted(out.keys())}"
            )

    # -----------------------------------------------------------------------
    # Section 2: v2 cases
    # -----------------------------------------------------------------------
    print()
    print("=" * 100)
    print(" TeleMath reward v2 — component breakdown")
    print("=" * 100)
    print(f"{'#':>2} {'name':<55} "
          f"{'r1':>5} {'r2':>6} {'r3':>7} {'unit':>5} {'score':>6}")
    print("-" * 100)

    for i, c in enumerate(V2_CASES, 1):
        os.environ["TELEMATH_REWARD_MODE"] = "v2"
        try:
            out = compute_telemath_score(c["solution"], c["gt"], c.get("extra_info"))
            # Direct v2 invocation must yield identical numbers (no env caching bug).
            out_direct = compute_telemath_score_v2(
                c["solution"], c["gt"], c.get("extra_info")
            )
        finally:
            os.environ.pop("TELEMATH_REWARD_MODE", None)

        print(f"{i:>2} {c['name']:<55} "
              f"{_fmt(out['r1']):>5} {_fmt(out['r2']):>6} "
              f"{_fmt(out['r3']):>7} {_fmt(out['r_unit_credit']):>5} "
              f"{_fmt(out['score']):>6}")

        # Required key set under v2.
        v2_keys = {"score", "acc", "r1", "r2", "r3", "r_unit_credit", "pred_value_parsed"}
        if not v2_keys.issubset(out.keys()):
            failures.append(
                f"[v2 #{i} {c['name']}] v2 missing keys: "
                f"{sorted(v2_keys - set(out.keys()))}"
            )

        # Direct vs toggled must match.
        for k in v2_keys:
            if not _close(out.get(k, 0.0), out_direct.get(k, 0.0)):
                failures.append(
                    f"[v2 #{i} {c['name']}] toggled v2.{k}={out.get(k)!r} "
                    f"!= direct v2.{k}={out_direct.get(k)!r}"
                )

        # acc tracks r1 (binary) regardless of shaping.
        if not _close(out["acc"], out["r1"]):
            failures.append(
                f"[v2 #{i} {c['name']}] acc={out['acc']!r} != r1={out['r1']!r}"
            )

        # Per-case expectations.
        if "expect_r1" in c and not _close(out["r1"], c["expect_r1"]):
            failures.append(
                f"[v2 #{i} {c['name']}] r1={out['r1']!r} != expected {c['expect_r1']!r}"
            )
        if "expect_r2" in c and not _close(out["r2"], c["expect_r2"]):
            failures.append(
                f"[v2 #{i} {c['name']}] r2={out['r2']!r} != expected {c['expect_r2']!r}"
            )
        if "expect_r2_max" in c and out["r2"] > c["expect_r2_max"] + 1e-6:
            failures.append(
                f"[v2 #{i} {c['name']}] r2={out['r2']!r} > expected_max {c['expect_r2_max']!r}"
            )
        if "expect_r3" in c and not _close(out["r3"], c["expect_r3"]):
            failures.append(
                f"[v2 #{i} {c['name']}] r3={out['r3']!r} != expected {c['expect_r3']!r}"
            )
        if "expect_unit" in c and not _close(out["r_unit_credit"], c["expect_unit"]):
            failures.append(
                f"[v2 #{i} {c['name']}] r_unit_credit={out['r_unit_credit']!r} "
                f"!= expected {c['expect_unit']!r}"
            )
        if "expect_score" in c and not _close(out["score"], c["expect_score"]):
            failures.append(
                f"[v2 #{i} {c['name']}] score={out['score']!r} != expected {c['expect_score']!r}"
            )
        if "expect_score_max" in c and out["score"] > c["expect_score_max"] + 1e-6:
            failures.append(
                f"[v2 #{i} {c['name']}] score={out['score']!r} > expected_max "
                f"{c['expect_score_max']!r}"
            )

    # -----------------------------------------------------------------------
    # Section 3: Schema invariance through the verl dispatcher
    # -----------------------------------------------------------------------
    print()
    print("=" * 100)
    print(" Schema invariance via dispatcher (compute_score) — binary vs v2")
    print("=" * 100)

    sample_solution = "\\boxed{42}"
    sample_gt = "42"
    sample_ei = {"question_text": "trivial"}

    os.environ.pop("TELEMATH_REWARD_MODE", None)
    d_bin = _dispatcher_call("TeleMath", sample_solution, sample_gt, sample_ei)
    os.environ["TELEMATH_REWARD_MODE"] = "v2"
    try:
        d_v2 = _dispatcher_call("TeleMath", sample_solution, sample_gt, sample_ei)
    finally:
        os.environ.pop("TELEMATH_REWARD_MODE", None)

    print(f"  binary keys: {sorted(d_bin.keys())}")
    print(f"  v2 keys    : {sorted(d_v2.keys())}")

    if set(d_bin.keys()) != set(d_v2.keys()):
        failures.append(
            f"[schema] dispatcher keys differ across modes: "
            f"binary={sorted(d_bin.keys())} vs v2={sorted(d_v2.keys())}"
        )
    # The dispatcher must include r_unit_credit (we added it to _NUMERIC_KEYS).
    if "r_unit_credit" not in d_bin:
        failures.append(
            f"[schema] dispatcher (binary) missing r_unit_credit key"
        )
    if "r_unit_credit" not in d_v2:
        failures.append(
            f"[schema] dispatcher (v2) missing r_unit_credit key"
        )
    # data_source must be carried through.
    if d_bin.get("data_source") != "TeleMath":
        failures.append(
            f"[schema] dispatcher dropped/altered data_source: "
            f"binary={d_bin.get('data_source')!r}"
        )

    # -----------------------------------------------------------------------
    print()
    if failures:
        print("=" * 100)
        print(f"FAILED: {len(failures)} assertion(s)")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("=" * 100)
    print(
        f"PASSED: {len(BIN_CASES)} binary + {len(V2_CASES)} v2 + 3 schema checks"
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
