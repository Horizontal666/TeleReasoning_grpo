#!/usr/bin/env python3
"""Smoke-test compute_mcq_score / compute_telemath_score v2 with mocked inputs.

Uses VARIED text (not repeated tokens) so we don't accidentally trip the
v2 runaway-repetition guard (tail uniq/total < 0.15 → r2 = 0).

Verifies:
  - Correct MCQ answers always get score=1.0 (ANSWER:X and \\boxed{X} both)
  - Wrong answers get 0.30 / 0.15 / 0.00 based on format
  - boxed and ANSWER trailing get symmetric format bonus (my v2 patch)
  - acc field stays binary

Out-of-scope (pre-existing, NOT my v2 patch):
  - Teletable freeform numeric-GT samples (e.g. GT="11.0") hit the bad-gt
    early-return in compute_mcq_score (binary and v2 both). Those 418/1238
    teletable training samples effectively get score=0 always → filter_groups
    drops them → never trained on. Val teletable parquet is 100% letter-GT so
    val metrics are unaffected.
"""
import os, sys, textwrap

os.environ["MCQ_REWARD_MODE"] = "v2"
os.environ["TELEMATH_REWARD_MODE"] = "v2"

sys.path.insert(0, "/dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl/examples/grpo_TeleInfer")
from telelogs_symbolic_pkg import telelogs_symbolic

# Varied CoT text — different sentences so tail uniq ratio stays > 0.15
LONG_COT = textwrap.dedent("""\
    Let me analyze this problem step by step. Looking at the question,
    I need to identify which option matches the specification given in the
    reference material. The first parameter mentioned is the delay spread,
    which in 5G NR scenarios typically ranges from -7.7 to -6.5 in log10
    seconds. The angular spread of departure depends on whether we are in
    LOS or NLOS conditions, and the cluster parameters scale accordingly.
    Considering UMi versus UMa scenarios, the per-cluster shadowing differs
    by approximately 3 dB. After examining each candidate carefully and
    cross-referencing with table B.1.2.2.1-4, the best matching value is""")

assert len(set(LONG_COT.split())) / len(LONG_COT.split()) > 0.5, "test fixture should be varied"

def case(name, solution, gt, extra=None, expect_score=None, expect_acc=None, data_source="teleqna"):
    r = telelogs_symbolic.compute_mcq_score(
        solution_str=solution, ground_truth=gt,
        extra_info=extra or {}, data_source=data_source,
    )
    score, acc = r.get("score"), r.get("acc")
    r1, r2, r3 = r.get("r1","—"), r.get("r2","—"), r.get("r3","—")
    ok_score = (expect_score is None) or abs(score - expect_score) < 1e-6
    ok_acc   = (expect_acc is None) or abs(acc - expect_acc) < 1e-6
    mark = "✓" if (ok_score and ok_acc) else "✗"
    print(f"{mark} {name:55s} score={score:.3f} acc={acc:.3f}  r1={r1} r2={r2} r3={r3}")
    return ok_score and ok_acc

print("=" * 80)
print("MCQ v2 — correct answers (must score=1.0; ANSWER and boxed parity)")
print("=" * 80)
all_ok = True
all_ok &= case("Correct ANSWER:X + long varied CoT", LONG_COT + " B.\nANSWER: B",
               gt="B", expect_score=1.0, expect_acc=1.0)
all_ok &= case("Correct \\boxed{X} + long varied CoT (NO ANSWER:)", LONG_COT + " B.\nFinal: \\boxed{B}",
               gt="B", expect_score=1.0, expect_acc=1.0)
all_ok &= case("Correct ANSWER:X bare-bones (no CoT) — r2=0.30 (short qualifies)",
               "ANSWER: B", gt="B", expect_score=1.0, expect_acc=1.0)
all_ok &= case("Correct \\boxed{B} bare-bones — r2=0.30 (boxed parity)",
               "\\boxed{B}", gt="B", expect_score=1.0, expect_acc=1.0)

print()
print("=" * 80)
print("MCQ v2 — wrong answers (variable score by format/hallucination)")
print("=" * 80)
all_ok &= case("Wrong + ANSWER:A + long varied CoT  →  r2=0.30", LONG_COT + "\nANSWER: A",
               gt="B", expect_score=0.30, expect_acc=0.0)
all_ok &= case("Wrong + \\boxed{A} + long varied CoT  →  r2=0.30 (boxed parity)",
               LONG_COT + "\nFinal: \\boxed{A}",
               gt="B", expect_score=0.30, expect_acc=0.0)
all_ok &= case("Wrong + no recognizable format  →  r2=0.15",
               "I do not really know which one is correct here honestly.",
               gt="B", expect_score=0.15, expect_acc=0.0)
all_ok &= case("Wrong + ANSWER:A bare-bones  →  r2=0.30",
               "ANSWER: A", gt="B", expect_score=0.30, expect_acc=0.0)
all_ok &= case("Wrong + 3 fake citations + ANSWER:  →  r2=0.30+r3=-0.30 → 0",
               LONG_COT + "\n\nAccording to 3GPP TS 38.211 section 5.2.3, and per O-RAN spec WG4 v3.0, "
               "and as documented in IETF RFC 9000 page 42.\nANSWER: A",
               gt="B", expect_score=0.00, expect_acc=0.0)
all_ok &= case("Wrong + 1 fake citation + ANSWER:  →  r2=0.30+r3=-0.10 → 0.20",
               LONG_COT + "\n\nPer IETF RFC 9000.\nANSWER: A",
               gt="B", expect_score=0.20, expect_acc=0.0)

print()
print("=" * 80)
print("Edge / guard cases")
print("=" * 80)
all_ok &= case("Empty response  →  r2=0", "", gt="B", expect_score=0.0, expect_acc=0.0)
all_ok &= case("Runaway repetition tail  →  r2=0",
               "Reasoning: " + "A " * 500 + "\nANSWER: A",
               gt="B", expect_score=0.0, expect_acc=0.0)
all_ok &= case("Correct + overlong (>4000 chars) ANSWER:X  →  r2=0.15",
               LONG_COT * 10 + "\nANSWER: B",
               gt="B", expect_score=1.0, expect_acc=1.0)

print()
print("=" * 80)
print("v2 vs binary parity — correct answer always 1.0 in BOTH modes")
print("=" * 80)
# Switch to binary, run same correct cases, confirm score=1.0
os.environ["MCQ_REWARD_MODE"] = "binary"
import importlib; importlib.reload(telelogs_symbolic)
all_ok &= case("BINARY: Correct ANSWER:X  →  1.0", LONG_COT + "\nANSWER: B",
               gt="B", expect_score=1.0, expect_acc=1.0)
all_ok &= case("BINARY: Correct \\boxed{X}  →  1.0", LONG_COT + "\n\\boxed{B}",
               gt="B", expect_score=1.0, expect_acc=1.0)
all_ok &= case("BINARY: Wrong ANSWER:X  →  0.0", LONG_COT + "\nANSWER: A",
               gt="B", expect_score=0.0, expect_acc=0.0)

# back to v2 for telemath
os.environ["MCQ_REWARD_MODE"] = "v2"
importlib.reload(telelogs_symbolic)

print()
print("=" * 80)
print("TeleMath v2 sanity (different scorer, correct must score=1.0)")
print("=" * 80)
from telelogs_symbolic_reward import compute_score as tele_compute_score
def case_tm(name, solution, gt, expect_score=None):
    r = tele_compute_score(data_source="TeleMath", solution_str=solution, ground_truth=gt, extra_info={})
    score, acc = r.get("score"), r.get("acc")
    ok = (expect_score is None) or abs(score - expect_score) < 1e-6
    mark = "✓" if ok else "✗"
    print(f"{mark} {name:55s} score={score:.3f} acc={acc:.3f}")
    return ok
all_ok &= case_tm("Correct TeleMath \\boxed{42}", "After computation: \\boxed{42}", "42", expect_score=1.0)
all_ok &= case_tm("Wrong TeleMath should not crash",
                  "After computation: \\boxed{99}", "42", expect_score=None)

print()
print("=" * 80)
print(f"OVERALL: {'PASS ✓' if all_ok else 'FAIL ✗'}")
print("=" * 80)
sys.exit(0 if all_ok else 1)
