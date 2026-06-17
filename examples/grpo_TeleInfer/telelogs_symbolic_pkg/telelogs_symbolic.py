"""
Deterministic reward for TeleLogs troubleshooting (S1-S8 simplified rulebook).

Reward = R1 + R2 + R3  (clamped to [0, 1])
  R1 = 1.0 if \\boxed{CX} matches ground-truth label
  R2 = 0.5 if the correct S-rule is identified as "Matched" in the response
  R3 = 0~0.3 for key feature values in the response matching reference_solution

Supports two response formats (auto-detected):
  English (new): feature table with feature_name | ... | value rows +
                 rule table with "S<n> | ... | Matched"
  Chinese (old): [计算] section with m01/m02/... labels +
                 [规则判断] section with "S<n>: TRUE"

No external LLM judge is called.  Ground-truth features come from
`extra_info.reference_solution` in the telelogs parquet.

Also dispatches 3GPP working-group classification samples by data_source and
gives 1.0 only when the final boxed working-group label exactly matches the
ground truth.

-----------------------------------------------------------------------------
MCQ reward mode (oranbench / srsran / teleqna / teletable / srsbench)
-----------------------------------------------------------------------------
The MCQ scorer ``compute_mcq_score`` defaults to legacy binary 0/1 letter-match
behaviour (byte-identical to the run that produced ckpt_1984). Setting

    export MCQ_REWARD_MODE=v2

activates ``compute_mcq_score_v2`` which adds two auxiliary reward components
without altering ``acc`` (validation metric stays comparable):

    r1 = letter-match correctness   (0 or 1)
    r2 = format-stability bonus     ([0, 0.3])      — clean ``ANSWER: X`` ending,
                                                       penalises runaway-repetition
                                                       and overlong output
    r3 = anti-hallucination penalty ([-0.3, 0])     — fires only on WRONG answers
                                                       that contain spec-shaped
                                                       citations (O-RAN.WGx, RFC,
                                                       3GPP TS, srsran/ paths,
                                                       Section x.y.z)

    score = clip(r1 + r2 + r3, 0.0, 1.0)
    acc   = r1                       # binary; preserved for cross-run comparison

The env var is read on every call (no module-import-time cache) so verl-side
re-launches or in-process reloads pick up changes immediately. See the smoke
test at ``examples/grpo_TeleInfer/test_mcq_reward_v2.py`` for concrete
side-by-side legacy/v2 outputs. Recommendation source:
``logs/qwen3_5_27b_telecominstruct_v2_6_qlora_stage3_checkpoint_1984-full/
failure_analysis.md`` §12.5 GRPO-A.

Setting ``MCQ_REWARD_MODE=v3_teleqna`` activates an *additional* layer of
penalties **only for samples with data_source == "teleqna"** (other MCQ
sources keep v2 behaviour). The v3 layer adds two components on top of v2
based on the teleqna-specific failure analysis at
``logs/qwen3_5_27b_telecominstruct_v2_6_qlora_stage3_checkpoint_1984-full/
teleqna_failure_analysis.md`` §8:

    r4 = fabrication-template penalty ([-0.30, 0])  — answer-key /
                                                       study-guide /
                                                       certification-context
                                                       sycophancy phrases
    r5 = hedging-density penalty      ([-0.20, 0])  — occurrences of
                                                       "wait"/"actually"/
                                                       "reconsider"/etc. beyond
                                                       the first; only on
                                                       WRONG answers

    score = clip(r1 + r2 + r3 + r4 + r5, 0.0, 1.0)
    acc   = r1   (still binary)

To revert: leave ``MCQ_REWARD_MODE`` unset (or set it to anything other than
``v2`` / ``v3_teleqna``).
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Optional

# Optional import of prime — used by the TeleMath v2 reward for math_equal
# tolerance grading and canonical boxed extraction. We tolerate import failure
# only at module-load time so that the other reward branches still load when
# prime is partially broken; the TeleMath v2 path requires it at runtime.
try:
    from . import prime as _prime  # noqa: E402
except Exception:  # pragma: no cover — fall back to package-relative resolution
    try:
        import prime as _prime  # type: ignore  # noqa: E402
    except Exception:
        _prime = None  # type: ignore

# ---------------------------------------------------------------------------
# 3GPP reward mode switch
# ---------------------------------------------------------------------------
# THREEGPP_REWARD_MODE controls compute_3gpp_score (see end of file).
#   v1  : legacy binary exact-match reward (r2=r3=0, score=r1). This is what
#         the ckpt_1984 GRPO run used and what the failure analysis at
#         logs/qwen3_5_27b_telecominstruct_v2_6_qlora_stage3_checkpoint_1984-full/
#         three_gpp_failure_analysis.md (Part B) measured. Set this to reproduce
#         that run exactly.
#   v2  : multi-component reward (default; recommended). Adds
#           r2 = confusion-aware partial credit for neighbour-WG misses
#           r3 = format-stability + TS-citation bonus
#         and an author-mention penalty (subtracted before clipping). See
#         Part B §16 of the failure analysis for the rationale.
#
#   v3  : process-aware reward (2026-05-27, designed from v2.7_rf failure
#         analysis at outputs/analysis/phase2_guide_validation_20260526/
#         v2.7_failure_mode_analysis.md). The v2.7_rf model produces
#         100% well-formed Step 0-4 CoT but loses 67.5% of wrong cases
#         to "Step 4 explicitly rejects the gold WG" (model talks itself
#         out of the right answer) and 31.6% to "GT never recalled in
#         Step 2". v1 binary reward gives no gradient into these steps;
#         v3 adds:
#           R_step2_recall  : +0.10 if GT label appears in Step 2 block
#                             (encourages listing the right candidate)
#           R_anti_reject_gt: -0.15 if a "Reject <GT>:" line is present
#                             AND model is wrong (penalizes talking out
#                             of the correct answer)
#           R_same_family   : +0.15 partial credit for wrong-but-same-family
#                             (SA/CT/RAN families); on top of v2's
#                             neighbour-credit which is +0.30 for known
#                             confusable pairs
#           R_gold_phrase   : -0.05 if "the gold label is" or similar
#                             phrase leaks (Deepseek-RF training artifact)
#         Score range: [-0.5, 1.0]. Right-answer score is flat 1.0 (no
#         additional bonuses) so within-group advantage is dominated by
#         the wrong-case shape.
#
# Switch back to v1 with:  export THREEGPP_REWARD_MODE=v1
_REWARD_MODE_DEFAULT = "v2"
_VALID_REWARD_MODES = {"v1", "v2", "v3"}


def _get_3gpp_reward_mode() -> str:
    """Read THREEGPP_REWARD_MODE per call so unit tests / shell can flip it
    without re-importing the module."""
    mode = os.environ.get("THREEGPP_REWARD_MODE", _REWARD_MODE_DEFAULT).strip().lower()
    return mode if mode in _VALID_REWARD_MODES else _REWARD_MODE_DEFAULT

# ---------------------------------------------------------------------------
# reference_solution / feature table parser (shared by ref and model output)
# ---------------------------------------------------------------------------

_FEATURE_LINE_RE = re.compile(
    r"^([a-z_A-Z0-9]+)\s*\|[^|]+\|\s*(.+?)\s*$",
    re.MULTILINE,
)
_NUMBER_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _parse_ref_features(ref_text: str) -> dict[str, Optional[float]]:
    """Parse feature_name → float (or None) from a pipe-table.

    Works on both reference_solution and the model's English feature table,
    since both use the same  feature_name | description | value  format.
    """
    feats: dict[str, Optional[float]] = {}
    if not ref_text:
        return feats
    for m in _FEATURE_LINE_RE.finditer(ref_text):
        name = m.group(1).strip()
        raw_val = m.group(2).strip()
        if "missing" in raw_val.lower():
            feats[name] = None
        else:
            num_m = _NUMBER_RE.search(raw_val)
            feats[name] = float(num_m.group(0)) if num_m else None
    return feats


# ---------------------------------------------------------------------------
# S1-S8 simplified rules → ground-truth rule id + class
# ---------------------------------------------------------------------------

def _apply_s8_rules(feats: dict[str, Optional[float]]) -> tuple[int, str]:
    """Return (rule_id 1-8, class C1-C8) for the first-hit simplified rule."""

    def v(key: str) -> Optional[float]:
        return feats.get(key)

    # S1 → C7
    if v("high_speed_ratio_gt40") is not None and v("high_speed_ratio_gt40") > 0:  # type: ignore[operator]
        return 1, "C7"

    # S2 → C8  (rb_low_ratio == 40.0 means exactly 4/10 rows)
    if v("rb_low_ratio_lt160") is not None and abs(v("rb_low_ratio_lt160") - 40.0) < 0.01:  # type: ignore[operator]
        return 2, "C8"

    # S3 → C2
    if v("distance_low_thr_mean_km") is not None and v("distance_low_thr_mean_km") > 1.0:  # type: ignore[operator]
        return 3, "C2"

    # S4 → C5
    if v("ho_count") is not None and abs(v("ho_count") - 3.0) < 0.01:  # type: ignore[operator]
        return 4, "C5"

    # S5 → C1
    m07 = v("handover_recovery_mean_mbps")
    m08 = v("mod30_collision_ratio_low_thr")
    m10 = v("serving_total_tilt_deg")
    if (m07 is not None and m07 <= 82.5
            and m08 is not None and m08 <= 50.0
            and m10 is not None and m10 > 12.0):
        return 5, "C1"

    # S6 → C4  (best_neighbor never became serving AND distance max is available)
    m06 = v("best_neighbor_becomes_serving_count")
    m11 = v("distance_low_thr_max_km")
    if m06 is not None and m06 <= 0 and m11 is not None:
        return 6, "C4"

    # S7 → C6
    m08 = v("mod30_collision_ratio_low_thr")
    m09 = v("mod30_collision_ratio")
    if (m08 is not None and m08 > 83.3
            and m09 is not None and m09 > 60.0):
        return 7, "C6"

    # S8 → C3 (fallback)
    return 8, "C3"


# ---------------------------------------------------------------------------
# Model output parsers
# ---------------------------------------------------------------------------

_BOXED_RE  = re.compile(r"\\boxed\s*\{\s*(?:C\s*)?([1-8])\s*\}", re.IGNORECASE)
_CHOICE_RE = re.compile(r"\bC([1-8])\b", re.IGNORECASE)
_3GPP_SOURCES = {"3gpp", "3gpp_grpo", "3gpp_working_group"}
_3GPP_GROUPS = {
    "CT1", "CT3", "CT4", "CT6",
    "RAN1", "RAN2", "RAN3", "RAN4", "RAN5", "RAN_AH1",
    "SA1", "SA2", "SA3", "SA4", "SA5", "SA6",
}
_3GPP_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}")
_3GPP_WORKING_GROUP_RE = re.compile(
    r'"?WORKING\s+GROUP"?\s*[:=]\s*["\'`{]?\s*([A-Z][A-Z0-9_\s-]*?)\s*["\'`}]*\s*(?=[,}\]\n\r]|$)',
    re.IGNORECASE,
)

# Generic 4-/5-option MCQ datasets that share the SFT 'ANSWER: $LETTER' format.
_MCQ_SOURCES = {"teleqna", "oranbench", "srsran", "srsbench", "teletable", "teletables"}

# Symbolic / numeric math data sources routed to compute_telemath_score.
# Same set as telelogs_symbolic_reward._PRIME_MATH_SOURCES so the explicit
# dispatch (compute_score / compute_score_batched) and the legacy fallback
# (_telemath_fallback in telelogs_symbolic_reward.py) agree on which sources
# are math-graded. Match against lower-cased data_source strings.
_TELEMATH_SOURCES = {
    "telemath",
    "aime24",
    "aime25",
    "math500",
    "real/deepmath-codeverifier",
}
_MCQ_LETTER_RE = re.compile(r"ANSWER\s*:\s*([A-E])\b", re.IGNORECASE)
_MCQ_BOXED_LETTER_RE = re.compile(r"\\boxed\s*\{\s*([A-Ea-e])\s*\}")
_MCQ_BARE_LETTER_RE = re.compile(r"^\s*([A-Ea-e])\s*$")

# [Rules] section (English new) or [规则判断] (Chinese legacy)
_RULE_SECTION_RE = re.compile(
    r"(?:\[Rules\]|\[规则判断\])(.*?)(?=\[Answer\]|\[答案\]|\Z)",
    re.DOTALL | re.IGNORECASE,
)
# S<n> (cond): TRUE  — anchored to line start so "S1-S7" inside S8's condition text is ignored
_RULE_TRUE_RE = re.compile(r"^\s*S([1-8])\b[^\n]*\bTRUE\b", re.IGNORECASE | re.MULTILINE)

# [Calculation] section (English new) or [计算] (Chinese legacy)
_CALC_SECTION_RE = re.compile(
    r"(?:\[Calculation\]|\[计算\])(.*?)(?=\[Rules\]|\[规则判断\]|\[Answer\]|\[答案\]|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# m01-m11 metric patterns — labels are the same in English and Chinese formats.
# m07 uses [-\d.]+ to handle negative handover-recovery values.
_METRIC_PATTERNS: dict[str, tuple[str, re.Pattern]] = {
    "high_speed_ratio_gt40":               ("m01", re.compile(r"m01[^\n]*=\s*([\d.]+)\s*%",            re.IGNORECASE)),
    "rb_low_ratio_lt160":                  ("m02", re.compile(r"m02[^\n]*=\s*([\d.]+)\s*%",            re.IGNORECASE)),
    "distance_low_thr_mean_km":            ("m04", re.compile(r"m04[^\n]*=\s*([\d.]+)\s*km",           re.IGNORECASE)),
    "ho_count":                            ("m05", re.compile(r"m05[^\n]*=\s*([0-9]+)(?:\s|$)",         re.IGNORECASE)),
    "best_neighbor_becomes_serving_count": ("m06", re.compile(r"m06[^\n]*=\s*([0-9]+)(?:\s|$)",         re.IGNORECASE)),
    "handover_recovery_mean_mbps":         ("m07", re.compile(r"m07[^\n]*=\s*([-\d.]+)\s*(?:Mbps|$)",  re.IGNORECASE)),
    "mod30_collision_ratio_low_thr":       ("m08", re.compile(r"m08[^\n]*=\s*([\d.]+)\s*%",            re.IGNORECASE)),
    "mod30_collision_ratio":               ("m09", re.compile(r"m09[^\n]*=\s*([\d.]+)\s*%",            re.IGNORECASE)),
    "serving_total_tilt_deg":              ("m10", re.compile(r"m10[^\n]*=\s*([\d.]+)\s*(?:度|deg|°|$)", re.IGNORECASE)),
    "distance_low_thr_max_km":             ("m11", re.compile(r"m11[^\n]*=\s*([\d.]+)\s*km",           re.IGNORECASE)),
}

# Key metrics graded for R3 (all metrics evaluated up to and including the firing rule)
_RULE_KEY_METRICS: dict[int, list[str]] = {
    1: ["high_speed_ratio_gt40"],
    2: ["high_speed_ratio_gt40", "rb_low_ratio_lt160"],
    3: ["high_speed_ratio_gt40", "rb_low_ratio_lt160", "distance_low_thr_mean_km"],
    4: ["high_speed_ratio_gt40", "rb_low_ratio_lt160", "distance_low_thr_mean_km", "ho_count"],
    5: ["high_speed_ratio_gt40", "rb_low_ratio_lt160", "distance_low_thr_mean_km", "ho_count",
        "handover_recovery_mean_mbps", "mod30_collision_ratio_low_thr", "serving_total_tilt_deg"],
    6: ["high_speed_ratio_gt40", "rb_low_ratio_lt160", "distance_low_thr_mean_km", "ho_count",
        "best_neighbor_becomes_serving_count", "distance_low_thr_max_km"],
    7: ["high_speed_ratio_gt40", "rb_low_ratio_lt160", "distance_low_thr_mean_km", "ho_count",
        "mod30_collision_ratio_low_thr", "mod30_collision_ratio"],
    8: ["high_speed_ratio_gt40", "rb_low_ratio_lt160", "distance_low_thr_mean_km", "ho_count"],
}


def _parse_answer(text: str) -> Optional[str]:
    m = _BOXED_RE.search(text)
    if m:
        return f"C{m.group(1)}"
    matches = list(_CHOICE_RE.finditer(text))
    return f"C{matches[-1].group(1)}" if matches else None


def _parse_rule_hit(text: str) -> Optional[int]:
    """Return S-rule id (1-8) whose condition is marked TRUE, or None."""
    sec_m = _RULE_SECTION_RE.search(text)
    section = sec_m.group(1) if sec_m else text
    m = _RULE_TRUE_RE.search(section)
    return int(m.group(1)) if m else None


def _parse_calc_metrics(text: str) -> dict[str, float]:
    """Extract m01-m11 numeric values from the [Calculation] section."""
    sec_m = _CALC_SECTION_RE.search(text)
    section = sec_m.group(1) if sec_m else text
    result: dict[str, float] = {}
    for feat_name, (_mid, pat) in _METRIC_PATTERNS.items():
        m = pat.search(section)
        if m:
            try:
                result[feat_name] = float(m.group(1))
            except ValueError:
                pass
    return result


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

_REL_TOL = 0.05   # 5 % relative tolerance for metric matching
_ABS_TOL = 0.5    # absolute tolerance (for small values near 0)
_R3_PER_METRIC = 0.05
_R3_MAX = 0.30


def _metric_close(model_val: float, ref_val: Optional[float]) -> bool:
    if ref_val is None:
        return False
    if ref_val == 0.0:
        return abs(model_val) <= _ABS_TOL
    return abs(model_val - ref_val) / (abs(ref_val) + 1e-9) <= _REL_TOL


def _extract_ref_text(extra_info: Any) -> str:
    if extra_info is None:
        return ""
    if isinstance(extra_info, dict):
        return str(extra_info.get("reference_solution") or "")
    # flat parquet row – field may arrive as "extra_info.reference_solution"
    try:
        return str(extra_info.get("reference_solution") or "")
    except AttributeError:
        return ""


def _normalize_choice(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    m = re.search(r"C([1-8])", text)
    if m:
        return f"C{m.group(1)}"
    if text.isdigit() and 1 <= int(text) <= 8:
        return f"C{text}"
    return None


def _normalize_3gpp_group(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw)
    text = re.sub(r"\\(?:text|mathrm)\s*\{\s*([^{}]+?)\s*\}", r"\1", text)
    text = text.strip(" \t\r\n\"'`.,;:()[]{}")
    canonical = re.sub(r"[^A-Z0-9]", "", text.upper())
    for group in sorted(_3GPP_GROUPS, key=len, reverse=True):
        if canonical == re.sub(r"[^A-Z0-9]", "", group):
            return group
    return None


def _parse_3gpp_answer(text: str) -> Optional[str]:
    matches = _3GPP_WORKING_GROUP_RE.findall(text or "")
    for candidate in reversed(matches):
        group = _normalize_3gpp_group(candidate)
        if group is not None:
            return group
    matches = _3GPP_BOXED_RE.findall(text or "")
    for candidate in reversed(matches):
        group = _normalize_3gpp_group(candidate)
        if group is not None:
            return group
    return None


def _parse_mcq_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = _MCQ_LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _MCQ_BOXED_LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _MCQ_BARE_LETTER_RE.match(text)
    return m.group(1).upper() if m else None


def _extract_last_boxed_content(text: str) -> Optional[str]:
    marker = "\\boxed{"
    start = (text or "").rfind(marker)
    if start < 0:
        return None
    i = start + len(marker)
    depth = 1
    chars: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)
        i += 1
    return None


def _normalize_mcq_value(raw: Any) -> str:
    text = str(raw or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\r\n\"'`.,;:")
    return text.lower()


# ---------------------------------------------------------------------------
# MCQ reward v2 — auxiliary components (GRPO-A in failure_analysis.md §12.5)
# ---------------------------------------------------------------------------
# Activated by `export MCQ_REWARD_MODE=v2`. Default ("binary" or unset) routes
# through the legacy compute_mcq_score body and is byte-identical to the
# behaviour that produced ckpt_1984.
#
# Implementation notes:
#   - The env var is re-read on every call inside compute_mcq_score (no
#     import-time caching), so a `MCQ_REWARD_MODE=v2 python ...` invocation
#     of an already-imported module behaves correctly.
#   - `acc` STAYS BINARY (= r1) so validation metrics are comparable across
#     runs. Only `score` changes shape under v2.
#   - r1/r2/r3 are populated explicitly in the v2 return dict so the verl 0.8
#     reward-key ordering contract (see telelogs_symbolic_reward.py
#     _NUMERIC_KEYS) is satisfied without relying on the dispatcher's
#     default-zero fill.

# Citation-shaped fragment patterns. Each pattern targets a hallucination
# signature catalogued in §3.4 / §4.3 / §5.5 of failure_analysis.md.
#
# Examples that should match each pattern (one per line):
#   O-RAN.WG4.MP.0-v09.00
#   O-RAN.WG9.XTRP-v02.0
#   srsran/asn1/rrc_nr/sib_msg.h
#   srsran/phy/upper/channel_coding/ldpc.cpp
#   Section 6.2.2.3
#   §6.2.3
#   RFC 3261
#   RFC8200
#   3GPP TS 38.331
#   3GPP TS 23.501.2
_MCQ_V2_CITATION_PATTERNS: tuple[re.Pattern, ...] = (
    # O-RAN spec IDs: "O-RAN.WG<n>" with optional sub-tokens and "-v<digits>.<digits>"
    re.compile(r"O-RAN\.WG\d+[A-Za-z0-9\-.]*-v?\d+\.\d+"),
    # srsRAN source paths: srsran/<components>.h|hpp|cpp|c
    re.compile(r"srsran/[A-Za-z0-9_/\-.]+\.(?:h|hpp|cpp|c)\b"),
    # Section numbers: "Section 6.2.2.3" (2+ dots) or "§6.2.3"
    re.compile(r"Section\s+\d+(?:\.\d+){1,}", re.IGNORECASE),
    re.compile(r"§\s*\d+(?:\.\d+){1,}"),
    # RFC citations
    re.compile(r"RFC\s*\d{3,5}\b", re.IGNORECASE),
    # 3GPP TS citations
    re.compile(r"3GPP\s+TS\s+\d+\.\d+(?:\.\d+)?", re.IGNORECASE),
)

# Format-stability tuning.
_MCQ_V2_FORMAT_FULL_BONUS = 0.30   # clean ANSWER: X ending, short, no runaway
_MCQ_V2_FORMAT_PARTIAL    = 0.15   # answer parses but missing some structural cue
_MCQ_V2_FORMAT_PENALTY    = 0.00   # runaway / empty: zero (no negative on r2 itself)
_MCQ_V2_FORMAT_MAX_LEN    = 4000   # chars; over this counts as overlong → partial

# Anti-hallucination tuning.
_MCQ_V2_HALLUCINATION_HEAVY = -0.30    # 3+ distinct fake-citation matches + wrong
_MCQ_V2_HALLUCINATION_LIGHT = -0.10    # 1-2 distinct matches + wrong

# Tail-window for repetition/runaway detection.
_MCQ_V2_TAIL_WINDOW = 1000
_MCQ_V2_TAIL_UNIQ_RATIO_MIN = 0.15

# Trailing ANSWER: X token detector — must appear in the LAST 200 chars of
# the response stripped of trailing whitespace. The presence of *any* clean
# ANSWER: X is necessary; "near the end" is a stronger signal of format
# stability than buried mid-response.
_MCQ_V2_TRAILING_ANSWER_RE = re.compile(r"ANSWER\s*:\s*[A-E]\b", re.IGNORECASE)

# Trailing \boxed{...} detector. teletable in particular mixes MCQ samples
# (which use ANSWER: X) with free-form table-value queries whose SFT-implied
# convention is `\boxed{value}` — without this fallback, v2 would
# systematically dock 0.15 off those 418/1238 teletable samples even when the
# value-match in r1 fires, training the model to abandon \boxed{} in favour
# of a (wrong) "ANSWER: A" coda. Matches `\boxed{...}` with any non-empty
# body, balanced-brace-naive (single-level — sufficient for MCQ values and
# numeric/string answers; nested boxes are rare and not worth special-casing
# here).
_MCQ_V2_TRAILING_BOXED_RE = re.compile(r"\\boxed\s*\{[^}]+\}", re.IGNORECASE)


def _mcq_v2_tail_uniq_ratio(text: str) -> float:
    """Fraction of *unique* whitespace-delimited tokens in the last
    ``_MCQ_V2_TAIL_WINDOW`` characters of ``text``. Used as a cheap
    runaway-repetition signature. Returns 1.0 for very short tails so we
    don't punish concise answers."""
    if not text:
        return 1.0
    tail = text[-_MCQ_V2_TAIL_WINDOW:]
    toks = tail.split()
    if len(toks) < 20:
        # Too short for a meaningful ratio; treat as healthy.
        return 1.0
    return len(set(toks)) / len(toks)


def _mcq_v2_count_distinct_citations(text: str) -> int:
    """Number of *distinct* citation-shaped fragments in ``text`` across the
    full pattern bank. Distinct = unique normalised string (whitespace
    collapsed, case lowered)."""
    if not text:
        return 0
    seen: set[str] = set()
    for pat in _MCQ_V2_CITATION_PATTERNS:
        for m in pat.finditer(text):
            frag = re.sub(r"\s+", " ", m.group(0).strip()).lower()
            seen.add(frag)
    return len(seen)


def _mcq_v2_format_bonus(solution_str: str) -> float:
    """Return the r2 component value for ``solution_str``.

    Decision tree:
      - empty or runaway-repetition tail (uniq/total < threshold) → 0.0
      - clean trailing ``ANSWER: X`` AND length < _MCQ_V2_FORMAT_MAX_LEN → 0.30
      - otherwise                                                     → 0.15
    """
    text = solution_str or ""
    if not text.strip():
        return _MCQ_V2_FORMAT_PENALTY  # 0.0 for empty
    if _mcq_v2_tail_uniq_ratio(text) < _MCQ_V2_TAIL_UNIQ_RATIO_MIN:
        return _MCQ_V2_FORMAT_PENALTY  # 0.0 for runaway
    # Trailing ANSWER: X or \boxed{...} check — look in the last 200
    # non-whitespace chars. Either format earns the full bonus so the
    # MCQ-letter convention and the teletable-freeform \boxed{value}
    # convention are treated symmetrically.
    stripped = text.rstrip()
    tail = stripped[-200:] if len(stripped) > 200 else stripped
    has_trailing_answer = bool(_MCQ_V2_TRAILING_ANSWER_RE.search(tail))
    has_trailing_boxed  = bool(_MCQ_V2_TRAILING_BOXED_RE.search(tail))
    if (has_trailing_answer or has_trailing_boxed) and len(text) < _MCQ_V2_FORMAT_MAX_LEN:
        return _MCQ_V2_FORMAT_FULL_BONUS
    return _MCQ_V2_FORMAT_PARTIAL


def _mcq_v2_hallucination_penalty(solution_str: str, is_correct: bool) -> float:
    """Return the r3 component value (≤ 0). No penalty on correct answers —
    we don't want to discourage *correct* citations."""
    if is_correct:
        return 0.0
    n_distinct = _mcq_v2_count_distinct_citations(solution_str or "")
    if n_distinct >= 3:
        return _MCQ_V2_HALLUCINATION_HEAVY
    if n_distinct >= 1:
        return _MCQ_V2_HALLUCINATION_LIGHT
    return 0.0


def compute_mcq_score_v2(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """MCQ reward with auxiliary components (GRPO-A).

    Returns the same dict shape as ``compute_mcq_score`` plus explicit
    ``r1``/``r2``/``r3`` keys. ``acc`` remains binary letter-match
    correctness so validation metrics are comparable to runs using the
    legacy scorer.

    Composition:
        r1 = letter_match in {0.0, 1.0}        — dominant signal
        r2 = format-stability bonus in [0, 0.3]
        r3 = anti-hallucination penalty in [-0.3, 0]
        score = clip(r1 + r2 + r3, 0.0, 1.0)
    """
    gt = str(ground_truth).strip().upper()
    if gt not in {"A", "B", "C", "D", "E"}:
        # Match the legacy scorer's bad-gt early return shape, but include
        # r1/r2/r3 = 0.0 so the verl key contract still holds.
        return {
            "score":              0.0,
            "acc":                0.0,
            "r1":                 0.0,
            "r2":                 0.0,
            "r3":                 0.0,
            "pred_letter_parsed": 0.0,
            "pred_value_parsed":  0.0,
            "error":              f"bad gt: {gt!r}",
        }

    pred = _parse_mcq_letter(solution_str or "")
    letter_match = pred == gt

    boxed_value = _extract_last_boxed_content(solution_str or "")
    correct_option_value = ""
    if isinstance(extra_info, dict):
        correct_option_value = str(extra_info.get("correct_option_value") or "")
    value_match = bool(
        correct_option_value
        and boxed_value is not None
        and _normalize_mcq_value(boxed_value) == _normalize_mcq_value(correct_option_value)
    )

    is_correct = bool(letter_match or value_match)
    r1 = 1.0 if is_correct else 0.0
    r2 = _mcq_v2_format_bonus(solution_str or "")
    r3 = _mcq_v2_hallucination_penalty(solution_str or "", is_correct=is_correct)

    # r2 (format bonus) gated on r1 (correctness). Investigation 2026-05-19:
    # on 4 MCQ domains (teletable/teleqna/oranbench/srsran) the ungated +0.30
    # format-bonus floor created a "guess + ANSWER: X coda" exploit basin that
    # collapsed response length and template-collapsed reasoning. Gating
    # eliminates the basin by construction; r3 hallucination penalty still
    # applies to wrong rollouts on domains where citations naturally appear.
    r2_eff = r2 if r1 == 1.0 else 0.0
    score = max(0.0, min(1.0, r1 + r2_eff + r3))

    return {
        "score":              float(score),
        "acc":                float(r1),  # acc MUST stay binary correctness
        "r1":                 float(r1),
        "r2":                 float(r2),
        "r3":                 float(r3),
        "pred_letter_parsed": float(pred is not None),
        "pred_value_parsed":  float(boxed_value is not None),
    }


# ---------------------------------------------------------------------------
# MCQ reward v3 (teleqna-only) — fabrication & hedging penalties
# ---------------------------------------------------------------------------
# Activated by `export MCQ_REWARD_MODE=v3_teleqna` AND data_source == "teleqna".
# For non-teleqna sources under v3_teleqna mode, behaviour is identical to v2.
# For teleqna under v3_teleqna mode, two additional penalty terms layer on top
# of v2:
#
#   r4 — fabrication-template penalty (HARD).
#     Fires on sycophancy-backdoor phrases catalogued in
#     teleqna_failure_analysis.md §8 (answer key / study guide / certification
#     context / common in similar quizzes / "I found a snippet" / etc.).
#     -0.30 if any phrase appears AND answer is WRONG.
#     -0.15 if any phrase appears AND answer is RIGHT (still discourage the
#     template even when it lands right).
#
#   r5 — hedging-density penalty (MILD).
#     Occurrences of "wait"/"actually"/"hmm"/"reconsider"/"re-evaluate"/
#     "let me reconsider"/"but actually"/"but wait"/"on second thought"
#     beyond the first occurrence are penalised at -0.05 each, capped at -0.20.
#     Only fires when the answer is WRONG (we don't want to discourage useful
#     self-check on correct paths).
#
#   score = clip(r1 + r2 + r3 + r4 + r5, 0.0, 1.0)
#   acc remains binary letter-match (r1).
#
# Re-read of MCQ_REWARD_MODE per call (no module-level cache), same pattern as
# the v2 dispatch. Constants live at module-level so they can be tweaked
# without code surgery.

_MCQ_V3_FABRICATION_PENALTY_WRONG = -0.30
_MCQ_V3_FABRICATION_PENALTY_CORRECT = -0.15

_MCQ_V3_HEDGING_PER_OCCURRENCE = -0.05
_MCQ_V3_HEDGING_MAX_PENALTY = -0.20

# Fabrication-template phrases (case-insensitive, word-boundary anchored).
# Source: teleqna_failure_analysis.md §8 (recommended phrases to penalise).
_MCQ_V3_FABRICATION_PHRASES: tuple[str, ...] = (
    "answer key",
    "the key says",
    "official answer",
    "study guide",
    "study bank",
    "flashcard",
    "quiz bank",
    "exam dump",
    "certification context",
    "common in similar quizzes",
    "common in certification exams",
    "I found a reference",
    "I found a snippet",
    "search query mental simulation",
    "often cited in similar",
    "Coursera quiz",
)

# Hedging phrases. Word-boundary anchored, case-insensitive. Sourced from
# teleqna_failure_analysis.md §8 — hedging density is ~8× higher in regressed
# samples than in both-correct samples (median 0 vs 3).
_MCQ_V3_HEDGING_PHRASES: tuple[str, ...] = (
    "wait",
    "actually",
    "hmm",
    "reconsider",
    "re-evaluate",
    "let me reconsider",
    "but actually",
    "but wait",
    "on second thought",
)


def _mcq_v3_compile_phrases(phrases: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    """Compile a tuple of literal phrases into word-boundary, case-insensitive
    regex patterns. ``re.escape`` handles dots/hyphens in inputs like
    ``re-evaluate``; we then bracket with ``\\b`` on each side."""
    compiled: list[re.Pattern] = []
    for phrase in phrases:
        compiled.append(
            re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        )
    return tuple(compiled)


_MCQ_V3_FABRICATION_RES: tuple[re.Pattern, ...] = _mcq_v3_compile_phrases(
    _MCQ_V3_FABRICATION_PHRASES
)
_MCQ_V3_HEDGING_RES: tuple[re.Pattern, ...] = _mcq_v3_compile_phrases(
    _MCQ_V3_HEDGING_PHRASES
)


def _mcq_v3_fabrication_penalty(solution_str: str, is_correct: bool) -> float:
    """r4: -0.30 if any fabrication phrase appears AND answer is wrong;
    -0.15 if any fabrication phrase appears AND answer is right; 0.0 otherwise.
    """
    text = solution_str or ""
    if not text:
        return 0.0
    for pat in _MCQ_V3_FABRICATION_RES:
        if pat.search(text):
            return (
                _MCQ_V3_FABRICATION_PENALTY_CORRECT
                if is_correct
                else _MCQ_V3_FABRICATION_PENALTY_WRONG
            )
    return 0.0


def _mcq_v3_hedging_penalty(solution_str: str, is_correct: bool) -> float:
    """r5: -0.05 per hedging occurrence beyond the first, capped at -0.20.
    Only fires on WRONG answers."""
    if is_correct:
        return 0.0
    text = solution_str or ""
    if not text:
        return 0.0
    total = 0
    for pat in _MCQ_V3_HEDGING_RES:
        total += len(pat.findall(text))
    extras = max(0, total - 1)
    if extras == 0:
        return 0.0
    penalty = extras * _MCQ_V3_HEDGING_PER_OCCURRENCE
    if penalty < _MCQ_V3_HEDGING_MAX_PENALTY:
        penalty = _MCQ_V3_HEDGING_MAX_PENALTY
    return penalty


def compute_mcq_score_v3_teleqna(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """MCQ reward with the teleqna-specific v3 penalties on top of v2.

    Returns the same dict shape as ``compute_mcq_score_v2`` plus explicit
    ``r4``/``r5`` keys. ``acc`` remains binary letter-match correctness.

    Composition:
        r1 = letter_match in {0.0, 1.0}              — dominant signal
        r2 = format-stability bonus in [0, 0.3]
        r3 = anti-hallucination penalty in [-0.3, 0]
        r4 = fabrication-template penalty in [-0.3, 0]
        r5 = hedging-density penalty in [-0.2, 0]
        score = clip(r1 + r2 + r3 + r4 + r5, 0.0, 1.0)
    """
    # Run v2 first to inherit r1/r2/r3 and acc. We then layer r4/r5 on top.
    v2 = compute_mcq_score_v2(solution_str, ground_truth, extra_info)
    # If v2 short-circuited on a bad gt, propagate the error shape verbatim
    # but still attach r4/r5 = 0 so the verl key contract is satisfied.
    if "error" in v2:
        v2_out = dict(v2)
        v2_out["r4"] = 0.0
        v2_out["r5"] = 0.0
        return v2_out

    is_correct = bool(v2.get("r1", 0.0) >= 1.0)
    r4 = _mcq_v3_fabrication_penalty(solution_str or "", is_correct=is_correct)
    r5 = _mcq_v3_hedging_penalty(solution_str or "", is_correct=is_correct)

    r1 = float(v2["r1"])
    r2 = float(v2["r2"])
    r3 = float(v2["r3"])
    score = max(0.0, min(1.0, r1 + r2 + r3 + r4 + r5))

    out = dict(v2)
    out["score"] = float(score)
    out["r4"] = float(r4)
    out["r5"] = float(r5)
    return out


def compute_mcq_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
    data_source: Optional[str] = None,
) -> dict[str, Any]:
    """Exact-match scoring for MCQ datasets.

    Supports the legacy letter formats (``ANSWER: A``, ``\\boxed{A}``, bare
    ``A``) and TeleTable value-format prompts where the model boxes the full
    correct option text.

    If ``MCQ_REWARD_MODE`` is set to ``"v2"`` (case-insensitive, read on every
    call), dispatches to ``compute_mcq_score_v2`` which adds auxiliary r2/r3
    components. If ``MCQ_REWARD_MODE`` is ``"v3_teleqna"`` AND
    ``data_source == "teleqna"`` (case-insensitive after strip), dispatches to
    ``compute_mcq_score_v3_teleqna`` which adds r4/r5 on top of v2. Other MCQ
    sources under v3_teleqna mode fall through to v2.

    Default behaviour (env unset or any other value) is byte-identical to the
    ckpt_1984 GRPO run — DO NOT MODIFY THE LEGACY PATH.
    """
    # Re-evaluate the env var on every call. verl 0.8 reloads modules across
    # workers, but reading at call-time also lets shell-driven A/B tests
    # toggle the mode mid-experiment without touching code.
    mode = os.environ.get("MCQ_REWARD_MODE", "binary").strip().lower()
    if mode == "v3_teleqna":
        ds = (data_source or "").strip().lower() if data_source is not None else ""
        if ds == "teleqna":
            return compute_mcq_score_v3_teleqna(solution_str, ground_truth, extra_info)
        # Non-teleqna sources under v3_teleqna mode → behave like v2.
        return compute_mcq_score_v2(solution_str, ground_truth, extra_info)
    if mode == "v2":
        return compute_mcq_score_v2(solution_str, ground_truth, extra_info)

    # ----- legacy binary path — UNCHANGED -----------------------------------
    gt = str(ground_truth).strip().upper()
    if gt not in {"A", "B", "C", "D", "E"}:
        return {"score": 0.0, "acc": 0.0, "pred_letter_parsed": 0.0, "error": f"bad gt: {gt!r}"}
    pred = _parse_mcq_letter(solution_str or "")
    letter_match = pred == gt

    boxed_value = _extract_last_boxed_content(solution_str or "")
    correct_option_value = ""
    if isinstance(extra_info, dict):
        correct_option_value = str(extra_info.get("correct_option_value") or "")
    value_match = bool(
        correct_option_value
        and boxed_value is not None
        and _normalize_mcq_value(boxed_value) == _normalize_mcq_value(correct_option_value)
    )

    score = 1.0 if letter_match or value_match else 0.0
    return {
        "score":              float(score),
        "acc":                float(score),
        "pred_letter_parsed": float(pred is not None),
        "pred_value_parsed":  float(boxed_value is not None),
    }


# ---------------------------------------------------------------------------
# TeleMath reward v2 — multi-component shaping reward
# ---------------------------------------------------------------------------
# Activated by `export TELEMATH_REWARD_MODE=v2`. Default ("binary" / unset)
# routes through compute_telemath_score's binary branch and is byte-identical
# to the legacy _telemath_fallback (prime.math_equal wrapped to {score, acc}).
# This is the user's roll-back path for the run that produced ckpt_1984.
#
# Design rationale: telemath_failure_analysis.md Parts I+II showed the binary
# reward causes (a) false zeros on numerically-correct but format-failed
# rollouts, (b) no gradient on the chronic regression patterns (Friis NF
# double-count, M/M/1-on-deterministic, V_OC voltage convention, ALOHA
# variant confusion), and (c) collapsed intra-group variance by step ~120.
#
# v2 reward composition:
#   r1            in {0.0, 1.0}    — math_equal(boxed, gt) (dominant)
#   r2            in [0, 0.20]     — format-stability bonus
#   r3            in [-0.30, 0.0]  — anti-pattern penalty (only when r1==0)
#   r_unit_credit in {0.0, 0.5}    — unit-canonicalised soft credit
#                                    (only when r1==0)
#   score = clip(r1 + r2 + r3 + r_unit_credit, 0.0, 1.0)
#   acc   = r1   (binary; preserved for cross-run comparison)

# ----- regex banks (pre-compiled) -------------------------------------------
# Sentinel that an arbitrary substring of solution_str looks like a Friis-style
# noise figure / cascade analysis. We match either of the canonical equivalent
# expressions for the *equivalent noise temperature*: T_sys=T_a+T_0(F-1) or
# T_e=T_0(F-1) (with optional sub/super-scripts, LaTeX braces, leading T_).
_TELEMATH_NF_TE_RE = re.compile(
    r"T[_\s\\\{]*(?:sys|e|eq|equiv|equivalent)[_\s\\\}]*"
    r"\s*=\s*"
    r"(?:T[_\s\\\{]*(?:a|amb|antenna|ant)[_\s\\\}]*\s*\+\s*)?"
    r"T[_\s\\\{]*0[_\s\\\}]*\s*"
    r"(?:\\cdot|\*|\\times)?\s*"
    r"[\(\\\{][\s]*F\s*[-−]\s*1[\s]*[\)\\\}]",
    re.IGNORECASE,
)
# Sentinel for "SNR_out = SNR_in - NF" (or rearranged "SNR_in = SNR_out + NF").
# This second subtraction on top of an already-T_e-incorporated noise floor is
# the double-count fingerprint we want to penalise.
_TELEMATH_SNR_NF_RE = re.compile(
    r"SNR[_\s\\\{]*(?:in|out|i|o)?[_\s\\\}]*"
    r"\s*=\s*"
    r"SNR[_\s\\\{]*(?:in|out|i|o)?[_\s\\\}]*"
    r"\s*[-−+]\s*"
    r"(?:NF|F[_\s\\\{]*dB)",
    re.IGNORECASE,
)
# Prompt-side cue: this trigger is only relevant when the question actually
# mentions noise figure / temperature.
_TELEMATH_NF_PROMPT_RE = re.compile(
    r"\b(?:noise\s+figure|noise\s+temperature|NF\b|T_?e\b|equivalent\s+noise)",
    re.IGNORECASE,
)

# Deterministic-service prompt cues (M/D/1, not M/M/1).
_TELEMATH_DETERMINISTIC_KEYWORDS = (
    "constant bit rate",
    "deterministic service",
    "deterministic arrival",
    "fixed packet length",
    "fixed packet size",
    "constant service time",
    "constant packet length",
    "constant packet size",
    "m/d/1",
)
# M/M/1 formula fingerprints. Each of these forms uses Markovian queueing
# arithmetic which is wrong when the service distribution is deterministic.
_TELEMATH_MM1_FORMULA_RES = (
    # 1/(mu - lambda)
    re.compile(r"\b1\s*/\s*[\(\\\{]\s*(?:\\mu|mu|μ)\s*[-−]\s*(?:\\lambda|lambda|λ)\s*[\)\\\}]"),
    # rho/(1-rho) * 1/mu
    re.compile(
        r"(?:\\rho|rho|ρ)\s*/\s*[\(\\\{]\s*1\s*[-−]\s*(?:\\rho|rho|ρ)\s*[\)\\\}]"
        r"\s*(?:\\cdot|\*|\\times)?\s*1\s*/\s*(?:\\mu|mu|μ)"
    ),
    # lambda/(mu*(mu-lambda))
    re.compile(
        r"(?:\\lambda|lambda|λ)\s*/\s*"
        r"[\(\\\{]\s*(?:\\mu|mu|μ)\s*(?:\\cdot|\*|\\times)?\s*"
        r"[\(\\\{]\s*(?:\\mu|mu|μ)\s*[-−]\s*(?:\\lambda|lambda|λ)\s*[\)\\\}]\s*[\)\\\}]"
    ),
    # explicit "M/M/1" assumption naming
    re.compile(r"\bM\s*/\s*M\s*/\s*1\b", re.IGNORECASE),
)
# "Get out of jail" markers — if any of these are present, we believe the
# model is aware of the deterministic service and is using the correct
# M/D/1 / Pollaczek-Khinchine formulation.
_TELEMATH_MD1_AWARENESS_RES = (
    re.compile(r"\bM\s*/\s*D\s*/\s*1\b", re.IGNORECASE),
    re.compile(r"pollaczek[\s-]?khinchine", re.IGNORECASE),
    re.compile(r"\bP[\s-]?K\s+formula\b", re.IGNORECASE),
    re.compile(r"deterministic\s+service\s+time", re.IGNORECASE),
    re.compile(r"deterministic\s+arrival", re.IGNORECASE),
    re.compile(r"variance\s+of\s+service\s+time\s*=\s*0", re.IGNORECASE),
)

# Voltage-convention prompt cues (available-power / matched-load / V_OC).
_TELEMATH_VOLTAGE_PROMPT_RE = re.compile(
    r"\b(?:available\s+power|matched\s+load|open[\s-]?circuit\s+voltage|"
    r"thevenin|thévenin|V_?OC\b)",
    re.IGNORECASE,
)
# Fingerprint of computing voltage as sqrt(P*R) — the load-convention form.
# Looks for a single P*R inside a sqrt with no leading 4* coefficient.
_TELEMATH_VOLTAGE_SQRT_PR_RE = re.compile(
    r"\\?sqrt\s*[\(\\\{]\s*(?!\s*4\s*[\\\*\\cdot\\times])"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\s*[\\\*\\cdot]\s*[A-Za-z_][A-Za-z0-9_]*|"
    r"P\s*[\\\*\\cdot]?\s*R|R\s*[\\\*\\cdot]?\s*P)"
    r"\s*[\)\\\}]",
    re.IGNORECASE,
)
# Fingerprint of computing voltage as sqrt(4*P*R) — the Thevenin form.
_TELEMATH_VOLTAGE_SQRT_4PR_RE = re.compile(
    r"\\?sqrt\s*[\(\\\{]\s*4\s*[\\\*\\cdot\\times]\s*"
    r"(?:P\s*[\\\*\\cdot\\times]?\s*R|R\s*[\\\*\\cdot\\times]?\s*P)"
    r"\s*[\)\\\}]",
    re.IGNORECASE,
)

# ALOHA variant cues.
_TELEMATH_ALOHA_PURE_PROMPT_RE = re.compile(r"\bpure\s+ALOHA\b", re.IGNORECASE)
_TELEMATH_ALOHA_SLOTTED_PROMPT_RE = re.compile(r"\bslotted\s+ALOHA\b", re.IGNORECASE)
# G * e^(-2G)  → slotted formula
_TELEMATH_ALOHA_SLOTTED_FORMULA_RE = re.compile(
    r"G\s*(?:\\cdot|\*|\\times)?\s*e\s*\^?\s*[\(\\\{]?\s*[-−]\s*2\s*(?:\\cdot|\*|\\times)?\s*G",
    re.IGNORECASE,
)
# G * e^(-G)  → pure formula. Use a negative lookahead to avoid matching the
# slotted "-2G" form.
_TELEMATH_ALOHA_PURE_FORMULA_RE = re.compile(
    r"G\s*(?:\\cdot|\*|\\times)?\s*e\s*\^?\s*[\(\\\{]?\s*[-−]\s*(?!2)G",
    re.IGNORECASE,
)

# Single \boxed{NUMBER} detector for r2. Numbers may carry sign, decimal point,
# scientific notation, fractions, or LaTeX. We require *digits inside* to
# avoid awarding the bonus for empty placeholder \boxed{} brackets.
_TELEMATH_BOXED_NUMERIC_RE = re.compile(
    r"\\boxed\s*\{\s*[^{}]*\d[^{}]*\s*\}",
)

# Cheap numeric extractor used by the r_unit_credit pass. Pull the first
# signed decimal / scientific-notation token out of a \boxed{...} payload.
_TELEMATH_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _get_telemath_reward_mode() -> str:
    """Read TELEMATH_REWARD_MODE per call so unit tests / shell can flip it
    without re-importing the module. Defaults to 'binary' (legacy)."""
    mode = os.environ.get("TELEMATH_REWARD_MODE", "binary").strip().lower()
    return mode if mode in {"binary", "v2"} else "binary"


def _telemath_envf(name: str, default: float) -> float:
    """Env-var float lookup, re-read per call. Mirrors _envf below."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _telemath_extract_prompt_text(extra_info: Any) -> str:
    """Best-effort prompt-text recovery for r3 heuristics.

    The reward function does NOT receive the rendered prompt directly. We
    therefore probe the well-known keys on extra_info (set by the TeleMath
    data prep). Returns "" if none are present — r3 then conservatively
    skips prompt-dependent triggers (preferred false-negative behaviour).
    """
    if extra_info is None:
        return ""
    try:
        getter = extra_info.get
    except AttributeError:
        return ""
    for key in (
        "question_text",
        "question_with_unit",
        "raw_question",
        "prompt",
        "question",
    ):
        try:
            val = getter(key)
        except Exception:
            val = None
        if val:
            return str(val)
    return ""


def _telemath_r2_boxed_bonus(solution_str: str, tail_chars: int = 500) -> float:
    """+bonus when exactly one parseable boxed-with-digits appears in the last
    ``tail_chars`` characters of the solution. Reward signal for a *trailing*
    boxed answer (the SFT format), not a buried mid-response placeholder."""
    text = solution_str or ""
    if not text:
        return 0.0
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    n = len(_TELEMATH_BOXED_NUMERIC_RE.findall(tail))
    return 1.0 if n == 1 else 0.0


def _telemath_r2_length_bonus(solution_str: str, lo: int, hi: int) -> float:
    """+bonus when total visible-text length is in [lo, hi]. Penalises both
    runaway outputs and silent stop-outs."""
    n = len(solution_str or "")
    return 1.0 if lo <= n <= hi else 0.0


def _telemath_r2_norep_bonus(
    solution_str: str,
    ngram_len: int,
    max_hits: int,
) -> float:
    """+bonus when no ``ngram_len``-char substring appears >= ``max_hits``
    times in the *last 25%* of the text. Cheap O(n) tail scan; we treat the
    whole text as the tail when it is short."""
    text = solution_str or ""
    if len(text) < ngram_len * max_hits:
        return 1.0  # too short to plausibly repeat — credit it
    tail_start = max(0, int(len(text) * 0.75))
    tail = text[tail_start:]
    if len(tail) < ngram_len:
        return 1.0
    counts: dict[str, int] = {}
    # Step by 1 char so we catch overlapping repetitions like "abcabcabc".
    for i in range(0, len(tail) - ngram_len + 1):
        sub = tail[i:i + ngram_len]
        c = counts.get(sub, 0) + 1
        if c >= max_hits:
            return 0.0
        counts[sub] = c
    return 1.0


def _telemath_r3_nf_double(prompt_text: str, solution_str: str) -> bool:
    """True iff the prompt is a noise-figure / temperature problem AND the
    response simultaneously uses the T_e=T_0(F-1) substitution AND the
    SNR-out = SNR-in - NF subtraction (= double-counts the noise figure)."""
    if not prompt_text or not solution_str:
        return False
    if not _TELEMATH_NF_PROMPT_RE.search(prompt_text):
        return False
    has_te = bool(_TELEMATH_NF_TE_RE.search(solution_str))
    has_snr_nf = bool(_TELEMATH_SNR_NF_RE.search(solution_str))
    return has_te and has_snr_nf


def _telemath_r3_mm1_on_deterministic(prompt_text: str, solution_str: str) -> bool:
    """True iff the prompt explicitly says service is deterministic / fixed
    AND the response uses an M/M/1 formula WITHOUT acknowledging M/D/1 or
    the Pollaczek-Khinchine variance term."""
    if not prompt_text or not solution_str:
        return False
    p_lower = prompt_text.lower()
    if not any(kw in p_lower for kw in _TELEMATH_DETERMINISTIC_KEYWORDS):
        return False
    has_mm1 = any(pat.search(solution_str) for pat in _TELEMATH_MM1_FORMULA_RES)
    if not has_mm1:
        return False
    # If the response shows any awareness that service is deterministic, do
    # NOT penalise — false positives are worse than false negatives here.
    aware = any(pat.search(solution_str) for pat in _TELEMATH_MD1_AWARENESS_RES)
    return not aware


def _telemath_r3_voltage(prompt_text: str, solution_str: str) -> bool:
    """True iff prompt mentions available-power / matched-load / V_OC AND the
    response uses one voltage convention while the other was canonical.

    This is conservative: we fire only when *exactly one* of the two forms
    appears. If both sqrt(P*R) and sqrt(4*P*R) appear (model showed both),
    we skip the penalty — too ambiguous."""
    if not prompt_text or not solution_str:
        return False
    if not _TELEMATH_VOLTAGE_PROMPT_RE.search(prompt_text):
        return False
    has_pr = bool(_TELEMATH_VOLTAGE_SQRT_PR_RE.search(solution_str))
    has_4pr = bool(_TELEMATH_VOLTAGE_SQRT_4PR_RE.search(solution_str))
    if has_pr == has_4pr:
        return False
    # We penalise either direction — the user can tighten with env vars if
    # they want directional credit. The point is the response is using the
    # wrong factor-of-two convention for the available-power reading.
    return True


def _telemath_r3_aloha(prompt_text: str, solution_str: str) -> bool:
    """True iff prompt says 'slotted ALOHA' and response uses the pure
    formula G*e^(-G), or vice versa."""
    if not prompt_text or not solution_str:
        return False
    is_slotted = bool(_TELEMATH_ALOHA_SLOTTED_PROMPT_RE.search(prompt_text))
    is_pure = bool(_TELEMATH_ALOHA_PURE_PROMPT_RE.search(prompt_text))
    has_slotted_f = bool(_TELEMATH_ALOHA_SLOTTED_FORMULA_RE.search(solution_str))
    has_pure_f = bool(_TELEMATH_ALOHA_PURE_FORMULA_RE.search(solution_str))
    if is_slotted and has_pure_f and not has_slotted_f:
        return True
    if is_pure and has_slotted_f and not has_pure_f:
        return True
    return False


def _telemath_try_math_equal(pred: str, gt: str) -> bool:
    """Wrap prime.math_equal with the same defaults as _telemath_fallback so
    the v2 r1 / r_unit_credit branches grade identically to the binary
    fallback."""
    if _prime is None:
        return False
    try:
        return bool(
            _prime.math_equal(
                pred,
                gt,
                timeout=True,
                timeout_seconds=2.0,
                rel_tol=1e-2,
                abs_tol=1e-4,
            )
        )
    except Exception:
        return False


def _telemath_unit_credit(boxed: Optional[str], gt: Any, prompt_text: str) -> bool:
    """True iff a small bank of unit conversions on ``boxed`` matches ``gt``.

    Covers:
      - dBm  <-> mW
      - dB   <-> linear  (10*log10 and 20*log10 conventions)
      - factor of 2 or 4  (only when prompt mentions matched-load / available)
      - percentage <-> fractional
      - 1 - boxed  (inside / outside region swap)
    """
    if boxed is None:
        return False
    m = _TELEMATH_NUM_RE.search(boxed)
    if not m:
        return False
    try:
        x = float(m.group(0))
    except (TypeError, ValueError):
        return False
    gt_s = str(gt)

    candidates: list[float] = []
    # dB / linear conversions — guard log/exp domains.
    if x > 0:
        candidates.append(10.0 * math.log10(x))
        candidates.append(20.0 * math.log10(x))
    try:
        candidates.append(10.0 ** (x / 10.0))
        candidates.append(10.0 ** (x / 20.0))
    except OverflowError:
        pass
    # Factor of 2 / 4 voltage adjustments — only when prompt hints at it,
    # to keep this from accidentally rewarding wholly-wrong magnitudes.
    if _TELEMATH_VOLTAGE_PROMPT_RE.search(prompt_text or ""):
        candidates.append(x * 2.0)
        candidates.append(x * 4.0)
        candidates.append(x / 2.0)
        candidates.append(x / 4.0)
    # Percent / fractional swap
    candidates.append(x * 100.0)
    candidates.append(x / 100.0)
    # Inside / outside region complement
    candidates.append(1.0 - x)

    for c in candidates:
        if _telemath_try_math_equal(repr(c), gt_s):
            return True
    return False


def compute_telemath_score_v2(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """TeleMath reward with multi-component shaping (failure_analysis Part II §17).

    Returns a dict containing:
        score             : clip(r1 + r2 + r3 + r_unit_credit, 0.0, 1.0)
        acc               : float(r1)            — binary, for cross-run parity
        r1                : math_equal(boxed, gt) ∈ {0.0, 1.0}
        r2                : format-stability bonus ∈ [0, 0.20]
        r3                : anti-pattern penalty   ∈ [-0.30, 0.0]
        r_unit_credit     : unit-canonicalised credit ∈ {0.0, TELEMATH_UNIT_CREDIT}
        pred_value_parsed : 1.0 if a parseable boxed was found

    All component weights and thresholds are env-overridable; see the
    docstring at the top of run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh for
    the canonical override list.
    """
    # Re-read env vars per call so a shell-driven A/B test picks them up
    # without re-import (matches MCQ v2 and 3GPP v2 contracts).
    w_box = _telemath_envf("TELEMATH_R2_BOXED_BONUS", 0.10)
    w_len = _telemath_envf("TELEMATH_R2_LENGTH_BONUS", 0.05)
    w_rep = _telemath_envf("TELEMATH_R2_NOREP_BONUS", 0.05)
    p_nf = _telemath_envf("TELEMATH_R3_NF_DOUBLE_PENALTY", -0.10)
    p_mm = _telemath_envf("TELEMATH_R3_MM1_DET_PENALTY", -0.10)
    p_vt = _telemath_envf("TELEMATH_R3_VOLTAGE_PENALTY", -0.10)
    p_al = _telemath_envf("TELEMATH_R3_ALOHA_PENALTY", -0.10)
    p_cap = _telemath_envf("TELEMATH_R3_PENALTY_CAP", -0.30)
    w_unit = _telemath_envf("TELEMATH_UNIT_CREDIT", 0.5)
    len_min = int(_telemath_envf("TELEMATH_R2_LEN_MIN", 500))
    len_max = int(_telemath_envf("TELEMATH_R2_LEN_MAX", 10000))
    ngram_n = int(_telemath_envf("TELEMATH_R2_REPETITION_NGRAM", 30))
    max_hits = int(_telemath_envf("TELEMATH_R2_REPETITION_MAX_HITS", 4))

    text = solution_str or ""
    prompt_text = _telemath_extract_prompt_text(extra_info)

    # ---- r1: canonical math_equal -----------------------------------------
    boxed = _prime.extract_boxed_content(text) if _prime is not None else None
    r1 = 1.0 if _telemath_try_math_equal(boxed or "", str(ground_truth)) else 0.0

    # ---- r2: format-stability bonus ---------------------------------------
    r2 = (
        w_box * _telemath_r2_boxed_bonus(text)
        + w_len * _telemath_r2_length_bonus(text, len_min, len_max)
        + w_rep * _telemath_r2_norep_bonus(text, ngram_n, max_hits)
    )

    # ---- r3: anti-pattern penalty (only when r1 == 0) ---------------------
    r3 = 0.0
    if r1 == 0.0:
        if _telemath_r3_nf_double(prompt_text, text):
            r3 += p_nf
        if _telemath_r3_mm1_on_deterministic(prompt_text, text):
            r3 += p_mm
        if _telemath_r3_voltage(prompt_text, text):
            r3 += p_vt
        if _telemath_r3_aloha(prompt_text, text):
            r3 += p_al
        # Cap penalty magnitude.
        if r3 < p_cap:
            r3 = p_cap

    # ---- r_unit_credit: unit-canonicalised soft credit (only when r1==0) --
    r_unit = 0.0
    if r1 == 0.0:
        if _telemath_unit_credit(boxed, ground_truth, prompt_text):
            r_unit = w_unit

    score = max(0.0, min(1.0, r1 + r2 + r3 + r_unit))

    return {
        "score":              float(score),
        "acc":                float(r1),          # binary, comparable across runs
        "r1":                 float(r1),
        "r2":                 float(r2),
        "r3":                 float(r3),
        "r_unit_credit":      float(r_unit),
        "pred_value_parsed":  float(boxed is not None and boxed != ""),
    }


def compute_telemath_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """TeleMath reward entry point.

    Reads ``TELEMATH_REWARD_MODE`` on every call:
      - ``binary`` (default / unset / any value != "v2") → byte-identical to
        _telemath_fallback in telelogs_symbolic_reward.py: returns
        ``{"score": 0/1, "acc": 0/1}`` from prime.math_equal.
      - ``v2`` → multi-component shaping reward; see compute_telemath_score_v2.

    DO NOT MODIFY THE BINARY PATH — it is the user's rollback contract.
    """
    if _get_telemath_reward_mode() == "v2":
        return compute_telemath_score_v2(solution_str, ground_truth, extra_info)

    # ----- legacy binary path — byte-identical to _telemath_fallback -------
    boxed = _prime.extract_boxed_content(solution_str or "") if _prime is not None else None
    ok = _telemath_try_math_equal(boxed or "", str(ground_truth))
    score = 1.0 if ok else 0.0
    return {"score": float(score), "acc": float(score)}


# ---------------------------------------------------------------------------
# 3GPP reward components (v2 mode)
# ---------------------------------------------------------------------------
# These constants encode the recommendations from
#   logs/qwen3_5_27b_telecominstruct_v2_6_qlora_stage3_checkpoint_1984-full/
#   three_gpp_failure_analysis.md  Part B §16.
#
# Tunable via env vars so the operator can ablate without editing code.

def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# r2 — partial credit for landing on a known-confusable neighbour WG. This
# converts the ~30% of GRPO prompts where all 5 rollouts are wrong into a
# non-zero-advantage gradient signal (see Part B §12).
_R2_NEIGHBOUR_CREDIT = _envf("THREEGPP_R2_NEIGHBOUR_CREDIT", 0.30)

# r3 — format-stability bonus. The SFT model emitted 26-char direct JSON;
# the GRPO checkpoint regrew the CoT to 3,560 chars (Part B §13.2 Mechanism 1).
# A modest positive bonus for short clean output and a penalty for runaway
# CoT pulls the policy back toward the SFT format.
_R3_FORMAT_BONUS    = _envf("THREEGPP_R3_FORMAT_BONUS",   0.10)  # ≤200 chars + valid JSON
_R3_CITATION_BONUS  = _envf("THREEGPP_R3_CITATION_BONUS", 0.10)  # TS-series cite matches pred
_R3_LONG_PENALTY    = _envf("THREEGPP_R3_LONG_PENALTY",   0.05)  # >8000 chars total

# Author/sender heuristic penalty. 100% of CT1 late-wrong rollouts invoke
# "the document is authored by X → label X" reasoning (Part B §13.2 Mechanism 3).
# When the prediction equals the most-mentioned WG in the model's own
# rationale AND that prediction is wrong, apply a small penalty.
_AUTHOR_PENALTY     = _envf("THREEGPP_AUTHOR_PENALTY",    0.30)

# Score envelope. Final score = clip(r1 + r2 + r3 - author_penalty, lo, hi).
# Allowing negative scores would change the GRPO advantage scale; v1/v2 clip
# to [0, 1] to stay schema-compatible with the existing reward-loop manager.
_SCORE_LO = 0.0
_SCORE_HI = 1.0

# v3-specific envelope. v3 deliberately allows negative scores to push
# rejects-GT samples BELOW clean-miss samples (i.e. talking yourself out of
# the right answer should hurt more than missing it entirely). DAPO/GRPO
# advantage normalization is within-group, so absolute negativity is fine.
_V3_SCORE_LO = -0.5
_V3_SCORE_HI = 1.0

# v3 constants (R2 / R3 / R4 / R5 per failure_mode_analysis recommendations)
_V3_STEP2_RECALL_BONUS    = _envf("THREEGPP_V3_STEP2_RECALL_BONUS",    0.10)
_V3_ANTI_REJECT_GT_PENALTY = _envf("THREEGPP_V3_ANTI_REJECT_GT_PENALTY", 0.15)
_V3_SAME_FAMILY_CREDIT    = _envf("THREEGPP_V3_SAME_FAMILY_CREDIT",    0.15)
_V3_GOLDPHRASE_PENALTY    = _envf("THREEGPP_V3_GOLDPHRASE_PENALTY",    0.05)

# WG-family membership (for R_same_family partial credit).
_3GPP_FAMILIES = {
    "SA": frozenset({"SA1", "SA2", "SA3", "SA4", "SA5", "SA6"}),
    "CT": frozenset({"CT1", "CT3", "CT4", "CT6"}),
    # RAN_AH1 is administratively part of RAN family
    "RAN": frozenset({"RAN1", "RAN2", "RAN3", "RAN4", "RAN5", "RAN_AH1"}),
}


def _same_family(a: Optional[str], b: Optional[str]) -> bool:
    """True if both WGs belong to the same family (SA / CT / RAN)."""
    if not a or not b or a == b:
        return False
    for fam in _3GPP_FAMILIES.values():
        if a in fam and b in fam:
            return True
    return False


# Step 2 block extractor. The RF-CoT format is:
#   Step 2 — Candidate WG recall by cues (from §5):
#     - <LABEL>: <reason>
#     - ...
#   Step 3 — Decide by official scope...
# Capture EVERYTHING between "Step 2" and "Step 3" markers (including the
# Step 2 header line itself, in case the candidate labels appear inline).
# Robust against both multi-line and inline formats.
_STEP2_BLOCK_RE = re.compile(
    r"Step\s*2\b(.+?)(?=Step\s*3\b)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_step2_block(solution_str: str) -> str:
    """Return the text between 'Step 2' and 'Step 3' headers, or empty string."""
    if not solution_str:
        return ""
    m = _STEP2_BLOCK_RE.search(solution_str)
    return m.group(1) if m else ""


# Rejected-labels extractor. The RF-CoT format uses "Reject <LABEL>: because ..."
# in Step 4. We collect every label that follows "Reject " (case-insensitive),
# matching the 16-WG set strictly so we don't catch other words.
_REJECT_LINE_RE = re.compile(
    r"\bReject\s+(SA[1-6]|CT[1346]|RAN(?:_AH1|[1-5]))\s*[:：]",
    re.IGNORECASE,
)


def _extract_rejected_labels(solution_str: str) -> set[str]:
    """Return the set of WG labels that appear in 'Reject <LABEL>:' lines."""
    if not solution_str:
        return set()
    return {m.group(1).upper().replace("_AH1", "_AH1") for m in _REJECT_LINE_RE.finditer(solution_str)}


# Gold-label phrase leak detector. Deepseek-RF training data sometimes leaked
# the prompt's "GOLD LABEL: X" line into the CoT as "the gold label is X".
# Catch common variants. v3 applies a small penalty when this phrase shows
# up — it's a contamination signal even when the actual prediction is right.
_GOLDPHRASE_RE = re.compile(
    r"\b(?:the\s+gold\s+label(?:\s+is)?|gold[\s-]*truth|the\s+answer\s+is\s+given|ground[\s-]*truth\s+label)\b",
    re.IGNORECASE,
)


def _has_gold_label_phrase(solution_str: str) -> bool:
    return bool(_GOLDPHRASE_RE.search(solution_str or ""))

# Confusable-WG map. Keys: ground-truth WG; values: set of neighbouring WGs.
# Bidirectional. Derived directly from `3GPP_WG_REFERENCE.md` "Common
# Confusion Patterns" + Part A §3.1 confusion-matrix top entries.
_3GPP_NEIGHBOURS: dict[str, frozenset[str]] = {
    # Core Network architecture ↔ NAS protocol ↔ intra-CN protocol ↔ external/policy protocol
    "SA2":     frozenset({"CT1", "CT3", "CT4", "SA1"}),
    "CT1":     frozenset({"SA2", "CT4", "CT3", "SA1"}),
    "CT3":     frozenset({"CT4", "SA2", "CT1", "SA4"}),
    "CT4":     frozenset({"SA2", "CT1", "CT3", "SA3"}),
    "CT6":     frozenset({"CT1", "SA2", "SA3", "SA1"}),
    # Service / system aspects
    "SA1":     frozenset({"SA2", "SA6", "SA3"}),
    "SA3":     frozenset({"SA2", "SA1", "CT4", "SA6"}),
    "SA4":     frozenset({"CT3", "SA2", "SA6"}),
    "SA5":     frozenset({"SA4", "SA1", "CT4"}),
    "SA6":     frozenset({"SA1", "SA2", "SA5"}),
    # Radio Access Network: PHY ↔ L2/L3 ↔ interfaces ↔ RF ↔ test
    "RAN1":    frozenset({"RAN2", "RAN4"}),
    "RAN2":    frozenset({"RAN1", "RAN3"}),
    "RAN3":    frozenset({"RAN2", "SA2"}),
    "RAN4":    frozenset({"RAN1", "RAN5"}),
    "RAN5":    frozenset({"RAN4"}),
    # Ad-hoc class. RAN_AH1 looks like RAN1/RAN4/SA1/RAN2 by surface text
    # (Part A §4.1). We mark these as the canonical neighbours.
    "RAN_AH1": frozenset({"RAN1", "RAN4", "SA1", "RAN2"}),
}


def _is_confusable_neighbour(gt: str, pred: str) -> bool:
    """True if (gt, pred) are listed as known-confusable in the reference map.
    Symmetric: also checks the reverse direction in case the map is asymmetric.
    """
    if gt == pred:
        return False
    return pred in _3GPP_NEIGHBOURS.get(gt, frozenset()) or \
           gt in _3GPP_NEIGHBOURS.get(pred, frozenset())


# TS-series → WG anchor map. Detecting one of these in the rationale, with
# matching predicted WG, restores the base model's "winning move"
# (Part B §13.2: SA5 RIGHT rollouts cite TS in 75% of cases vs 50% for SA5
# WRONG). Each rule is "TS <prefix>.* → WG".
#
# Pattern matches both "TS 23.501" and "TS 38.211" forms (decimal or three-
# digit subgroup), and tolerates "TR" as well.
_TS_CITATION_RE = re.compile(
    r"\bT[SR]\s*(\d{2})\.(\d{3})\b",
    re.IGNORECASE,
)

# Series-prefix (first 2 digits) + first sub-digit → owning WG. Most rules are
# series-wide; a few branches need 3-digit precision and are encoded as
# nested keys. See Part A §6 P1 rec 5 and Part B §16 P1 rec 5.
_TS_SERIES_TO_WG: dict[str, str] = {
    "22": "SA1",
    "23": "SA2",
    "24": "CT1",
    "25": "RAN2",   # legacy UTRA (UMTS); mostly RAN2-owned, some RAN3
    "26": "SA4",
    "28": "SA5",
    "31": "CT6",
    "32": "SA5",
    "33": "SA3",
    "34": "RAN5",
    "35": "SA3",
    "36": "RAN1",   # LTE PHY (36.211-213). Refined below for 36.331/4xx/5xx.
    "37": "RAN5",   # multi-RAT test specs (37.544/571)
    "38": "RAN1",   # NR PHY (38.211-214). Refined below for 38.331/4xx/5xx.
    "43": "RAN3",
    "44": "CT1",
    "45": "RAN2",
    "48": "CT1",
    "49": "CT4",
    "52": "RAN3",
}

# Refinement rules for TS 36.* and TS 38.* (LTE / NR), keyed on full XX.YYY.
_TS_SPECIFIC_TO_WG: dict[str, str] = {
    # NR
    "38.211": "RAN1", "38.212": "RAN1", "38.213": "RAN1", "38.214": "RAN1",
    "38.300": "RAN2", "38.321": "RAN2", "38.322": "RAN2", "38.323": "RAN2",
    "38.331": "RAN2",
    "38.401": "RAN3", "38.410": "RAN3", "38.411": "RAN3", "38.412": "RAN3",
    "38.413": "RAN3", "38.420": "RAN3", "38.421": "RAN3", "38.422": "RAN3",
    "38.423": "RAN3", "38.470": "RAN3", "38.471": "RAN3", "38.472": "RAN3",
    "38.473": "RAN3",
    "38.521": "RAN5", "38.522": "RAN5", "38.523": "RAN5", "38.531": "RAN5",
    "38.533": "RAN5",
    "38.101": "RAN4", "38.102": "RAN4", "38.104": "RAN4", "38.113": "RAN4",
    "38.124": "RAN4", "38.133": "RAN4", "38.141": "RAN4",
    # LTE
    "36.211": "RAN1", "36.212": "RAN1", "36.213": "RAN1", "36.214": "RAN1",
    "36.300": "RAN2", "36.321": "RAN2", "36.322": "RAN2", "36.323": "RAN2",
    "36.331": "RAN2",
    "36.411": "RAN3", "36.412": "RAN3", "36.413": "RAN3", "36.420": "RAN3",
    "36.421": "RAN3", "36.422": "RAN3", "36.423": "RAN3", "36.424": "RAN3",
    "36.521": "RAN5", "36.523": "RAN5", "36.533": "RAN5", "36.571": "RAN5",
    "36.101": "RAN4", "36.104": "RAN4", "36.133": "RAN4", "36.141": "RAN4",
    # CT3 vs CT4 split inside 29.*: 29.2xx mostly CT3, 29.5xx CT4
    # (handled below in _ts_owner_for as a special case)
}


def _ts_owner_for(major: str, minor: str) -> Optional[str]:
    """Map a parsed TS number ('29', '512') → owning WG or None."""
    full = f"{major}.{minor}"
    if full in _TS_SPECIFIC_TO_WG:
        return _TS_SPECIFIC_TO_WG[full]
    # 29.xxx split: 29.2xx -> CT3 (policy/exposure-facing AVPs, Gx/Rx/Sd);
    # 29.5xx -> CT4 (intra-CN SBI services, S6a, N4/N7/N16). See Part A §4.2.
    if major == "29":
        if minor.startswith("2") or minor.startswith("3"):
            return "CT3"
        if minor.startswith("5") or minor.startswith("1") or minor.startswith("6"):
            return "CT4"
        return None
    return _TS_SERIES_TO_WG.get(major)


def _extract_ts_owners(text: str) -> list[str]:
    """Return list of WG names implied by TS citations in `text`."""
    if not text:
        return []
    owners: list[str] = []
    for m in _TS_CITATION_RE.finditer(text):
        owner = _ts_owner_for(m.group(1), m.group(2))
        if owner is not None:
            owners.append(owner)
    return owners


# Format helpers --------------------------------------------------------------

_DIRECT_JSON_RE = re.compile(
    r'^\s*\{\s*"WORKING\s+GROUP"\s*:\s*"[A-Z][A-Z0-9_]*"\s*\}\s*$',
    re.IGNORECASE,
)
_POST_THINK_RE = re.compile(r"</think>\s*(.*)\Z", re.DOTALL | re.IGNORECASE)


def _post_think(solution_str: str) -> str:
    """Return the part of `solution_str` after the last </think>, or the
    whole string if no </think> is present."""
    m = _POST_THINK_RE.search(solution_str or "")
    return m.group(1) if m else (solution_str or "")


# Author-mention helpers ------------------------------------------------------

_WG_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9_])(" + "|".join(
        sorted(
            ("RAN_AH1", "RAN1", "RAN2", "RAN3", "RAN4", "RAN5",
             "SA1", "SA2", "SA3", "SA4", "SA5", "SA6",
             "CT1", "CT3", "CT4", "CT6"),
            key=len, reverse=True,
        )
    ) + r")(?![A-Z0-9_])",
)


def _most_mentioned_wg(text: str) -> Optional[str]:
    """Return the WG name with the highest token-count in `text`, or None
    if no WG token appears."""
    if not text:
        return None
    counts: dict[str, int] = {}
    for m in _WG_TOKEN_RE.finditer(text):
        wg = m.group(1).upper()
        counts[wg] = counts.get(wg, 0) + 1
    if not counts:
        return None
    # ties broken by first-occurrence order (stable max)
    return max(counts.items(), key=lambda kv: (kv[1], -list(counts.keys()).index(kv[0])))[0]


# ---------------------------------------------------------------------------
# 3GPP scorer entry-point
# ---------------------------------------------------------------------------

def compute_3gpp_score(solution_str: str, ground_truth: Any) -> dict[str, Any]:
    """Compute the 3GPP working-group reward for one rollout.

    Mode is selected by env var `THREEGPP_REWARD_MODE`:
      v1 -> legacy binary exact-match (r2=r3=0, score=r1)
      v2 -> multi-component: r1 exact + r2 neighbour-credit + r3 format/cite
            minus author-mention penalty, clipped to [0, 1].

    The returned dict keeps the legacy key schema so the BatchRewardManager
    / reward-loop manager can stack it alongside other data sources.
    """
    mode = _get_3gpp_reward_mode()

    gt_group = _normalize_3gpp_group(ground_truth)
    pred_group = _parse_3gpp_answer(solution_str)
    exact_match = float(gt_group is not None and pred_group == gt_group)
    pred_parsed = float(pred_group is not None)

    # ----- v1: legacy binary -------------------------------------------------
    if mode == "v1":
        return {
            "score": exact_match,
            "acc":   exact_match,
            "r1":    exact_match,
            "r2":    0.0,
            "r3":    0.0,
            "pred_choice_parsed": pred_parsed,
            "pred_rule_matched":  0.0,
            "gt_rule_id":         0.0,
        }

    # ----- v3: v1 binary + process-aware shaping (CoT-aware) ---------------
    # Built on v1 base (NOT v2 — v2's neighbour-credit + format/citation +
    # author-penalty combo was found to underperform v1 in practice).
    # v3 adds 4 independent shaping signals derived from the v2.7_rf failure
    # mode analysis (2026-05-27). See `_REWARD_MODE_DEFAULT` docstring above
    # for full rationale.
    if mode == "v3":
        r1 = exact_match
        r_step2_recall = 0.0
        r_anti_reject  = 0.0
        r_same_family  = 0.0
        r_goldphrase   = 0.0

        # R2 — Step-2 recall bonus.
        # The failure analysis shows 31.6% of wrong cases never recall GT in
        # Step 2 at all. Giving a small bonus when the GT label appears in
        # the Step 2 candidate block converts these from sparse-zero into
        # learnable: the model gets *some* signal for considering the right
        # answer, even if it ultimately picks wrong.
        step2_block = _extract_step2_block(solution_str or "")
        if gt_group is not None and step2_block and gt_group in step2_block:
            r_step2_recall = _V3_STEP2_RECALL_BONUS

        # R3 — anti-reject-GT penalty (the BIGGEST lever).
        # 67.5% of wrong cases come from "Step 4: Reject <GT>:" — the model
        # had GT as a candidate and talked itself out of it. v1's monolithic
        # 0/1 reward gives no gradient against this; v3 explicitly penalizes
        # rejecting the GT label when the final answer is wrong.
        rejected_labels = _extract_rejected_labels(solution_str or "")
        if gt_group and gt_group in rejected_labels and not exact_match:
            r_anti_reject = -_V3_ANTI_REJECT_GT_PENALTY

        # R4 — same-family partial credit.
        # 51% of wrong predictions stay within family (SA/CT/RAN). Giving a
        # moderate partial credit for wrong-same-family vs cross-family teaches
        # the model "at least get the family right before disambiguating".
        # This is broader and simpler than v2's narrow neighbour set.
        if not exact_match and _same_family(gt_group, pred_group):
            r_same_family = _V3_SAME_FAMILY_CREDIT

        # R5 — gold-label phrase suppression.
        # The phrase "the gold label is X" leaked into 4.4% of training CoT
        # from the Deepseek RF generation (prompt had GOLD LABEL field). On
        # held-out data the model hallucinates this phrase with random labels.
        # Apply a small flat penalty whenever any goldphrase variant appears.
        if _has_gold_label_phrase(solution_str):
            r_goldphrase = -_V3_GOLDPHRASE_PENALTY

        raw_score = r1 + r_step2_recall + r_anti_reject + r_same_family + r_goldphrase
        score = max(_V3_SCORE_LO, min(_V3_SCORE_HI, raw_score))

        return {
            "score": float(score),
            "acc":   float(exact_match),
            "r1":    float(r1),
            # Use r2 / r3 channels to carry v3 shaping signals for rollout-log
            # introspection (advantages don't depend on these names; we just
            # reuse the dict keys the schema already declared).
            "r2":    float(r_step2_recall + r_same_family),
            "r3":    float(r_anti_reject + r_goldphrase),
            "pred_choice_parsed": pred_parsed,
            "pred_rule_matched":  0.0,
            # Repurpose for offline analysis: encode the 4-bit flag pattern
            # (step2_recall + anti_reject + same_family + goldphrase).
            "gt_rule_id":         float(
                (1.0 if r_step2_recall > 0 else 0.0)
                + (2.0 if r_anti_reject < 0 else 0.0)
                + (4.0 if r_same_family > 0 else 0.0)
                + (8.0 if r_goldphrase < 0 else 0.0)
            ),
        }

    # ----- v2: multi-component ----------------------------------------------
    r1 = exact_match

    # r2 — neighbour-WG partial credit (only when wrong)
    r2 = 0.0
    if gt_group is not None and pred_group is not None and not exact_match:
        if _is_confusable_neighbour(gt_group, pred_group):
            r2 = _R2_NEIGHBOUR_CREDIT

    # r3 — format + citation bonus / long-output penalty
    r3 = 0.0
    post = _post_think(solution_str or "")
    total_len = len(solution_str or "")
    # Format bonus: short post-think part that is a clean JSON object.
    if len(post) <= 200 and _DIRECT_JSON_RE.match(post.strip()):
        r3 += _R3_FORMAT_BONUS
    # Long-output penalty: regrowth of base-model CoT (>8 K chars).
    if total_len > 8000:
        r3 -= _R3_LONG_PENALTY
    # Citation bonus: TS-series citation that maps to the *predicted* WG.
    # Only credit if the model is committing to a defensible anchor; we do
    # not award when wrong, to avoid teaching the model to confabulate TS
    # numbers (Part B §13.2 Mechanism 2 noted hallucinated citations).
    if pred_group is not None and exact_match:
        ts_owners = _extract_ts_owners(solution_str)
        if any(owner == pred_group for owner in ts_owners):
            r3 += _R3_CITATION_BONUS

    # Author/sender heuristic penalty. The reward function only sees the
    # model's output (no prompt), so we approximate "most-mentioned WG in
    # the prompt" with "most-mentioned WG in the rationale" — the model
    # typically restates the prompt's key tokens. False-positive cost is
    # low because the penalty only fires when the prediction is also wrong.
    author_penalty = 0.0
    if pred_group is not None and not exact_match:
        mm = _most_mentioned_wg(solution_str)
        if mm is not None and mm == pred_group:
            author_penalty = _AUTHOR_PENALTY

    score = max(_SCORE_LO, min(_SCORE_HI, r1 + r2 + r3 - author_penalty))

    return {
        "score": float(score),
        "acc":   float(exact_match),   # acc tracks exact-match for monitoring parity with v1
        "r1":    float(r1),
        "r2":    float(r2),
        "r3":    float(r3),
        # The remaining keys exist only to keep the dict-schema stable across
        # branches. We use `gt_rule_id` to carry the author-penalty signal
        # for offline analysis (rollout logs already record gt_rule_id=0 for
        # 3gpp under v1, so re-purposing it for v2 telemetry is safe — the
        # field is not consumed elsewhere for 3gpp samples).
        "pred_choice_parsed": pred_parsed,
        "pred_rule_matched":  0.0,
        "gt_rule_id":         float(author_penalty),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict[str, Any]:
    """Compute deterministic R1+R2+R3 reward for one sample."""
    del data_source, kwargs

    gt_choice = _normalize_choice(ground_truth)
    if gt_choice is None:
        return {"score": 0.0, "acc": 0.0, "error": f"bad ground_truth: {ground_truth!r}"}

    # --- parse reference features ----------------------------------------
    ref_text = _extract_ref_text(extra_info)
    ref_feats = _parse_ref_features(ref_text)
    gt_rule_id, gt_class_from_rules = _apply_s8_rules(ref_feats)

    # --- parse model output -----------------------------------------------
    pred_choice   = _parse_answer(solution_str)
    pred_rule_id  = _parse_rule_hit(solution_str)
    pred_metrics  = _parse_calc_metrics(solution_str)

    # --- R1: final answer -------------------------------------------------
    r1 = 1.0 if pred_choice == gt_choice else 0.0

    # --- R2: correct rule hit --------------------------------------------
    r2 = 0.5 if (pred_rule_id is not None and pred_rule_id == gt_rule_id) else 0.0

    # --- R3: key metric values within tolerance --------------------------
    key_metrics = _RULE_KEY_METRICS.get(gt_rule_id, ["high_speed_ratio_gt40", "rb_low_ratio_lt160"])
    r3_earned = 0.0
    r3_detail: dict[str, Any] = {}
    for feat_name in key_metrics:
        ref_val   = ref_feats.get(feat_name)
        model_val = pred_metrics.get(feat_name)
        if model_val is not None and _metric_close(model_val, ref_val):
            r3_earned += _R3_PER_METRIC
            r3_detail[feat_name] = "ok"
        else:
            r3_detail[feat_name] = f"miss(model={model_val}, ref={ref_val})"
    r3 = min(r3_earned, _R3_MAX)

    # TELELOGS_REWARD_MODE controls whether telelogs uses the decomposed
    # R1+R2+R3 reward (default, unset) or pure binary final-class match
    # (`binary`). Read on every call so shell-side flips are picked up without
    # re-import, mirroring how MCQ_REWARD_MODE / TELEMATH_REWARD_MODE /
    # THREEGPP_REWARD_MODE work. Conference-paper companion runs use `binary`
    # to match the paper's "every axis is pure binary" framing; journal
    # extension keeps the default R1+R2+R3 rule-decomposed reward.
    _tlogs_mode = os.environ.get("TELELOGS_REWARD_MODE", "").strip().lower()
    if _tlogs_mode == "binary":
        score = r1   # already in [0, 1]
    else:
        score = min(r1 + r2 + r3, 1.0)

    # Only return numeric fields: everything here goes into `reward_extra_info`
    # and is aggregated per data_source in validation. String labels
    # (pred_choice / gt_choice) and dict diagnostics (r3_detail) belong in the
    # rollout dump (`trainer.validation_data_dir`), not in the metric panel.
    return {
        "score":              float(score),
        "acc":                float(r1 >= 1.0),
        "r1":                 float(r1),
        "r2":                 float(r2),
        "r3":                 float(r3),
        "pred_choice_parsed": float(pred_choice is not None),
        "pred_rule_matched":  float(pred_rule_id is not None and pred_rule_id == gt_rule_id),
        "gt_rule_id":         float(gt_rule_id),
    }


def compute_score_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    fallback_score_fn=None,
    **kwargs,
) -> list[Any]:
    """Batch version – dispatches telelogs samples here, others to fallback."""
    results: list[Any] = []
    for i, (ds, sol, gt, ei) in enumerate(
        zip(data_sources, solution_strs, ground_truths, extra_infos)
    ):
        src = str(ds).strip().lower() if ds is not None else ""
        if src in {"telelogs", "telelogs_troubleshooting"}:
            results.append(
                compute_score(
                    data_source=ds,
                    solution_str=sol,
                    ground_truth=gt,
                    extra_info=ei,
                    **kwargs,
                )
            )
        elif src in _3GPP_SOURCES:
            results.append(compute_3gpp_score(solution_str=sol, ground_truth=gt))
        elif src in _MCQ_SOURCES:
            results.append(
                compute_mcq_score(
                    solution_str=sol,
                    ground_truth=gt,
                    extra_info=ei,
                    data_source=ds,
                )
            )
        elif src in _TELEMATH_SOURCES:
            # Explicit TeleMath / math-source branch. Resolves via
            # compute_telemath_score which reads TELEMATH_REWARD_MODE per call:
            #   binary → byte-identical to the prior _telemath_fallback path
            #   v2     → multi-component shaping reward
            # The legacy fallback below remains as the ultimate safety net for
            # math-shaped data_sources we have not yet enumerated here.
            results.append(
                compute_telemath_score(
                    solution_str=sol,
                    ground_truth=gt,
                    extra_info=ei,
                )
            )
        elif fallback_score_fn is not None:
            r = fallback_score_fn(
                data_source=ds,
                solution_str=sol,
                ground_truth=gt,
                extra_info=ei,
                **kwargs,
            )
            # Validation asserts that every per-sample reward_extra_info key
            # is the same length as sample_scores. The fallback (e.g.
            # prime.math_equal for TeleMath) returns a scalar, which would
            # leave 'score' missing for those samples and fail the assert.
            # Wrap into a dict so every sample contributes a 'score' entry.
            if not isinstance(r, dict):
                r = {"score": float(r), "acc": float(r)}
            results.append(r)
        else:
            results.append({"score": 0.0, "acc": 0.0})

    # ------------------------------------------------------------------
    # Schema alignment: verl's DataProto.chunk() asserts every non-tensor
    # field has the same length as the batch.  Different reward branches
    # here return different key sets (telelogs/3gpp emit r1..r3 + diagnostic
    # flags, MCQ emits pred_letter_parsed, TeleMath fallback only emits
    # score/acc).  Take the union of keys across the whole batch and pad
    # missing entries with 0.0 so every sample contributes to every key.
    # ------------------------------------------------------------------
    all_keys: set[str] = set()
    for r in results:
        if isinstance(r, dict):
            all_keys.update(r.keys())
    for r in results:
        if isinstance(r, dict):
            for k in all_keys:
                if k not in r:
                    r[k] = 0.0
    return results
