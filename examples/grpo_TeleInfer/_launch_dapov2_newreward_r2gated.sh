#!/usr/bin/env bash
# Re-launch from global_step_25 of the failed `_newreward` run with the
# r2-gating patch applied to compute_mcq_score_v2 (telelogs_symbolic.py).
#
# Why this exists:
#   The prior `_launch_dapov2_newreward.sh` run (started 2026-05-18 16:19,
#   killed at step 50 on 2026-05-19) used MCQ_REWARD_MODE=v2 with an
#   un-gated r2 format-bonus (+0.30 for any clean ANSWER: X coda regardless
#   of correctness). The model collapsed teletable val acc 0.90 → 0.63 and
#   mean response length 3147 → 521 chars by step 50. Forensics on all 4
#   MCQ domains (teletable/teleqna/oranbench/srsran) confirmed the same
#   "guess + format" exploit; r3 (anti-hallucination) was dead (0 % firing
#   on wrong rollouts) by step 25 because the templated short rollouts
#   contain no citation patterns to police.
#
# Fix applied (telelogs_symbolic.py:627, compute_mcq_score_v2):
#   r2_eff = r2 if r1 == 1.0 else 0.0
#   score = clip(r1 + r2_eff + r3, 0, 1)
#   ⇒ wrong rollouts no longer earn +0.30, exploit basin eliminated.
#
# Init source:
#   `save_dapo_mixed_v3_stage2_0_dapo_qwen35_27b_v2_7_2084_newreward/
#    global_step_25/actor/huggingface` — the merged HF weights from the
#   last clean-ish step (val acc 0.90, length 3191, template adoption
#   ~22-30 %). The raw FSDP global_step_25 was pruned (only data.pt stub
#   remains). Loading as model.path = fresh init: optimizer state resets,
#   step counter restarts, BUT this also wipes Adam momentum that was
#   contaminated by 25 steps of the broken v2 reward — a feature, not a
#   bug (see _launch_dapov2_v27_resume75_v2reward.sh comments for the
#   "Adam momentum resists new signal" hypothesis you've already
#   diagnosed).
#
# Experiment naming:
#   Uses `..._r2gated` suffix so W&B keeps the failed run + this one as
#   separate traces for direct comparison. New ckpt dir is empty, so the
#   default resume_mode=auto in the parent script will not auto-resume
#   from the failed run's collapsed checkpoints.

export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/grpo/checkpoints/dapo_mixed_v3_qwen35_27B/save_dapo_mixed_v3_stage2_0_dapo_qwen35_27b_v2_7_2084_newreward/global_step_25/actor/huggingface
export STAGE_NAME=stage2.0_dapo
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_dapo_qwen35_27b_v2_7_2084_newreward_r2gated

# Reward modes — unchanged from the failed _newreward run. The MCQ v2 path
# now applies the in-code r2 gate, so v2 stays SAFE here. 3GPP and TeleMath
# v2 are unaffected by the patch (they have separate scoring functions).
export MCQ_REWARD_MODE=v2
export TELEMATH_REWARD_MODE=v2
# THREEGPP_REWARD_MODE defaults to v2 in the parent run script.

# Bumped from 4 → 10 so the next collapse forensic (if any) has trajectory.
export MAX_CKPT=10

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] MCQ_REWARD_MODE=$MCQ_REWARD_MODE   (gated in code; safe)"
echo "[launcher] TELEMATH_REWARD_MODE=$TELEMATH_REWARD_MODE"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

# Sanity-check the init dir before launch — fail fast if path doesn't exist
# or doesn't look like an HF model dir.
if [[ ! -f "$INIT_MODEL/config.json" || ! -f "$INIT_MODEL/model.safetensors.index.json" ]]; then
  echo "[launcher] ERROR: $INIT_MODEL is not a valid HF model dir (missing config.json or safetensors.index.json)" >&2
  exit 1
fi

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
