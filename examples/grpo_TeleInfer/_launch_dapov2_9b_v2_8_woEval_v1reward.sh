#!/usr/bin/env bash
# DAPO with v1 (binary) reward on Qwen3.5-9B SFT v2.8_rf_woEval.
#
# v2.8_rf_woEval = v2.8_rf with every train row whose normalised question stem
# overlaps any GSMA --full eval sample removed (94,147 → 64,953, -31%).
# This is the CONTAMINATION-FREE SFT base — the conference paper's defensible
# "we strictly dedupe against eval before training" experiment.
#
# v1 reward = pure binary on every axis (same as _launch_dapov2_9b_v2_8_6444_v1reward.sh):
#   3gpp     : binary exact-match {0,1}       (THREEGPP_REWARD_MODE=v1)
#   MCQ      : binary letter-match {0,1}      (MCQ_REWARD_MODE=binary)
#                                             — ORAN / srsRAN / TeleQnA / TeleTables
#   TeleMath : binary math_equal {0,1}        (TELEMATH_REWARD_MODE=binary)
#   telelogs : binary final-class match {0,1} (TELELOGS_REWARD_MODE=binary, NEW)
#                                             — was R1+R2+R3 by default
#
# Pre-requisite: 9B SFT on v2.8_rf_woEval must have completed + been merged to
# outputs/model_FT_merged/Qwen3.5-9B-TelecomInstruct_v2.8_woEval_qlora_stage3_checkpoint-XXXX.
# Run scripts/launch_sft_v2.8_woEval_qwen3.5_9b.sh first, then merge with
# TeleSFT/export_full_model_qwen35.py (or run_ft.sh export).
#
# To stop:   Ctrl+C inside the screen.  NEVER scancel.

export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

# Auto-pick newest merged checkpoint for v2.8_woEval
MERGED_BASE=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged
INIT_MODEL=$(ls -d $MERGED_BASE/Qwen3.5-9B-TelecomInstruct_v2.8_woEval_qlora_stage3_checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$INIT_MODEL" ]; then
    echo "[launcher] ERROR: no v2.8_woEval merged checkpoint found in $MERGED_BASE"
    echo "[launcher] Run SFT first (scripts/launch_sft_v2.8_woEval_qwen3.5_9b.sh)"
    echo "[launcher] Then merge LoRA adapter to HF format."
    exit 1
fi
CKPT_NAME=$(basename "$INIT_MODEL" | grep -oE 'checkpoint-[0-9]+')

export INIT_MODEL
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v2_8_woEval_${CKPT_NAME}_v1reward

# v1 reward modes — pure binary on every axis (conference-paper companion)
export THREEGPP_REWARD_MODE=v1
export MCQ_REWARD_MODE=binary
export TELEMATH_REWARD_MODE=binary
# TELELOGS_REWARD_MODE=binary (was R1+R2+R3 by default).
# Conference paper drops the staged-reward design; journal keeps R1+R2+R3.
export TELELOGS_REWARD_MODE=binary

# Keep up to 2 checkpoints
export MAX_CKPT=${MAX_CKPT:-2}

# Deterministic val for clean step-to-step + DAPO vs GRPO comparison (2026-06-08)
# Greedy decoding → same model state → identical val score on rerun.
# Training rollout (rollout.temperature/top_p/n) is unaffected.
export VAL_TEMPERATURE=0.0
export VAL_DO_SAMPLE=False
# Val at step 0 so DAPO curve has same anchor as GRPO (which val_before_train=True hardcoded)
export VAL_BEFORE_TRAIN=True

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] algorithm: DAPO (main_dapo)"
echo "[launcher] THREEGPP_REWARD_MODE=v1 (binary exact-match)"
echo "[launcher] MCQ_REWARD_MODE=binary"
echo "[launcher] TELEMATH_REWARD_MODE=binary"
echo "[launcher] TELELOGS_REWARD_MODE=binary  (was R1+R2+R3 by default)"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
