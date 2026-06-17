#!/usr/bin/env bash
# DAPO with v1 (binary) reward on Qwen3.5-9B SFT v2.8 ckpt-6444.
# Conference-paper companion run to the existing v3reward sibling
#   (_launch_dapov2_9b_v2_8_6444_v3reward.sh).
#
# v1 reward = pure binary on EVERY axis (including telelogs):
#   3gpp     : binary exact-match {0,1}       (THREEGPP_REWARD_MODE=v1)
#   MCQ      : binary letter-match {0,1}      (MCQ_REWARD_MODE=binary)
#                                             — ORAN / srsRAN / TeleQnA / TeleTables
#   TeleMath : binary math_equal {0,1}        (TELEMATH_REWARD_MODE=binary)
#   telelogs : binary final-class match {0,1} (TELELOGS_REWARD_MODE=binary, NEW)
#                                             — was R1+R2+R3 by default
#
# Purpose:
#   - Conference paper headline:  9B SFT 74.89  ->  9B SFT+DAPO ~78-81 (expected).
#   - Journal paper extension:    27B + v3 process reward = 89.03 (already measured).
#   - The conference / journal gap isolates the v3 reward + 27B scale contribution.
#
# To stop:   Ctrl+C inside the screen.  NEVER scancel (kills the salloc node alloc).

export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

# v2.8 9B SFT base (same as the v3reward sibling)
export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-9B-TelecomInstruct_v2.8_qlora_stage3_checkpoint-6444
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v2_8_6444_v1reward

# v1 reward modes — pure binary on every axis (conference-paper companion)
export THREEGPP_REWARD_MODE=v1
export MCQ_REWARD_MODE=binary
export TELEMATH_REWARD_MODE=binary
# TELELOGS_REWARD_MODE=binary added 2026-05-31 (was R1+R2+R3 by default).
# Conference paper drops the staged-reward design; journal keeps R1+R2+R3.
export TELELOGS_REWARD_MODE=binary

# Keep up to 2 checkpoints
export MAX_CKPT=${MAX_CKPT:-2}

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] THREEGPP_REWARD_MODE=v1 (binary exact-match)"
echo "[launcher] MCQ_REWARD_MODE=binary"
echo "[launcher] TELEMATH_REWARD_MODE=binary"
echo "[launcher] TELELOGS_REWARD_MODE=binary  (NEW: was R1+R2+R3 by default)"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
