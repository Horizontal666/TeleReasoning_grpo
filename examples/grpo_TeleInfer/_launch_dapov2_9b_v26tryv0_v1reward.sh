#!/usr/bin/env bash
# Sibling run to _launch_dapov2_9b_v26tryv0.sh — SAME model + SAME data,
# but reverted to v1 (binary) reward for every source. Lets us A/B
# v1 vs v2 reward holding everything else fixed.
#
# v1 reward stack:
#   telelogs : rule-based R1+R2+R3 (no v1/v2 toggle, unchanged)
#   TeleMath : binary math_equal {0,1}
#   3gpp     : binary exact-match {0,1}
#   MCQ      : binary letter-match {0,1}
#
# Compare against the v2 run on the same model+data (screen wang,
# experiment dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v2_6_tryv0_7742)
# to evaluate whether v2's multi-component reward actually helps or
# whether v1 binary is enough (DeepSeek-R1 / ORZ stance).

# Qwen3-Next fused kernels need ninja on PATH
export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-9B-TelecomInstruct_v2.6_try_v0_qlora_stage3_checkpoint-7742
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v2_6_tryv0_7742_v1reward

# v1 reward modes — explicit, so the toggle is unambiguous in logs.
export THREEGPP_REWARD_MODE=v1   # parent script defaults to v2; force v1
export MCQ_REWARD_MODE=binary    # also the default; explicit for clarity
export TELEMATH_REWARD_MODE=binary

# Keep latest 4 ckpts
export MAX_CKPT=4

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] THREEGPP_REWARD_MODE=$THREEGPP_REWARD_MODE"
echo "[launcher] MCQ_REWARD_MODE=$MCQ_REWARD_MODE"
echo "[launcher] TELEMATH_REWARD_MODE=$TELEMATH_REWARD_MODE"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
