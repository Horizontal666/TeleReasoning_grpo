#!/usr/bin/env bash
# DAPO on the annealing-R1 merged 9B model, with **v1 (binary) reward** stack.
# Sibling of _launch_dapov2_9b_annealing_r1.sh which uses v2 reward.
# Together they complete the 2x2 matrix: {7742 vs annealed} x {v1 vs v2 reward}.
#
# v1 reward stack (binary):
#   telelogs : rule-based R1+R2+R3 (unchanged across versions)
#   TeleMath : binary math_equal {0,1}
#   3gpp     : binary exact-match {0,1}
#   MCQ      : binary letter-match {0,1}

export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

# Annealed checkpoint (same as the v2-reward annealing_r1 DAPO)
export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-9B-TelecomInstruct_v2.6_try_v0_annealing_r1_qlora_stage3_checkpoint-34
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_annealing_r1_v1reward

# v1 reward modes — explicit, so the toggle is unambiguous in logs.
export THREEGPP_REWARD_MODE=v1
export MCQ_REWARD_MODE=binary
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
