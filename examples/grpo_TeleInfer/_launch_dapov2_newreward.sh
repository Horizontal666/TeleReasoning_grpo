#!/usr/bin/env bash
# Fresh restart with v2 reward modes (MCQ + TeleMath).
# Old run resumed from step_75 (v1 reward weights) into v2 reward and only
# 3gpp clearly responded; srsran / oranbench / teletable stayed flat.
# Hypothesis: the policy was already biased by v1 reward and Adam momentum
# resists the new signal. A clean start from the SFT ckpt should give the
# v2 reward the full training trajectory to imprint.
#
# Folder suffix `_newreward` keeps wandb run + ckpt dir separate from the
# previous resume75 attempt — both runs preserved for comparison.

export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-27B-TelecomInstruct_v2.7_qlora_stage3_checkpoint-2084
export STAGE_NAME=stage2.0_dapo
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_dapo_qwen35_27b_v2_7_2084_newreward

# Reward mode: v2 active from step 1
export MCQ_REWARD_MODE=v2
export TELEMATH_REWARD_MODE=v2
# 3GPP already defaults to v2 in dapov2.sh

# Keep latest 4 ckpts (was 3 in resume75 attempt, 2 in original)
export MAX_CKPT=4

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] MCQ_REWARD_MODE=$MCQ_REWARD_MODE"
echo "[launcher] TELEMATH_REWARD_MODE=$TELEMATH_REWARD_MODE"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
