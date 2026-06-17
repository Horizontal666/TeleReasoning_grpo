#!/usr/bin/env bash
# DAPO with v3 (process-aware) reward on Qwen3.5-9B SFT v2.11.
# Auto-detects latest merged v2.11 checkpoint.

export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

MERGED_BASE=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged
INIT_MODEL=$(ls -d $MERGED_BASE/Qwen3.5-9B-TelecomInstruct_v2.11_qlora_stage3_checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$INIT_MODEL" ]; then
    echo "[launcher] ERROR: no v2.11 merged checkpoint found in $MERGED_BASE" >&2
    exit 1
fi
CKPT_NAME=$(basename "$INIT_MODEL" | grep -oE 'checkpoint-[0-9]+')

export INIT_MODEL
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v2_11_${CKPT_NAME}_v3rs

# v3 reward (process-aware) for 3GPP
export THREEGPP_REWARD_MODE=v3
export MCQ_REWARD_MODE=binary
export TELEMATH_REWARD_MODE=binary

# v3 STRONG variant — anti-reject-GT penalty doubled to -0.30
export THREEGPP_V3_ANTI_REJECT_GT_PENALTY=0.30

export MAX_CKPT=${MAX_CKPT:-2}

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] THREEGPP_REWARD_MODE=v3 (R_step2_recall + R_anti_reject_gt=0.30 + R_same_family + R_goldphrase)"
echo "[launcher] MCQ_REWARD_MODE=binary"
echo "[launcher] TELEMATH_REWARD_MODE=binary"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
