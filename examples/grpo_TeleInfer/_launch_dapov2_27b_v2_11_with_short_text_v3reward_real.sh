#!/usr/bin/env bash
# DAPO with v3 (process-aware) reward on Qwen3.5-27B SFT v2.11_with_short_text.
# Auto-detects latest merged checkpoint.

module purge 2>/dev/null || true
module load intel_h200_gpu 2>/dev/null || true
module load miniconda/3 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate /dpc/kuin0100/conda_env/qwen35
export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

MERGED_BASE=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged
INIT_MODEL=$(ls -d $MERGED_BASE/Qwen3.5-27B-TelecomInstruct_v2.11_with_short_text_qlora_stage3_checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$INIT_MODEL" ]; then
    echo "[launcher] ERROR: no v2.11_with_short_text 27B merged checkpoint found in $MERGED_BASE" >&2
    exit 1
fi
CKPT_NAME=$(basename "$INIT_MODEL" | grep -oE 'checkpoint-[0-9]+')

export INIT_MODEL
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_27b_v2_11_with_short_text_${CKPT_NAME}_v3rs

export THREEGPP_REWARD_MODE=v3
export MCQ_REWARD_MODE=binary
export TELEMATH_REWARD_MODE=binary
export THREEGPP_V3_ANTI_REJECT_GT_PENALTY=0.30
export MAX_CKPT=${MAX_CKPT:-2}  # user 2026-05-31: keep latest 2 actor ckpts (safety: latest-1 acts as fallback resume if latest save fails)

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] THREEGPP_REWARD_MODE=v3 (STRONG)"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
