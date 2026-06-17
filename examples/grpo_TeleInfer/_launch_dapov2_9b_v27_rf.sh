#!/usr/bin/env bash
# DAPO v2 reward on Qwen3.5-9B SFT v2.7_rf (the new RF-CoT SFT).
# Pair with _launch_dapov2_9b_v27_rf_v1reward.sh for A/B reward comparison.
#
# INIT_MODEL is auto-detected at run time from latest v2.7 merged checkpoint.

# Ensure qwen35 conda env (transformers etc.) — important when launcher runs from
# a screen that previously had a different env active (e.g. grpo_py311 / base).
module purge 2>/dev/null || true
module load intel_h200_gpu 2>/dev/null || true
module load miniconda/3 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate /dpc/kuin0100/conda_env/qwen35
export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

# Auto-detect latest v2.7 merged model
MERGED_BASE=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged
INIT_MODEL=$(ls -d $MERGED_BASE/Qwen3.5-9B-TelecomInstruct_v2.7_rf_qlora_stage3_checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$INIT_MODEL" ]; then
    echo "[launcher] ERROR: no v2.7_rf merged checkpoint found in $MERGED_BASE/"
    echo "[launcher] expected pattern: Qwen3.5-9B-TelecomInstruct_v2.7_rf_qlora_stage3_checkpoint-*"
    exit 1
fi
CKPT_NAME=$(basename "$INIT_MODEL" | grep -oE 'checkpoint-[0-9]+')

export INIT_MODEL
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v27_rf_${CKPT_NAME}_v2reward

# v2 reward (multi-component with r2-gate)
export MCQ_REWARD_MODE=v2
export TELEMATH_REWARD_MODE=v2
# THREEGPP defaults to v2 inside the parent script

export MAX_CKPT=4

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] MCQ_REWARD_MODE=$MCQ_REWARD_MODE  (v2)"
echo "[launcher] TELEMATH_REWARD_MODE=$TELEMATH_REWARD_MODE  (v2)"
echo "[launcher] THREEGPP_REWARD_MODE=(default v2)"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
