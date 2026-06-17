#!/usr/bin/env bash
# GRPO (plain) with v3 (process-aware) reward on Qwen3.5-9B SFT v2.7_rf.
# Sibling of _launch_dapov2_9b_v27_rf_v3reward.sh — same model + reward,
# different optimizer (GRPO via main_ppo vs DAPO via main_dapo).
# Lets us A/B "DAPO clip-higher / filter_groups vs plain GRPO" under the
# identical v3 reward.

# Ensure qwen35 conda env
module purge 2>/dev/null || true
module load intel_h200_gpu 2>/dev/null || true
module load miniconda/3 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate /dpc/kuin0100/conda_env/qwen35
export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

MERGED_BASE=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged
INIT_MODEL=$(ls -d $MERGED_BASE/Qwen3.5-9B-TelecomInstruct_v2.7_rf_qlora_stage3_checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$INIT_MODEL" ]; then
    echo "[launcher] ERROR: no v2.7_rf merged checkpoint found"
    exit 1
fi
CKPT_NAME=$(basename "$INIT_MODEL" | grep -oE 'checkpoint-[0-9]+')

export INIT_MODEL
export STAGE_NAME=stage2.0_grpo_eval
# v3 reward STRONG variant — see DAPO v3 sibling for rationale (R3 doubled).
export EXPERIMENT_NAME=grpo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v27_rf_${CKPT_NAME}_v3rs

# v3 reward (process-aware) — same as DAPO v3 sibling
export THREEGPP_REWARD_MODE=v3
# v1 binary for MCQ/TeleMath (user feedback: v2 multi-component underperformed v1)
export MCQ_REWARD_MODE=binary
export TELEMATH_REWARD_MODE=binary

# v3 STRONG: R3 anti-reject-GT penalty doubled to -0.30
export THREEGPP_V3_ANTI_REJECT_GT_PENALTY=0.30

export MAX_CKPT=4

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] algorithm: GRPO (main_ppo)"
echo "[launcher] THREEGPP_REWARD_MODE=v3"
echo "[launcher] MCQ_REWARD_MODE=binary"
echo "[launcher] TELEMATH_REWARD_MODE=binary"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh
