#!/usr/bin/env bash
# DAPO with v3 (process-aware) reward on Qwen3.5-27B SFT v2.8 ckpt-3222.
# Sibling of _launch_dapov2_9b_v2_8_6444_v3reward.sh — same reward design, 27B size.
# Goal: A/B 27B vs 9B under identical v3rs reward + v2.8 SFT base, to see whether
# capacity buys faster Step 4 habit suppression and higher final 3gpp acc.
#
# v3 reward = v1 binary base + 4 shaping signals (same as 9B run):
#   R_step2_recall   (+0.10) : reward listing GT in Step 2 candidates
#   R_anti_reject_gt (-0.30) : STRONG variant — penalty when "Reject <GT>:" leak
#   R_same_family    (+0.15) : partial credit when wrong but same SA/CT/RAN family
#   R_goldphrase     (-0.05) : penalty for "the gold label is X" leak

# Ensure qwen35 conda env
module purge 2>/dev/null || true
module load intel_h200_gpu 2>/dev/null || true
module load miniconda/3 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate /dpc/kuin0100/conda_env/qwen35
export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

# v2.8 SFT 27B base
export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-27B-TelecomInstruct_v2.8_qlora_stage3_checkpoint-3222
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_27b_v2_8_3222_v3rs

# v3 reward (process-aware) for 3GPP — same as 9B sibling
export THREEGPP_REWARD_MODE=v3
export MCQ_REWARD_MODE=binary
export TELEMATH_REWARD_MODE=binary

# v3 STRONG tuning — anti-reject-GT penalty doubled to -0.30
export THREEGPP_V3_ANTI_REJECT_GT_PENALTY=0.30

export MAX_CKPT=${MAX_CKPT:-2}  # user 2026-05-31: keep latest 2 actor ckpts (safety: latest-1 acts as fallback resume if latest save fails)

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] THREEGPP_REWARD_MODE=v3 (R_step2_recall + R_anti_reject_gt=0.30 + R_same_family + R_goldphrase)"
echo "[launcher] MCQ_REWARD_MODE=binary (v1-compatible)"
echo "[launcher] TELEMATH_REWARD_MODE=binary (v1-compatible)"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
