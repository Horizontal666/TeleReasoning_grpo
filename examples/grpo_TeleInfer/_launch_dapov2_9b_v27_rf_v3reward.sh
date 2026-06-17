#!/usr/bin/env bash
# DAPO with v3 (process-aware) reward on Qwen3.5-9B SFT v2.7_rf.
# Built on v1 binary base + 4 shaping signals derived from v2.7_rf failure
# mode analysis (2026-05-27):
#   R_step2_recall (+0.10) : reward listing the GT in Step 2 candidates
#   R_anti_reject_gt (-0.15): penalty when "Reject <GT>:" appears in Step 4
#                              and final answer is wrong (the BIGGEST lever —
#                              67.5% of v2.7 errors were "rejects-GT")
#   R_same_family    (+0.15): partial credit when wrong but same SA/CT/RAN family
#   R_goldphrase     (-0.05): penalty for "the gold label is X" leak (Deepseek
#                              RF training artifact, 4.4% of samples)
#
# Sibling of _launch_dapov2_9b_v27_rf_v1reward.sh — same model + data, only
# the 3GPP reward differs.

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
# v3 reward STRONG variant — R3 anti-reject-GT penalty doubled to -0.30 after
# observing DAPO v3 (with -0.15) still had 76-91% rejects_gt at step 21,
# essentially same as DAPO v1. The signal needed to break the habit.
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v27_rf_${CKPT_NAME}_v3rs

# v3 reward for 3GPP — see telelogs_symbolic.py _REWARD_MODE_DEFAULT docstring
export THREEGPP_REWARD_MODE=v3
# Keep MCQ/TeleMath at binary (matching v1 sibling, NOT v2 — user feedback:
# v2 multi-component performed worse than v1 binary in prior runs)
export MCQ_REWARD_MODE=binary
export TELEMATH_REWARD_MODE=binary

# v3 STRONG tuning — only override the anti-reject penalty (the biggest lever).
# Other components keep their failure-analysis defaults.
export THREEGPP_V3_ANTI_REJECT_GT_PENALTY=0.30   # was 0.15
# export THREEGPP_V3_STEP2_RECALL_BONUS=0.10
# export THREEGPP_V3_SAME_FAMILY_CREDIT=0.15
# export THREEGPP_V3_GOLDPHRASE_PENALTY=0.05

export MAX_CKPT=4

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] THREEGPP_REWARD_MODE=v3 (R_step2_recall + R_anti_reject_gt + R_same_family + R_goldphrase)"
echo "[launcher] MCQ_REWARD_MODE=binary (v1-compatible)"
echo "[launcher] TELEMATH_REWARD_MODE=binary (v1-compatible)"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
