#!/usr/bin/env bash
# DAPO with v3 (process-aware) reward on Qwen3.5-9B SFT v2.8 ckpt-6444.
# v3 reward = v1 binary base + 4 shaping signals:
#   R_step2_recall   (+0.10) : reward listing GT in Step 2 candidates
#   R_anti_reject_gt (-0.30) : STRONG variant — penalty when "Reject <GT>:" leak
#                              and final answer is wrong (biggest lever)
#   R_same_family    (+0.15) : partial credit when wrong but same SA/CT/RAN family
#   R_goldphrase     (-0.05) : penalty for "the gold label is X" leak
#
# Compare step-by-step against v1 reward (binary) data from bohao annealing_r1_v1reward
# run that just preceded this — to evaluate whether v3 process-aware reward
# unlocks more 3gpp gain than v1 binary.

export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

# v2.8 SFT base (94,147 rows training including reverse-CoT injection, see memory)
export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-9B-TelecomInstruct_v2.8_qlora_stage3_checkpoint-6444
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v2_8_6444_v3rs

# v3 reward (process-aware) for 3GPP only
export THREEGPP_REWARD_MODE=v3
# Keep MCQ/TeleMath at binary (v1-compatible — sticking with prior user preference
# that v2 multi-component reward did not outperform v1 binary)
export MCQ_REWARD_MODE=binary
export TELEMATH_REWARD_MODE=binary

# v3 STRONG tuning — anti-reject-GT penalty doubled to -0.30
export THREEGPP_V3_ANTI_REJECT_GT_PENALTY=0.30

export MAX_CKPT=${MAX_CKPT:-2}  # user 2026-05-29: keep at most 2 checkpoints

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] THREEGPP_REWARD_MODE=v3 (process-aware, R_step2_recall + R_anti_reject_gt=0.30 + R_same_family + R_goldphrase)"
echo "[launcher] MCQ_REWARD_MODE=binary (v1-compatible)"
echo "[launcher] TELEMATH_REWARD_MODE=binary (v1-compatible)"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
