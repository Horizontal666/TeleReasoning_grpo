#!/usr/bin/env bash
# Qwen3.5-9B (v2.6_try_v0 SFT ckpt-7742) DAPO on stage2.0_grpo_eval.
# Sibling run to _launch_dapov2_9b_grpoeval.sh — same data + same v2 reward,
# different SFT ancestor (v2.6_try_v0 vs v2.6plus). Lets us compare DAPO
# trajectories from two different SFT initializations on identical mix.

# Qwen3-Next fused kernels need ninja on PATH at JIT time. ninja IS in
# the qwen35 conda env but `(base)` shell PATH doesn't see it → workers
# crash with FileNotFoundError. Prepend so JIT resolves the binary.
export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-9B-TelecomInstruct_v2.6_try_v0_qlora_stage3_checkpoint-7742
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v2_6_tryv0_7742

# Reward modes: v2 with in-code r2-gate patch (May 19)
export MCQ_REWARD_MODE=v2
export TELEMATH_REWARD_MODE=v2
# THREEGPP defaults to v2 inside the parent script

# Keep latest 4 ckpts
export MAX_CKPT=4

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] MCQ_REWARD_MODE=$MCQ_REWARD_MODE"
echo "[launcher] TELEMATH_REWARD_MODE=$TELEMATH_REWARD_MODE"
echo "[launcher] MAX_CKPT=$MAX_CKPT"

cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
