#!/usr/bin/env bash
# Qwen3.5-9B DAPO on stage2.0_grpo_eval — smaller-model sanity-check run.
#
# Compared to the 27B runs:
#   - Model: 9B (one quarter the size) → lots of memory headroom
#   - Data:  stage2.0_grpo_eval (smaller mix curated for fast eval cadence)
#   - Reward: same v2 stack (MCQ + TeleMath) with the in-code r2-gate patch
#     applied 2026-05-19 (see telelogs_symbolic.py:627)
#
# The 27B DAPO infra (GEN_TP=4, FSDP across 8 GPUs, max_num_batched_tokens=32768)
# is over-provisioned for 9B but harmless — we let it run as-is to keep the
# stack identical and the comparison clean. If we want to speed up later we
# can drop GEN_TP to 1 or 2.

# Qwen3.5-9B triggers vLLM's Qwen3-Next fused-kernel JIT path which requires
# `ninja` on PATH (ninja IS installed in qwen35 conda env at
# /dpc/kuin0100/conda_env/qwen35/bin/ninja, but the screen launched from
# `(base)` env doesn't have it in PATH → vLLM workers crash with
# FileNotFoundError: 'ninja'). Prepend the qwen35 bin dir so the JIT call
# resolves the binary at runtime.
export PATH=/dpc/kuin0100/conda_env/qwen35/bin:$PATH

export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-9B-TelecomInstruct_v2.6plus_qlora_stage3_checkpoint-4076
export STAGE_NAME=stage2.0_grpo_eval
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_grpoeval_qwen35_9b_v2_6plus_4076

# Reward mode: same as 27B newreward run
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
