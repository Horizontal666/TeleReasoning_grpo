#!/usr/bin/env bash
# Wrapper: stage2.0_dapo data + v2.7 ckpt-2084 model
export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-27B-TelecomInstruct_v2.7_qlora_stage3_checkpoint-2084
export STAGE_NAME=stage2.0_dapo
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_dapo_qwen35_27b_v2_7_2084
echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
exec bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
