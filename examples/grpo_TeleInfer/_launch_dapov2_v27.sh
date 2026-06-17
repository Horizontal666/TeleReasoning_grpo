#!/usr/bin/env bash
export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-27B-TelecomInstruct_v2.7_qlora_stage3_checkpoint-2084
export STAGE_NAME=stage2.0
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_qwen35_27b_v2_7_2084
# Save frequency: omitted here so the main launcher's default (SAVE_FREQ=5)
# takes effect. The dapo_ray_trainer.py sleep+wake patch was validated for
# 6 consecutive cycles at SAVE_FREQ=1 (steps 11-16, 2026-05-17 09:42 run);
# safe to drop back to the standard cadence.
# Note: do NOT set PYTORCH_ALLOC_CONF=expandable_segments:True with vLLM TP>1.
# vLLM's custom_all_reduce uses CUDA IPC which requires contiguous physical
# backing; expandable segments break that and the EngineCore dies at init with
# "Cuda error custom_all_reduce.cuh:455 invalid argument". We rely on
# sleep_replicas() before save_checkpoint (dapo_ray_trainer.py) instead — that
# returns all 137 GiB of vLLM's footprint to the allocator before the FSDP
# gather, sidestepping fragmentation entirely.
echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
exec bash /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl/examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh
