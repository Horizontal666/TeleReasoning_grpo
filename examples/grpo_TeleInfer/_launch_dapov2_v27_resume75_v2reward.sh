#!/usr/bin/env bash
# Resume the dropped run (experiment_name has the `_dapo_` infix) from
# global_step_75 with MCQ_REWARD_MODE=v2 + TELEMATH_REWARD_MODE=v2 +
# max_actor_ckpt_to_keep=4. step_75 is the earliest LOADABLE ckpt because
# max_actor_ckpt_to_keep=2 in the prior run pruned earlier actor weights
# (step_70 and earlier kept only data.pt stubs).
#
# State at step_75:
#   teletable=0.82  (down from peak 0.92 at step 30, drop accelerating)
#   response_length=822  (down from ~1900, but not yet collapsed to <600)
#   telelogs=0.978 / TeleMath=0.766 / 3gpp=0.72 / teleqna=0.78 / oranbench=0.80 / srsran=0.84
#
# v2 reward kicks in immediately on first new rollout; over the next 5-10
# steps Adam momentum reorients toward "longer CoT + format" → teletable
# should arrest the slide and slowly recover.

export INIT_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-27B-TelecomInstruct_v2.7_qlora_stage3_checkpoint-2084
export STAGE_NAME=stage2.0_dapo
export EXPERIMENT_NAME=dapo_mixed_v3_stage2_0_dapo_qwen35_27b_v2_7_2084

# Reward mode upgrade — see telelogs_symbolic.py / failure_analysis MCQ §12.5
export MCQ_REWARD_MODE=v2
export TELEMATH_REWARD_MODE=v2
# THREEGPP was already v2 by default in the script

# Keep 3 ckpts (was 2) so we have a safer rollback buffer
export MAX_CKPT=3

# Resume from step_75 (last loadable ckpt with teletable not yet collapsed)
RESUME_PATH=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/grpo/checkpoints/dapo_mixed_v3_qwen35_27B/dapo_mixed_v3_stage2_0_dapo_qwen35_27b_v2_7_2084/global_step_75

echo "[launcher] INIT_MODEL=$INIT_MODEL"
echo "[launcher] STAGE_NAME=$STAGE_NAME"
echo "[launcher] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[launcher] MCQ_REWARD_MODE=$MCQ_REWARD_MODE"
echo "[launcher] TELEMATH_REWARD_MODE=$TELEMATH_REWARD_MODE"
echo "[launcher] MAX_CKPT=$MAX_CKPT"
echo "[launcher] RESUME_FROM=$RESUME_PATH"

# Sanity: confirm ckpt exists before launch
if [[ ! -d "$RESUME_PATH/actor" ]]; then
  echo "[launcher] ERROR: $RESUME_PATH/actor not found — ckpt is a stub, cannot resume" >&2
  exit 1
fi

# Hand off to the main DAPO script. Hydra overrides at the end add the
# explicit resume path (overrides the script's default resume_mode=auto so
# verl ignores latest_checkpointed_iteration.txt=80 and uses step_75
# instead).
exec bash /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl/examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_dapov2.sh \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$RESUME_PATH"
