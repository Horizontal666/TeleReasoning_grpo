#!/usr/bin/env bash

set -euo pipefail

PROFILE="${1:-}"
if [[ -z "${PROFILE}" ]]; then
    echo "Usage: bash run_telemath_grpo_ablation.sh A0|A1|A2|A3" >&2
    exit 1
fi

REPO_ROOT="/dpc/kuin0100/bohao/202509_InferenceModel"
# shellcheck source=/dev/null
. "${REPO_ROOT}/scripts/use_project_cache.sh"
FLOWRL_SCRIPT="${REPO_ROOT}/Inference/verl/examples/grpo_TeleInfer/flowrl_qwen3-8b_TeleInfer.sh"
ANALYZE_SCRIPT="${REPO_ROOT}/Inference/verl/examples/grpo_TeleInfer/analyze_rollout_groups.py"
EVAL_SCRIPT="${REPO_ROOT}/TeleReasoning_Eval_git/old_vllmUnify/telemath_eval_vllm_mathverify.py"
COLLECT_SCRIPT="${REPO_ROOT}/Inference/verl/examples/grpo_TeleInfer/collect_telemath_eval_metrics.py"

EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-Qwen2.5-7B-Instruct-telemath_self_gen_v0_root_cause}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_PREFIX}_${PROFILE}}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
RESUME_MODE="${RESUME_MODE:-disable}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-$TOTAL_TRAINING_STEPS}"

RUN_ROLLOUT_ANALYSIS="${RUN_ROLLOUT_ANALYSIS:-1}"
RUN_MODEL_MERGE="${RUN_MODEL_MERGE:-1}"
RUN_OFFLINE_EVAL="${RUN_OFFLINE_EVAL:-1}"
RUN_EVAL_SUMMARY="${RUN_EVAL_SUMMARY:-1}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-16000}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0}"
EVAL_TOP_P="${EVAL_TOP_P:-1}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_REL_TOL="${EVAL_REL_TOL:-0.01}"
EVAL_ABS_TOL="${EVAL_ABS_TOL:-0.01}"

TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/data/GRPO/telemath_self_gen_v0/train.parquet}"
TEST_PARQUET="${TEST_PARQUET:-${REPO_ROOT}/data/GRPO/telemath_self_gen_v0/test.parquet}"
EVAL_DATASET="${EVAL_DATASET:-${REPO_ROOT}/data/eval_benchmark_CT/telemath/telemath_chat_template.json}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/grpo/checkpoints/TeleReasoning_GRPO}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${REPO_ROOT}/outputs/eval/rollout}"
VALIDATION_ROOT="${VALIDATION_ROOT:-${REPO_ROOT}/outputs/eval/validation}"
ROLLOUT_ANALYSIS_ROOT="${ROLLOUT_ANALYSIS_ROOT:-${REPO_ROOT}/outputs/eval/root_cause_ablation/rollout_analysis}"
RAW_EVAL_ROOT="${RAW_EVAL_ROOT:-${REPO_ROOT}/outputs/eval/root_cause_ablation/raw}"
EVAL_SUMMARY_ROOT="${EVAL_SUMMARY_ROOT:-${REPO_ROOT}/outputs/eval/root_cause_ablation/summary}"
METADATA_ROOT="${METADATA_ROOT:-${REPO_ROOT}/outputs/eval/root_cause_ablation/metadata}"
VISIBLE_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

case "${PROFILE}" in
    A0)
        TRAIN_BATCH_SIZE=128
        ROLLOUT_N=8
        PPO_MINI_BATCH_SIZE=32
        MAX_RESPONSE_LENGTH=12000
        ;;
    A1)
        TRAIN_BATCH_SIZE=128
        ROLLOUT_N=8
        PPO_MINI_BATCH_SIZE=32
        MAX_RESPONSE_LENGTH=4096
        ;;
    A2)
        TRAIN_BATCH_SIZE=32
        ROLLOUT_N=8
        PPO_MINI_BATCH_SIZE=16
        MAX_RESPONSE_LENGTH=4096
        ;;
    A3)
        TRAIN_BATCH_SIZE=32
        ROLLOUT_N=16
        PPO_MINI_BATCH_SIZE=16
        MAX_RESPONSE_LENGTH=4096
        ;;
    *)
        echo "Unknown profile: ${PROFILE}. Expected A0|A1|A2|A3." >&2
        exit 1
        ;;
esac

TOKEN_BUDGET=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}"
ROLLOUT_DIR="${ROLLOUT_ROOT}/${EXPERIMENT_NAME}"
VALIDATION_DIR="${VALIDATION_ROOT}/${EXPERIMENT_NAME}"
ROLLOUT_ANALYSIS_DIR="${ROLLOUT_ANALYSIS_ROOT}/${EXPERIMENT_NAME}"
RAW_EVAL_PATH="${RAW_EVAL_ROOT}/${EXPERIMENT_NAME}_telemath_eval_vllm_mathverify.json"
EVAL_SUMMARY_DIR="${EVAL_SUMMARY_ROOT}/${EXPERIMENT_NAME}"
METADATA_PATH="${METADATA_ROOT}/${EXPERIMENT_NAME}.json"
MERGED_MODEL_DIR="${CHECKPOINT_DIR}/global_step_${TOTAL_TRAINING_STEPS}/actor/huggingface"

mkdir -p \
    "${CHECKPOINT_DIR}" \
    "${ROLLOUT_DIR}" \
    "${VALIDATION_DIR}" \
    "${ROLLOUT_ANALYSIS_DIR}" \
    "${RAW_EVAL_ROOT}" \
    "${EVAL_SUMMARY_DIR}" \
    "${METADATA_ROOT}"

STATUS="running"
START_TS="$(date +%s)"
START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_TS=""
END_ISO=""

write_metadata() {
    local duration_seconds=""
    if [[ -n "${END_TS}" ]]; then
        duration_seconds=$((END_TS - START_TS))
    fi

    cat > "${METADATA_PATH}" <<EOF
{
  "run_label": "${EXPERIMENT_NAME}",
  "profile": "${PROFILE}",
  "status": "${STATUS}",
  "start_time_utc": "${START_ISO}",
  "end_time_utc": "${END_ISO}",
  "duration_seconds": ${duration_seconds:-null},
  "total_training_steps": ${TOTAL_TRAINING_STEPS},
  "config": {
    "train_batch_size": ${TRAIN_BATCH_SIZE},
    "rollout_n": ${ROLLOUT_N},
    "ppo_mini_batch_size": ${PPO_MINI_BATCH_SIZE},
    "max_prompt_length": ${MAX_PROMPT_LENGTH},
    "max_response_length": ${MAX_RESPONSE_LENGTH},
    "kl_loss_coef": ${KL_LOSS_COEF},
    "resume_mode": "${RESUME_MODE}",
    "cuda_visible_devices": "${VISIBLE_CUDA_DEVICES}"
  },
  "paths": {
    "checkpoint_dir": "${CHECKPOINT_DIR}",
    "rollout_dir": "${ROLLOUT_DIR}",
    "validation_dir": "${VALIDATION_DIR}",
    "rollout_analysis_dir": "${ROLLOUT_ANALYSIS_DIR}",
    "raw_eval_path": "${RAW_EVAL_PATH}",
    "eval_summary_dir": "${EVAL_SUMMARY_DIR}",
    "merged_model_dir": "${MERGED_MODEL_DIR}"
  }
}
EOF
}

on_exit() {
    local exit_code=$?
    trap - EXIT
    END_TS="$(date +%s)"
    END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ "${STATUS}" == "running" ]]; then
        STATUS="failed"
    fi
    write_metadata
    exit "${exit_code}"
}
trap on_exit EXIT

echo "Launching ${PROFILE} as ${EXPERIMENT_NAME}"
echo "  train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  rollout_n=${ROLLOUT_N}"
echo "  ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
echo "  max_response_length=${MAX_RESPONSE_LENGTH}"
echo "  total_training_steps=${TOTAL_TRAINING_STEPS}"
echo "  cuda_visible_devices=${VISIBLE_CUDA_DEVICES}"

bash "${FLOWRL_SCRIPT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    trainer.validation_data_dir="${VALIDATION_DIR}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${TOKEN_BUDGET}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${TOKEN_BUDGET}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${TOKEN_BUDGET}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${TOKEN_BUDGET}"

if [[ "${RUN_ROLLOUT_ANALYSIS}" == "1" ]]; then
    python "${ANALYZE_SCRIPT}" \
        --run "${EXPERIMENT_NAME}=${ROLLOUT_DIR}" \
        --output-dir "${ROLLOUT_ANALYSIS_DIR}" \
        --expected-group-size "${ROLLOUT_N}"
fi

if [[ "${RUN_MODEL_MERGE}" == "1" ]]; then
    (
        cd "${REPO_ROOT}/Inference/verl"
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${CHECKPOINT_DIR}/global_step_${TOTAL_TRAINING_STEPS}/actor" \
            --target_dir "${MERGED_MODEL_DIR}"
    )
fi

if [[ "${RUN_OFFLINE_EVAL}" == "1" ]]; then
    python "${EVAL_SCRIPT}" \
        --model-dir "${MERGED_MODEL_DIR}" \
        --run-name "${EXPERIMENT_NAME}" \
        --dataset "${EVAL_DATASET}" \
        --batch-size "${EVAL_BATCH_SIZE}" \
        --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
        --temperature "${EVAL_TEMPERATURE}" \
        --top-p "${EVAL_TOP_P}" \
        --seed "${EVAL_SEED}" \
        --rel-tol "${EVAL_REL_TOL}" \
        --abs-tol "${EVAL_ABS_TOL}" \
        --output-path "${RAW_EVAL_PATH}"
fi

if [[ "${RUN_EVAL_SUMMARY}" == "1" ]]; then
    python "${COLLECT_SCRIPT}" \
        --run "${EXPERIMENT_NAME}=${RAW_EVAL_PATH}" \
        --dataset "${EVAL_DATASET}" \
        --train-parquet "${TRAIN_PARQUET}" \
        --test-parquet "${TEST_PARQUET}" \
        --output-dir "${EVAL_SUMMARY_DIR}"
fi

STATUS="success"
