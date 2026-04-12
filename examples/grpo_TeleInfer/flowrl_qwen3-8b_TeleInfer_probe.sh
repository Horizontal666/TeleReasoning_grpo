#!/usr/bin/env bash
#
# One-pass rollout probing before GRPO training.
# It does not update model weights. Instead it:
# 1. traverses the dataset once
# 2. generates rollout_n samples per prompt
# 3. scores every rollout with the same custom reward
# 4. writes all rollouts plus per-sample averages into outputs/eval/rollout
#
# Typical usage:
#   PRETRAINED_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/model/Qwen-2.5-32B \
#   bash flowrl_qwen3-8b_TeleInfer_probe.sh
#
# Quick dry run:
#   PRETRAINED_MODEL=/path/to/model DRY_RUN=1 bash flowrl_qwen3-8b_TeleInfer_probe.sh
#
# Common overrides:
#   DATA_PROBE_PATH=/path/to/train.parquet ROLLOUT_N=8 bash flowrl_qwen3-8b_TeleInfer_probe.sh
#   TOO_EASY_THRESHOLD=1.0 TOO_HARD_THRESHOLD=-1.0 bash flowrl_qwen3-8b_TeleInfer_probe.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
export CUDA_VISIBLE_DEVICES

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
SHORT_RUNTIME_USER="${USER:-$(id -un)}"
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-${REPO_ROOT}/.cache}"
PROJECT_VOLATILE_CACHE_BASE="${PROJECT_VOLATILE_CACHE_BASE:-/tmp/${SHORT_RUNTIME_USER}/fr}"
PROJECT_RUNTIME_SESSION_ID="${PROJECT_RUNTIME_SESSION_ID:-p$$}"
export PROJECT_CACHE_ROOT PROJECT_VOLATILE_CACHE_BASE PROJECT_RUNTIME_SESSION_ID
# shellcheck source=/dev/null
. "${REPO_ROOT}/scripts/use_project_cache.sh"
RESET_RUNTIME_COMPILE_CACHE="${RESET_RUNTIME_COMPILE_CACHE:-0}"
export RESET_RUNTIME_COMPILE_CACHE

DEFAULT_MODEL_CANDIDATES=(
    "/dpc/kuin0100/bohao/202509_InferenceModel/model/Qwen-2.5-32B"
    "${REPO_ROOT}/model/Qwen-2.5-32B"
    "${REPO_ROOT}/model/Qwen2.5-32B"
    "/dpc/kuin0100/bohao/202509_InferenceModel/model/Qwen3-235B-A22B-Instruct-2507"
    "${REPO_ROOT}/model/Qwen3-235B-A22B-Instruct-2507"
    "${REPO_ROOT}/model/Qwen2.5-7B-Instruct"
    "${REPO_ROOT}/model/Qwen3-8B"
    "${REPO_ROOT}/outputs/model_FT_merged/Qwen-2.5-32B"
    "${REPO_ROOT}/outputs/model_FT_merged/Qwen2.5-32B"
    "${REPO_ROOT}/outputs/model_FT_merged/Qwen2.5-7B-Instruct"
    "${REPO_ROOT}/outputs/model_FT_merged/Qwen3-8B"
)

if [[ -z "${PRETRAINED_MODEL:-}" ]]; then
    for candidate in "${DEFAULT_MODEL_CANDIDATES[@]}"; do
        if [[ -d "${candidate}" ]]; then
            PRETRAINED_MODEL="${candidate}"
            break
        fi
    done
fi

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/GRPO/telemath}"
DATA_TRAIN_PATH="${DATA_TRAIN_PATH:-${DATASET_DIR}/train.jsonl}"
DATA_TEST_PATH="${DATA_TEST_PATH:-${REPO_ROOT}/data/GRPO/telemath/test.jsonl}"
DATA_PROBE_PATH="${DATA_PROBE_PATH:-${DATA_TRAIN_PATH}}"

MODEL_BASENAME="$(basename "${PRETRAINED_MODEL:-unknown}")"
if [[ "${MODEL_BASENAME}" == *235B* || "${MODEL_BASENAME}" == *A22B* ]]; then
    DEFAULT_TENSOR_MODEL_PARALLEL_SIZE=8
    DEFAULT_ROLLOUT_N=4
    DEFAULT_GPU_MEMORY_UTILIZATION=0.85
    DEFAULT_MAX_NUM_SEQS=16
    DEFAULT_ENABLE_PREFIX_CACHING=False
    DEFAULT_BATCH_SIZE=2
    DEFAULT_EXPERIMENT_NAME="qwen3_235b_a22b_gsm8k_flowrl_probe_$(date +%m%d)"
elif [[ "${MODEL_BASENAME}" == *32B* || "${MODEL_BASENAME}" == *48B* || "${MODEL_BASENAME}" == *A3B* ]]; then
    DEFAULT_TENSOR_MODEL_PARALLEL_SIZE=8
    DEFAULT_ROLLOUT_N=8
    DEFAULT_GPU_MEMORY_UTILIZATION=0.82
    DEFAULT_MAX_NUM_SEQS=256
    DEFAULT_ENABLE_PREFIX_CACHING=True
    DEFAULT_BATCH_SIZE=32
    DEFAULT_EXPERIMENT_NAME="qwen2_5_32b_telemath_flowrl_probe_$(date +%m%d)"
else
    DEFAULT_TENSOR_MODEL_PARALLEL_SIZE=1
    DEFAULT_ROLLOUT_N=4
    DEFAULT_GPU_MEMORY_UTILIZATION=0.5
    DEFAULT_MAX_NUM_SEQS=64
    DEFAULT_ENABLE_PREFIX_CACHING=True
    DEFAULT_BATCH_SIZE=16
    DEFAULT_EXPERIMENT_NAME="gsm8k_flowrl_probe_$(date +%m%d)"
fi

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8000}"
TOTAL_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-${DEFAULT_TENSOR_MODEL_PARALLEL_SIZE}}"
ROLLOUT_N="${ROLLOUT_N:-${DEFAULT_ROLLOUT_N}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${DEFAULT_GPU_MEMORY_UTILIZATION}}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-${DEFAULT_MAX_NUM_SEQS}}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-True}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-${DEFAULT_ENABLE_PREFIX_CACHING}}"
ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
DTYPE="${DTYPE:-auto}"
PROBE_BATCH_SIZE_WAS_SET="${PROBE_BATCH_SIZE+x}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
PROBE_BACKEND="${PROBE_BACKEND:-auto}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
ROLL_OUT_DIR="${ROLL_OUT_DIR:-${REPO_ROOT}/outputs/eval/rollout/${EXPERIMENT_NAME}}"

CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-${SCRIPT_DIR}/custom_reward.py}"
CUSTOM_REWARD_NAME="${CUSTOM_REWARD_NAME:-my_math_reward_fn_deepmath_boxed}"
TOO_EASY_THRESHOLD="${TOO_EASY_THRESHOLD:-0.75}"
TOO_HARD_THRESHOLD="${TOO_HARD_THRESHOLD:--0.75}"
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-True}"
TRUNCATION="${TRUNCATION:-left}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
SEED="${SEED:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TEST_MAX_SAMPLES="${TEST_MAX_SAMPLES:-${MAX_SAMPLES}}"
HF_TOKENIZER_PATH="${HF_TOKENIZER_PATH:-}"
HF_DEVICE_MAP="${HF_DEVICE_MAP:-}"
HF_DEVICE="${HF_DEVICE:-cuda}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
RUN_TEST_PROBE="${RUN_TEST_PROBE:-0}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-${ROLL_OUT_DIR}/test}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a visible_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS_PER_NODE="${#visible_gpu_array[@]}"
if [[ "${N_GPUS_PER_NODE}" -lt 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES resolved to zero GPUs: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi

if (( TENSOR_MODEL_PARALLEL_SIZE < 1 )); then
    echo "TENSOR_MODEL_PARALLEL_SIZE must be >= 1, got: ${TENSOR_MODEL_PARALLEL_SIZE}" >&2
    exit 1
fi

if (( N_GPUS_PER_NODE % TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
    echo "n_gpus_per_node (${N_GPUS_PER_NODE}) must be divisible by tensor parallel size (${TENSOR_MODEL_PARALLEL_SIZE})" >&2
    exit 1
fi

if (( ROLLOUT_N < 1 )); then
    echo "ROLLOUT_N must be >= 1, got: ${ROLLOUT_N}" >&2
    exit 1
fi

MAX_PROMPTS_PER_BATCH=$((ROLLOUT_MAX_NUM_SEQS / ROLLOUT_N))
if (( MAX_PROMPTS_PER_BATCH < 1 )); then
    echo "ROLLOUT_MAX_NUM_SEQS (${ROLLOUT_MAX_NUM_SEQS}) must be >= ROLLOUT_N (${ROLLOUT_N})." >&2
    exit 1
fi

if (( PROBE_BATCH_SIZE > MAX_PROMPTS_PER_BATCH )); then
    if [[ -n "${PROBE_BATCH_SIZE_WAS_SET}" ]]; then
        echo "PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE} is too large for ROLLOUT_N=${ROLLOUT_N} and ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS}." >&2
        echo "Need PROBE_BATCH_SIZE <= ${MAX_PROMPTS_PER_BATCH} so one batch can fit into vLLM's max_num_seqs budget." >&2
        exit 1
    fi

    echo "Auto-adjusting PROBE_BATCH_SIZE from ${PROBE_BATCH_SIZE} to ${MAX_PROMPTS_PER_BATCH} for ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS} and ROLLOUT_N=${ROLLOUT_N}."
    PROBE_BATCH_SIZE="${MAX_PROMPTS_PER_BATCH}"
fi

if [[ -z "${PRETRAINED_MODEL:-}" ]]; then
    echo "PRETRAINED_MODEL is not set and no local default model was found under ${REPO_ROOT}." >&2
    exit 1
fi

if [[ ! -d "${PRETRAINED_MODEL}" ]]; then
    echo "Pretrained model directory not found: ${PRETRAINED_MODEL}" >&2
    exit 1
fi

for required_file in "${DATA_PROBE_PATH}" "${CUSTOM_REWARD_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ "${RUN_TEST_PROBE}" == "1" || "${RUN_TEST_PROBE}" == "true" || "${RUN_TEST_PROBE}" == "True" ]]; then
    if [[ ! -f "${DATA_TEST_PATH}" ]]; then
        echo "DATA_TEST_PATH is enabled for probing but file was not found: ${DATA_TEST_PATH}" >&2
        exit 1
    fi
fi

mkdir -p \
    "${ROLL_OUT_DIR}" \
    "${HF_HOME}" \
    "${HF_MODULES_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${RUNTIME_CACHE_ROOT}" \
    "${XDG_CACHE_HOME}" \
    "${VLLM_CACHE_ROOT}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" \
    "${OUTLINES_CACHE_DIR}"

if [[ "${RESET_RUNTIME_COMPILE_CACHE}" == "1" || "${RESET_RUNTIME_COMPILE_CACHE}" == "true" || "${RESET_RUNTIME_COMPILE_CACHE}" == "True" ]]; then
    echo "RESET_RUNTIME_COMPILE_CACHE=${RESET_RUNTIME_COMPILE_CACHE}; clearing stale vLLM runtime caches under ${RUNTIME_CACHE_ROOT}"
    rm -rf \
        "${VLLM_CACHE_ROOT}" \
        "${XDG_CACHE_HOME}/vllm"
    mkdir -p \
        "${VLLM_CACHE_ROOT}" \
        "${TORCHINDUCTOR_CACHE_DIR}" \
        "${TRITON_CACHE_DIR}" \
        "${XDG_CACHE_HOME}" \
        "${VLLM_CACHE_ROOT}/torch_compile_cache" \
        "${VLLM_CACHE_ROOT}/deep_gemm" \
        "${VLLM_CACHE_ROOT}/outlines"
fi

echo "Using PRETRAINED_MODEL=${PRETRAINED_MODEL}"
echo "Using DATA_PROBE_PATH=${DATA_PROBE_PATH}"
echo "Using DATA_TEST_PATH=${DATA_TEST_PATH}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (n_gpus_per_node=${N_GPUS_PER_NODE})"
echo "Using backend=${PROBE_BACKEND} rollout_n=${ROLLOUT_N} batch_size=${PROBE_BATCH_SIZE}"
echo "Using max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS:-default} gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "Using CACHE_ROOT=${CACHE_ROOT}"
echo "Using cache roots: XDG_CACHE_HOME=${XDG_CACHE_HOME} VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "Using RESET_RUNTIME_COMPILE_CACHE=${RESET_RUNTIME_COMPILE_CACHE}"
echo "Using thresholds: too_easy>=${TOO_EASY_THRESHOLD}, too_hard<=${TOO_HARD_THRESHOLD}"
echo "Writing probe outputs to ${ROLL_OUT_DIR}"
echo "RUN_TEST_PROBE=${RUN_TEST_PROBE}"

COMMON_ARGS=(
    python3
    "${SCRIPT_DIR}/rollout_probe.py"
    "--model" "${PRETRAINED_MODEL}"
    "--reward-path" "${CUSTOM_REWARD_PATH}"
    "--reward-name" "${CUSTOM_REWARD_NAME}"
    "--backend" "${PROBE_BACKEND}"
    "--tensor-parallel-size" "${TENSOR_MODEL_PARALLEL_SIZE}"
    "--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}"
    "--enable-chunked-prefill" "${ENABLE_CHUNKED_PREFILL}"
    "--enable-prefix-caching" "${ENABLE_PREFIX_CACHING}"
    "--enforce-eager" "${ENFORCE_EAGER}"
    "--load-format" "${LOAD_FORMAT}"
    "--dtype" "${DTYPE}"
    "--device" "${HF_DEVICE}"
    "--batch-size" "${PROBE_BATCH_SIZE}"
    "--rollout-n" "${ROLLOUT_N}"
    "--temperature" "${TEMPERATURE}"
    "--top-p" "${TOP_P}"
    "--top-k" "${TOP_K}"
    "--max-prompt-length" "${MAX_PROMPT_LENGTH}"
    "--max-response-length" "${MAX_RESPONSE_LENGTH}"
    "--max-model-len" "${TOTAL_TOKEN_LEN}"
    "--max-num-seqs" "${ROLLOUT_MAX_NUM_SEQS}"
    "--filter-overlong-prompts" "${FILTER_OVERLONG_PROMPTS}"
    "--truncation" "${TRUNCATION}"
    "--too-easy-threshold" "${TOO_EASY_THRESHOLD}"
    "--too-hard-threshold" "${TOO_HARD_THRESHOLD}"
)

if [[ -n "${ROLLOUT_MAX_BATCHED_TOKENS}" ]]; then
    COMMON_ARGS+=("--max-num-batched-tokens" "${ROLLOUT_MAX_BATCHED_TOKENS}")
fi

if [[ -n "${HF_TOKENIZER_PATH}" ]]; then
    COMMON_ARGS+=("--tokenizer" "${HF_TOKENIZER_PATH}")
fi

if [[ -n "${HF_DEVICE_MAP}" ]]; then
    COMMON_ARGS+=("--device-map" "${HF_DEVICE_MAP}")
fi

if [[ -n "${SEED}" ]]; then
    COMMON_ARGS+=("--seed" "${SEED}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" || "${TRUST_REMOTE_CODE}" == "true" || "${TRUST_REMOTE_CODE}" == "True" ]]; then
    COMMON_ARGS+=("--trust-remote-code")
fi

PASS_ARGS=("$@")

if [[ "${DRY_RUN}" == "1" ]]; then
    TRAIN_CMD=(
        "${COMMON_ARGS[@]}"
        "--data-path" "${DATA_PROBE_PATH}"
        "--output-dir" "${ROLL_OUT_DIR}"
    )
    if [[ -n "${MAX_SAMPLES}" ]]; then
        TRAIN_CMD+=("--max-samples" "${MAX_SAMPLES}")
    fi
    TRAIN_CMD+=("${PASS_ARGS[@]}")
    printf '%q ' "${TRAIN_CMD[@]}"
    printf '\n'

    if [[ "${RUN_TEST_PROBE}" == "1" || "${RUN_TEST_PROBE}" == "true" || "${RUN_TEST_PROBE}" == "True" ]]; then
        TEST_CMD=(
            "${COMMON_ARGS[@]}"
            "--data-path" "${DATA_TEST_PATH}"
            "--output-dir" "${TEST_OUTPUT_DIR}"
        )
        if [[ -n "${TEST_MAX_SAMPLES}" ]]; then
            TEST_CMD+=("--max-samples" "${TEST_MAX_SAMPLES}")
        fi
        TEST_CMD+=("${PASS_ARGS[@]}")
        printf '%q ' "${TEST_CMD[@]}"
        printf '\n'
    fi
    exit 0
fi

set -x
TRAIN_CMD=(
    "${COMMON_ARGS[@]}"
    "--data-path" "${DATA_PROBE_PATH}"
    "--output-dir" "${ROLL_OUT_DIR}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
    TRAIN_CMD+=("--max-samples" "${MAX_SAMPLES}")
fi
TRAIN_CMD+=("${PASS_ARGS[@]}")
"${TRAIN_CMD[@]}"

if [[ "${RUN_TEST_PROBE}" == "1" || "${RUN_TEST_PROBE}" == "true" || "${RUN_TEST_PROBE}" == "True" ]]; then
    TEST_CMD=(
        "${COMMON_ARGS[@]}"
        "--data-path" "${DATA_TEST_PATH}"
        "--output-dir" "${TEST_OUTPUT_DIR}"
    )
    if [[ -n "${TEST_MAX_SAMPLES}" ]]; then
        TEST_CMD+=("--max-samples" "${TEST_MAX_SAMPLES}")
    fi
    TEST_CMD+=("${PASS_ARGS[@]}")
    "${TEST_CMD[@]}"
fi
