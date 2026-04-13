#!/usr/bin/env bash
#
# GRPO launcher inspired by the symbolic recipe provided by the user, adapted to
# this repository's verl.main_ppo entrypoint and the FlowRL project's cache/W&B setup.
#
# Typical usage:
#   module purge
#   module load intel_h200_gpu
#   module load miniconda/3
#   module load cuda/12.4
#   conda activate /dpc/kuin0100/conda_env/grpo_py311
#   bash flowrl_grpo_symbolic_8xH200.sh
#
# Dry run:
#   DRY_RUN=1 bash flowrl_grpo_symbolic_8xH200.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

slugify_identifier() {
    printf '%s' "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

build_list_literal() {
    python3 - "$@" <<'PY'
import sys

items = sys.argv[1:]
print("[" + ", ".join(repr(item) for item in items) + "]")
PY
}

parse_list_literal() {
    python3 - "$1" <<'PY'
import ast
import sys

value = sys.argv[1]
parsed = ast.literal_eval(value)
if isinstance(parsed, str):
    parsed = [parsed]
if not isinstance(parsed, list):
    raise SystemExit(f"Expected a Python list literal or string, got: {type(parsed)!r}")
for item in parsed:
    print(item)
PY
}

# Respect externally provided GPU selection and default to the 2-GPU symbolic recipe layout.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
export CUDA_VISIBLE_DEVICES

# Communication and vLLM defaults from the symbolic recipe, while keeping them overrideable.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_HARMONY="${VLLM_USE_HARMONY:-0}"
export VLLM_ENABLE_SLEEP_MODE="${VLLM_ENABLE_SLEEP_MODE:-1}"
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
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-${TIKTOKEN_CACHE_DIR}}"

DEFAULT_MODEL_CANDIDATES=(
    "/home/shenyl/hf/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "${REPO_ROOT}/model/DeepSeek-R1-Distill-Qwen-1.5B"
    "/dpc/kuin0100/bohao/202509_InferenceModel/model/DeepSeek-R1-Distill-Qwen-1.5B"
    "${REPO_ROOT}/model/Qwen2.5-7B-Instruct"
    "/dpc/kuin0100/bohao/202509_InferenceModel/model/Qwen2.5-7B-Instruct"
    "${REPO_ROOT}/model/Qwen3-8B"
    "/dpc/kuin0100/bohao/202509_InferenceModel/model/Qwen3-8B"
    "/dpc/kuin0100/bohao/202509_InferenceModel/model/DeepSeek-R1-Distill-Qwen-32B"
)

if [[ -z "${PRETRAINED_MODEL:-}" ]]; then
    for candidate in "${DEFAULT_MODEL_CANDIDATES[@]}"; do
        if [[ -d "${candidate}" ]]; then
            PRETRAINED_MODEL="${candidate}"
            break
        fi
    done
fi

TRAIN_FILE_CANDIDATES=(
    "/home/lihaoyu/data/train_4_merged.parquet"
    "${REPO_ROOT}/data/GRPO/telemath_param_augment_deepseek_v1/train.parquet"
)

VAL_FILE_CANDIDATES=(
    "${REPO_ROOT}/data/math500.parquet"
    "${REPO_ROOT}/data/aime24.parquet"
    "${REPO_ROOT}/data/aime25.parquet"
    "${REPO_ROOT}/data/GRPO/telemath_param_augment_deepseek_v1/test.parquet"
)

if [[ -z "${TRAIN_FILES:-}" ]]; then
    if [[ -n "${TRAIN_PATH:-}" ]]; then
        TRAIN_FILES="$(build_list_literal "${TRAIN_PATH}")"
    else
        for candidate in "${TRAIN_FILE_CANDIDATES[@]}"; do
            if [[ -f "${candidate}" ]]; then
                TRAIN_FILES="$(build_list_literal "${candidate}")"
                break
            fi
        done
    fi
fi

if [[ -z "${VAL_FILES:-}" ]]; then
    if [[ -n "${VAL_PATH:-}" ]]; then
        VAL_FILES="$(build_list_literal "${VAL_PATH}")"
    else
        val_defaults=()
        for candidate in "${VAL_FILE_CANDIDATES[@]}"; do
            if [[ -f "${candidate}" ]]; then
                val_defaults+=("${candidate}")
            fi
        done
        if (( ${#val_defaults[@]} > 0 )); then
            VAL_FILES="$(build_list_literal "${val_defaults[@]}")"
        fi
    fi
fi

if [[ -n "${TRAIN_FILES:-}" ]]; then
    mapfile -t TRAIN_FILE_ARRAY < <(parse_list_literal "${TRAIN_FILES}")
else
    TRAIN_FILE_ARRAY=()
fi

if [[ -n "${VAL_FILES:-}" ]]; then
    mapfile -t VAL_FILE_ARRAY < <(parse_list_literal "${VAL_FILES}")
else
    VAL_FILE_ARRAY=()
fi

if (( ${#TRAIN_FILE_ARRAY[@]} == 0 )); then
    echo "No training parquet resolved. Set TRAIN_PATH=/abs/path/train.parquet or TRAIN_FILES=\"['/abs/path/a.parquet', ...]\"." >&2
    exit 1
fi

if (( ${#VAL_FILE_ARRAY[@]} == 0 )); then
    echo "No validation parquet resolved. Set VAL_PATH=/abs/path/test.parquet or VAL_FILES=\"['/abs/path/a.parquet', ...]\"." >&2
    exit 1
fi

for required_path in "${TRAIN_FILE_ARRAY[@]}" "${VAL_FILE_ARRAY[@]}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Required parquet not found: ${required_path}" >&2
        exit 1
    fi
done

if [[ -z "${PRETRAINED_MODEL:-}" ]]; then
    echo "PRETRAINED_MODEL is not set and no default model candidate was found." >&2
    echo "Set PRETRAINED_MODEL=/abs/path/to/model before launching." >&2
    exit 1
fi

if [[ ! -d "${PRETRAINED_MODEL}" ]]; then
    echo "Pretrained model directory not found: ${PRETRAINED_MODEL}" >&2
    exit 1
fi

MODEL_BASENAME="$(basename "${PRETRAINED_MODEL}")"
MODEL_SLUG="$(slugify_identifier "${MODEL_BASENAME}")"
TRAIN_BASENAME="$(basename "${TRAIN_FILE_ARRAY[0]}")"
TRAIN_BASENAME="${TRAIN_BASENAME%.parquet}"
DATASET_SLUG="$(slugify_identifier "${TRAIN_BASENAME}")"

# Keep a light model-size profile only for auxiliary rollout knobs that are not
# pinned by the requested symbolic recipe defaults below.
MODEL_CAPACITY_PROFILE="balanced_h200"
if [[ "${MODEL_BASENAME}" == *235B* || "${MODEL_BASENAME}" == *A22B* || "${MODEL_BASENAME}" == *70B* || "${MODEL_BASENAME}" == *72B* ]]; then
    MODEL_CAPACITY_PROFILE="xlarge_h200"
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=16
    DEFAULT_ENABLE_PREFIX_CACHING=False
elif [[ "${MODEL_BASENAME}" == *32B* || "${MODEL_BASENAME}" == *34B* || "${MODEL_BASENAME}" == *48B* || "${MODEL_BASENAME}" == *A3B* ]]; then
    MODEL_CAPACITY_PROFILE="large_h200"
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=64
    DEFAULT_ENABLE_PREFIX_CACHING=False
elif [[ "${MODEL_BASENAME}" == *14B* || "${MODEL_BASENAME}" == *15B* || "${MODEL_BASENAME}" == *16B* ]]; then
    MODEL_CAPACITY_PROFILE="mid_h200"
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=96
    DEFAULT_ENABLE_PREFIX_CACHING=True
else
    MODEL_CAPACITY_PROFILE="small_h200"
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=128
    DEFAULT_ENABLE_PREFIX_CACHING=True
fi

N_NODES="${N_NODES:-1}"
# Pin the train/test defaults to the requested symbolic recipe while keeping
# dataset/model/project/W&B resolution from this launcher intact.
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-True}"
TRUNCATION="${TRUNCATION:-error}"
DATA_SHUFFLE="${DATA_SHUFFLE:-False}"
USE_REWARD_LOOP="${USE_REWARD_LOOP:-False}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-TeleReasoning_GRPO}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-${TRAINER_PROJECT_NAME}}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console','wandb']}"
CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-${SCRIPT_DIR}/custom_reward.py}"
CUSTOM_REWARD_NAME="${CUSTOM_REWARD_NAME:-my_math_reward_fn_deepmath_boxed}"

ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
USE_KL_LOSS="${USE_KL_LOSS:-True}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-True}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
ACTOR_LR="${ACTOR_LR:-1e-6}"

ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_M="${ROLLOUT_M:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-32768}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-${DEFAULT_ROLLOUT_MAX_NUM_SEQS}}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-${DEFAULT_ENABLE_PREFIX_CACHING}}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-True}"
ROLLOUT_LOGPROBS_MODE="${ROLLOUT_LOGPROBS_MODE:-processed_logprobs}"

VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TOP_K="${VAL_TOP_K:--1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-True}"
VAL_N="${VAL_N:-16}"

LOGPROB_MAX_TOKEN_LEN_PER_GPU="${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-32768}"
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
RUN_TAG="${RUN_TAG:-}"
DRY_RUN="${DRY_RUN:-0}"
STEP_TIME_MINUTES_ESTIMATE="${STEP_TIME_MINUTES_ESTIMATE:-}"

RUN_TAG_SLUG=""
if [[ -n "${RUN_TAG}" ]]; then
    RUN_TAG_SLUG="$(slugify_identifier "${RUN_TAG}")"
fi

DEFAULT_EXPERIMENT_NAME="symbolic_like_${MODEL_SLUG}_${DATASET_SLUG}_n${ROLLOUT_N}_m${ROLLOUT_M}_resp${MAX_RESPONSE_LENGTH}_bs${TRAIN_BATCH_SIZE}_$(date +%m%d_%H%M%S)"
if [[ -n "${RUN_TAG_SLUG}" ]]; then
    DEFAULT_EXPERIMENT_NAME="${DEFAULT_EXPERIMENT_NAME}_${RUN_TAG_SLUG}"
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/outputs/wandb}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/outputs/grpo/checkpoints/${TRAINER_PROJECT_NAME}/${EXPERIMENT_NAME}}"
ROLL_OUT_DIR="${ROLL_OUT_DIR:-${REPO_ROOT}/outputs/eval/rollout/${EXPERIMENT_NAME}}"
VAL_DATA_DIR="${VAL_DATA_DIR:-${REPO_ROOT}/outputs/eval/validation/${EXPERIMENT_NAME}}"
RUN_MANIFEST_PATH="${RUN_MANIFEST_PATH:-${CHECKPOINT_DIR}/launch_config.env}"

IFS=',' read -r -a visible_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS_PER_NODE="${#visible_gpu_array[@]}"
if [[ "${N_GPUS_PER_NODE}" -lt 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES resolved to zero GPUs: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi

TOTAL_GPUS=$((N_GPUS_PER_NODE * N_NODES))

if (( N_GPUS_PER_NODE % TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
    echo "n_gpus_per_node (${N_GPUS_PER_NODE}) must be divisible by tensor parallel size (${TENSOR_MODEL_PARALLEL_SIZE})" >&2
    exit 1
fi

if (( TRAIN_BATCH_SIZE < PPO_MINI_BATCH_SIZE )); then
    echo "TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be >= PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE})." >&2
    exit 1
fi

if (( ROLLOUT_N < 1 )); then
    echo "ROLLOUT_N must be >= 1, got: ${ROLLOUT_N}" >&2
    exit 1
fi

if [[ ! -f "${CUSTOM_REWARD_PATH}" ]]; then
    echo "Custom reward file not found: ${CUSTOM_REWARD_PATH}" >&2
    exit 1
fi

if ! python3 -c "import wandb" >/dev/null 2>&1; then
    echo "wandb is required but could not be imported in the current environment." >&2
    echo "Install it into the active conda env before launching training." >&2
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" && -n "${wandb_api_key:-}" ]]; then
    WANDB_API_KEY="${wandb_api_key}"
    export WANDB_API_KEY
fi

if [[ -n "${WANDB_API_KEY_FILE}" && -z "${WANDB_API_KEY:-}" ]]; then
    if [[ ! -f "${WANDB_API_KEY_FILE}" ]]; then
        echo "WANDB_API_KEY_FILE not found: ${WANDB_API_KEY_FILE}" >&2
        exit 1
    fi
    WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_API_KEY_FILE}")"
    export WANDB_API_KEY
fi

export WANDB_MODE WANDB_BASE_URL WANDB_PROJECT WANDB_NAME WANDB_DIR
if [[ -n "${WANDB_ENTITY}" ]]; then
    export WANDB_ENTITY
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    if ! python3 - <<'PY'
import netrc
import os
from urllib.parse import urlparse

hosts = []
base_url = os.environ.get("WANDB_BASE_URL", "").strip()
if base_url:
    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    if parsed.netloc:
        hosts.append(parsed.netloc)
hosts.extend(["api.wandb.ai", "wandb.ai"])

try:
    auth = netrc.netrc()
except (FileNotFoundError, netrc.NetrcParseError):
    raise SystemExit(1)

for host in hosts:
    if auth.authenticators(host):
        raise SystemExit(0)

raise SystemExit(1)
PY
    then
        echo "wandb login is not configured." >&2
        echo "Set WANDB_API_KEY=... or WANDB_API_KEY_FILE=/path/to/wandb_api_key before running." >&2
        exit 1
    fi
fi

mkdir -p "${CHECKPOINT_DIR}" "${ROLL_OUT_DIR}" "${VAL_DATA_DIR}" "${WANDB_DIR}"
mkdir -p "${RUNTIME_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${OUTLINES_CACHE_DIR}"

if [[ "${RESET_RUNTIME_COMPILE_CACHE}" == "1" || "${RESET_RUNTIME_COMPILE_CACHE}" == "true" || "${RESET_RUNTIME_COMPILE_CACHE}" == "True" ]]; then
    echo "RESET_RUNTIME_COMPILE_CACHE=${RESET_RUNTIME_COMPILE_CACHE}; clearing stale vLLM runtime caches under ${RUNTIME_CACHE_ROOT}"
    rm -rf \
        "${VLLM_RUNTIME_CACHE_ROOT}" \
        "${VLLM_TORCH_COMPILE_CACHE_DIR}" \
        "${TORCHINDUCTOR_CACHE_DIR}" \
        "${TRITON_CACHE_DIR}" \
        "${OUTLINES_CACHE_DIR}" \
        "${CUDA_CACHE_PATH}" \
        "${RAY_TMPDIR}" \
        "${TMPDIR}"
    mkdir -p \
        "${RUNTIME_CACHE_ROOT}" \
        "${VLLM_RUNTIME_CACHE_ROOT}" \
        "${VLLM_TORCH_COMPILE_CACHE_DIR}" \
        "${TORCHINDUCTOR_CACHE_DIR}" \
        "${TRITON_CACHE_DIR}" \
        "${OUTLINES_CACHE_DIR}" \
        "${CUDA_CACHE_PATH}" \
        "${RAY_TMPDIR}" \
        "${TMPDIR}"
fi

TRAIN_ROWS="$(python3 - "${TRAIN_FILE_ARRAY[@]}" <<'PY'
import sys
import pyarrow.parquet as pq

total_rows = 0
for path in sys.argv[1:]:
    total_rows += pq.ParquetFile(path).metadata.num_rows
print(total_rows)
PY
)"

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    ESTIMATED_TOTAL_STEPS="${TOTAL_TRAINING_STEPS}"
    STEPS_PER_EPOCH=""
else
    STEPS_PER_EPOCH=$(((TRAIN_ROWS + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE))
    ESTIMATED_TOTAL_STEPS=$((STEPS_PER_EPOCH * TOTAL_EPOCHS))
fi

ESTIMATED_RUNTIME_HOURS=""
if [[ -n "${STEP_TIME_MINUTES_ESTIMATE}" ]]; then
    ESTIMATED_RUNTIME_HOURS="$(python3 - <<'PY' "${ESTIMATED_TOTAL_STEPS}" "${STEP_TIME_MINUTES_ESTIMATE}"
import sys

steps = float(sys.argv[1])
minutes_per_step = float(sys.argv[2])
print(f"{steps * minutes_per_step / 60:.2f}")
PY
)"
fi

echo "Using REPO_ROOT=${REPO_ROOT}"
echo "Using PRETRAINED_MODEL=${PRETRAINED_MODEL}"
echo "Using MODEL_CAPACITY_PROFILE=${MODEL_CAPACITY_PROFILE}"
echo "Using TRAIN_FILES=${TRAIN_FILES}"
echo "Using VAL_FILES=${VAL_FILES}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (n_gpus_per_node=${N_GPUS_PER_NODE})"
echo "Using W&B project=${WANDB_PROJECT} run=${WANDB_NAME}"
echo "Using EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "Using CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "Using ROLL_OUT_DIR=${ROLL_OUT_DIR}"
echo "Using VAL_DATA_DIR=${VAL_DATA_DIR}"
echo "Using CACHE_ROOT=${CACHE_ROOT}"
echo "Using RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT}"
echo "Using RAY_TMPDIR=${RAY_TMPDIR}"
echo "Using reward_loop=${USE_REWARD_LOOP} rollout_n=${ROLLOUT_N} rollout_m(metadata_only)=${ROLLOUT_M}"
echo "Using train_rows=${TRAIN_ROWS} total_epochs=${TOTAL_EPOCHS} estimated_total_steps=${ESTIMATED_TOTAL_STEPS}"
if [[ -n "${STEPS_PER_EPOCH}" ]]; then
    echo "Using estimated_steps_per_epoch=${STEPS_PER_EPOCH}"
fi
if [[ -n "${ESTIMATED_RUNTIME_HOURS}" ]]; then
    echo "Estimated wall-clock runtime=${ESTIMATED_RUNTIME_HOURS}h (STEP_TIME_MINUTES_ESTIMATE=${STEP_TIME_MINUTES_ESTIMATE})"
fi
if [[ -n "${ROLLOUT_M}" ]]; then
    echo "Note: current verl config does not expose actor_rollout_ref.rollout.m; this value is tracked in metadata only." >&2
fi

{
    printf 'LAUNCH_TIME=%s\n' "$(date -Iseconds)"
    printf 'EXPERIMENT_NAME=%s\n' "${EXPERIMENT_NAME}"
    printf 'WANDB_PROJECT=%s\n' "${WANDB_PROJECT}"
    printf 'WANDB_NAME=%s\n' "${WANDB_NAME}"
    printf 'PRETRAINED_MODEL=%s\n' "${PRETRAINED_MODEL}"
    printf 'MODEL_CAPACITY_PROFILE=%s\n' "${MODEL_CAPACITY_PROFILE}"
    printf 'TRAIN_FILES=%s\n' "${TRAIN_FILES}"
    printf 'VAL_FILES=%s\n' "${VAL_FILES}"
    printf 'TRAIN_ROWS=%s\n' "${TRAIN_ROWS}"
    printf 'TOTAL_EPOCHS=%s\n' "${TOTAL_EPOCHS}"
    printf 'ESTIMATED_TOTAL_STEPS=%s\n' "${ESTIMATED_TOTAL_STEPS}"
    printf 'STEP_TIME_MINUTES_ESTIMATE=%s\n' "${STEP_TIME_MINUTES_ESTIMATE}"
    printf 'ESTIMATED_RUNTIME_HOURS=%s\n' "${ESTIMATED_RUNTIME_HOURS}"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
    printf 'TENSOR_MODEL_PARALLEL_SIZE=%s\n' "${TENSOR_MODEL_PARALLEL_SIZE}"
    printf 'MAX_PROMPT_LENGTH=%s\n' "${MAX_PROMPT_LENGTH}"
    printf 'MAX_RESPONSE_LENGTH=%s\n' "${MAX_RESPONSE_LENGTH}"
    printf 'TRAIN_BATCH_SIZE=%s\n' "${TRAIN_BATCH_SIZE}"
    printf 'PPO_MINI_BATCH_SIZE=%s\n' "${PPO_MINI_BATCH_SIZE}"
    printf 'ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=%s\n' "${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}"
    printf 'ROLLOUT_N=%s\n' "${ROLLOUT_N}"
    printf 'ROLLOUT_M=%s\n' "${ROLLOUT_M}"
    printf 'ROLLOUT_MAX_BATCHED_TOKENS=%s\n' "${ROLLOUT_MAX_BATCHED_TOKENS}"
    printf 'ROLLOUT_MAX_NUM_SEQS=%s\n' "${ROLLOUT_MAX_NUM_SEQS}"
    printf 'GPU_MEMORY_UTILIZATION=%s\n' "${GPU_MEMORY_UTILIZATION}"
    printf 'USE_KL_LOSS=%s\n' "${USE_KL_LOSS}"
    printf 'KL_LOSS_COEF=%s\n' "${KL_LOSS_COEF}"
    printf 'SAVE_FREQ=%s\n' "${SAVE_FREQ}"
    printf 'TEST_FREQ=%s\n' "${TEST_FREQ}"
    printf 'TOTAL_TRAINING_STEPS=%s\n' "${TOTAL_TRAINING_STEPS}"
    printf 'CHECKPOINT_DIR=%s\n' "${CHECKPOINT_DIR}"
    printf 'ROLL_OUT_DIR=%s\n' "${ROLL_OUT_DIR}"
    printf 'VAL_DATA_DIR=%s\n' "${VAL_DATA_DIR}"
} > "${RUN_MANIFEST_PATH}"
echo "Wrote launch manifest to ${RUN_MANIFEST_PATH}"

CMD=(
    python3
    -m
    verl.trainer.main_ppo
    "algorithm.adv_estimator=grpo"
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS}"
    "data.truncation=${TRUNCATION}"
    "data.shuffle=${DATA_SHUFFLE}"
    "reward_model.use_reward_loop=${USE_REWARD_LOOP}"
    "trainer.critic_warmup=0"
    "trainer.logger=${TRAINER_LOGGERS}"
    "trainer.project_name=${TRAINER_PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.default_local_dir=${CHECKPOINT_DIR}"
    "trainer.rollout_data_dir=${ROLL_OUT_DIR}"
    "trainer.validation_data_dir=${VAL_DATA_DIR}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=${N_NODES}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "algorithm.use_kl_in_reward=False"
    "actor_rollout_ref.model.path=${PRETRAINED_MODEL}"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.model.trust_remote_code=False"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_USE_DYNAMIC_BSZ}"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.name=${ROLLOUT_NAME}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}"
    "actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}"
    "actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL}"
    "actor_rollout_ref.rollout.enable_prefix_caching=${ENABLE_PREFIX_CACHING}"
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKEN_LEN_PER_GPU}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=${ROLLOUT_LOGPROBS_MODE}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}"
    "custom_reward_function.path=${CUSTOM_REWARD_PATH}"
    "custom_reward_function.name=${CUSTOM_REWARD_NAME}"
)

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    CMD+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

CMD+=("$@")

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

set -x
"${CMD[@]}"
