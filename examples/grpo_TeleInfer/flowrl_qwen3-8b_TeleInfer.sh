#!/usr/bin/env bash
#
# Full-parameter FlowRL / GRPO training entrypoint for the local HPC layout.
#
# Typical usage:
#   PRETRAINED_MODEL=/dpc/kuin0100/bohao/202509_InferenceModel/model/Qwen-2.5-32B \
#   bash flowrl_qwen3-8b_TeleInfer.sh
#
# Dry run:
#   PRETRAINED_MODEL=/path/to/model DRY_RUN=1 bash flowrl_qwen3-8b_TeleInfer.sh
#
# Export after training:
#   cd /dpc/kuin0100/bohao/202509_InferenceModel/Inference/verl
#   python3 -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor --target_dir <ckpt>/actor/huggingface

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

# Respect externally provided GPU selection and default to the original 8-GPU layout.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
export CUDA_VISIBLE_DEVICES
export VLLM_ENABLE_SLEEP_MODE=1
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

DEFAULT_PRETRAINED_MODEL="/dpc/kuin0100/bohao/202509_InferenceModel/model/DeepSeek-R1-Distill-Qwen-32B"
DEFAULT_MODEL_CANDIDATES=(
    "/dpc/kuin0100/bohao/202509_InferenceModel/model/DeepSeek-R1-Distill-Qwen-32B"
    "${REPO_ROOT}/model/Qwen-2.5-32B"
    "${REPO_ROOT}/model/Qwen2.5-32B"
    "/dpc/kuin0100/bohao/202509_InferenceModel/model/Kimi-Linear-48B-A3B-Instruct"
    "${REPO_ROOT}/model/Kimi-Linear-48B-A3B-Instruct"
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
    if [[ -d "${DEFAULT_PRETRAINED_MODEL}" ]]; then
        PRETRAINED_MODEL="${DEFAULT_PRETRAINED_MODEL}"
    else
        for candidate in "${DEFAULT_MODEL_CANDIDATES[@]}"; do
            if [[ -d "${candidate}" ]]; then
                PRETRAINED_MODEL="${candidate}"
                break
            fi
        done
    fi
fi

DEFAULT_DATASET_DIR="/dpc/kuin0100/bohao/202509_InferenceModel/data/GRPO/telemath_param_augment_deepseek_v1"
DATASET_DIR="${DATASET_DIR:-${DEFAULT_DATASET_DIR}}"
DATA_TRAIN_PATH="${DATA_TRAIN_PATH:-${DATASET_DIR}/train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${DATASET_DIR}/test.parquet}"

slugify_identifier() {
    printf '%s' "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

MODEL_BASENAME="$(basename "${PRETRAINED_MODEL:-unknown}")"
MODEL_SLUG="$(slugify_identifier "${MODEL_BASENAME:-unknown}")"
DATASET_SLUG="$(slugify_identifier "$(basename "${DATASET_DIR}")")"
if [[ "${MODEL_BASENAME}" == *235B* || "${MODEL_BASENAME}" == *A22B* ]]; then
    DEFAULT_TENSOR_MODEL_PARALLEL_SIZE=8
    DEFAULT_TRAIN_BATCH_SIZE=128
    DEFAULT_PPO_MINI_BATCH_SIZE=32
    DEFAULT_ROLLOUT_N=4
    DEFAULT_GPU_MEMORY_UTILIZATION=0.85
    DEFAULT_ACTOR_PARAM_OFFLOAD=True
    DEFAULT_REF_PARAM_OFFLOAD=True
    DEFAULT_ACTOR_OPTIMIZER_OFFLOAD=True
    DEFAULT_USE_KL_LOSS=False
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=16
    DEFAULT_ENABLE_PREFIX_CACHING=False
elif [[ "${MODEL_BASENAME}" == *32B* || "${MODEL_BASENAME}" == *48B* || "${MODEL_BASENAME}" == *A3B* ]]; then
    DEFAULT_TENSOR_MODEL_PARALLEL_SIZE=8
    # Tuned for 8x H200 140G: substantially higher rollout and batch throughput
    # while keeping enough room for the hybrid actor/ref engines.
    DEFAULT_TRAIN_BATCH_SIZE=128
    DEFAULT_PPO_MINI_BATCH_SIZE=32
    DEFAULT_ROLLOUT_N=4
    DEFAULT_GPU_MEMORY_UTILIZATION=0.75
    DEFAULT_ACTOR_PARAM_OFFLOAD=False
    DEFAULT_REF_PARAM_OFFLOAD=False
    DEFAULT_ACTOR_OPTIMIZER_OFFLOAD=False
    DEFAULT_USE_KL_LOSS=True
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=32
    DEFAULT_ROLLOUT_MAX_BATCHED_TOKENS=32768
    DEFAULT_ENABLE_PREFIX_CACHING=False
else
    DEFAULT_TENSOR_MODEL_PARALLEL_SIZE=1
    DEFAULT_TRAIN_BATCH_SIZE=128
    DEFAULT_PPO_MINI_BATCH_SIZE=32
    DEFAULT_ROLLOUT_N=4
    DEFAULT_GPU_MEMORY_UTILIZATION=0.5
    DEFAULT_ACTOR_PARAM_OFFLOAD=False
    DEFAULT_REF_PARAM_OFFLOAD=False
    DEFAULT_ACTOR_OPTIMIZER_OFFLOAD=False
    DEFAULT_USE_KL_LOSS=True
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=64
    DEFAULT_ROLLOUT_MAX_BATCHED_TOKENS=65536
    DEFAULT_ENABLE_PREFIX_CACHING=False
fi

if [[ -z "${DEFAULT_ROLLOUT_MAX_BATCHED_TOKENS:-}" ]]; then
    DEFAULT_ROLLOUT_MAX_BATCHED_TOKENS=32768
fi

DEFAULT_EXPERIMENT_NAME="${MODEL_SLUG}_${DATASET_SLUG}_flowrl_$(date +%m%d)"
if [[ "${MODEL_BASENAME}" == *Kimi* || "${MODEL_BASENAME}" == *kimi* ]]; then
    DEFAULT_TRUST_REMOTE_CODE=True
else
    DEFAULT_TRUST_REMOTE_CODE=False
fi

N_NODES="${N_NODES:-1}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-${DEFAULT_TENSOR_MODEL_PARALLEL_SIZE}}"
# Checkpoints are saved by step count rather than wall-clock time.
# At roughly 17 minutes per step, SAVE_FREQ=21 is about one checkpoint every 6 hours.
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
PPO_MINI_BATCH_SIZE_WAS_SET="${PPO_MINI_BATCH_SIZE+x}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${DEFAULT_PPO_MINI_BATCH_SIZE}}"
ROLLOUT_N="${ROLLOUT_N:-${DEFAULT_ROLLOUT_N}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${DEFAULT_GPU_MEMORY_UTILIZATION}}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-${DEFAULT_ACTOR_PARAM_OFFLOAD}}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-${DEFAULT_REF_PARAM_OFFLOAD}}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-${DEFAULT_ACTOR_OPTIMIZER_OFFLOAD}}"
USE_KL_LOSS="${USE_KL_LOSS:-${DEFAULT_USE_KL_LOSS}}"
ROLLOUT_LOGPROBS_MODE="${ROLLOUT_LOGPROBS_MODE:-processed_logprobs}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-TeleReasoning_GRPO}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-${TRAINER_PROJECT_NAME}}"
WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/outputs/wandb}"

ROLL_OUT_DIR="${ROLL_OUT_DIR:-${REPO_ROOT}/outputs/eval/rollout/${EXPERIMENT_NAME}}"
VAL_DATA_DIR="${VAL_DATA_DIR:-${REPO_ROOT}/outputs/eval/validation/${EXPERIMENT_NAME}}"

CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-${SCRIPT_DIR}/custom_reward.py}"
CUSTOM_REWARD_NAME="${CUSTOM_REWARD_NAME:-my_math_reward_fn_deepmath_boxed}"
DRY_RUN="${DRY_RUN:-0}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console','wandb']}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-${DEFAULT_ROLLOUT_MAX_NUM_SEQS}}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-${DEFAULT_ROLLOUT_MAX_BATCHED_TOKENS}}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-True}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-${DEFAULT_ENABLE_PREFIX_CACHING}}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-${DEFAULT_TRUST_REMOTE_CODE}}"

IFS=',' read -r -a visible_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS_PER_NODE="${#visible_gpu_array[@]}"
if [[ "${N_GPUS_PER_NODE}" -lt 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES resolved to zero GPUs: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi

TOTAL_GPUS=$((N_GPUS_PER_NODE * N_NODES))

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

# verl normalizes actor PPO mini-batch size as:
#   ppo_mini_batch_size * rollout.n // world_size
# Keep it positive here so we fail early with a clear message instead of inside Ray actor init.
MIN_PPO_MINI_BATCH_SIZE=$(((TOTAL_GPUS + ROLLOUT_N - 1) / ROLLOUT_N))
if (( PPO_MINI_BATCH_SIZE < MIN_PPO_MINI_BATCH_SIZE )); then
    if [[ -n "${PPO_MINI_BATCH_SIZE_WAS_SET}" ]]; then
        echo "PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} is too small for TOTAL_GPUS=${TOTAL_GPUS} and ROLLOUT_N=${ROLLOUT_N}." >&2
        echo "Need PPO_MINI_BATCH_SIZE >= ${MIN_PPO_MINI_BATCH_SIZE} so verl's normalized actor mini-batch stays > 0." >&2
        exit 1
    fi

    echo "Auto-adjusting PPO_MINI_BATCH_SIZE from ${PPO_MINI_BATCH_SIZE} to ${MIN_PPO_MINI_BATCH_SIZE} for TOTAL_GPUS=${TOTAL_GPUS} and ROLLOUT_N=${ROLLOUT_N}."
    PPO_MINI_BATCH_SIZE="${MIN_PPO_MINI_BATCH_SIZE}"
fi

if (( TRAIN_BATCH_SIZE < PPO_MINI_BATCH_SIZE )); then
    echo "TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be >= PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE})." >&2
    exit 1
fi

if [[ -z "${PRETRAINED_MODEL:-}" ]]; then
    echo "PRETRAINED_MODEL is not set and no local default model was found under ${REPO_ROOT}." >&2
    echo "Set it explicitly, for example:" >&2
    echo "  PRETRAINED_MODEL=${REPO_ROOT}/model/Qwen-2.5-32B bash ${0##*/}" >&2
    exit 1
fi

if [[ ! -d "${PRETRAINED_MODEL}" ]]; then
    echo "Pretrained model directory not found: ${PRETRAINED_MODEL}" >&2
    exit 1
fi

for required_file in "${DATA_TRAIN_PATH}" "${VAL_DATA_PATH}" "${CUSTOM_REWARD_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

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
import sys
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

python3 - <<'PY' "${DATA_TRAIN_PATH}" "${VAL_DATA_PATH}"
import sys
try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    sys.exit(0)

train_path, val_path = sys.argv[1], sys.argv[2]
for label, path in (("train", train_path), ("val", val_path)):
    pf = pq.ParquetFile(path)
    num_rows = pf.metadata.num_rows
    if num_rows <= 0:
        raise SystemExit(
            f"{label} parquet is empty: {path}\n"
            "This FlowRL script requires a non-empty validation parquet.\n"
            "Quick unblock: point VAL_DATA_PATH to a non-empty parquet, or temporarily reuse DATA_TRAIN_PATH.\n"
            "Proper fix: rebuild the GRPO dataset with --train-ratio < 1.0 so grpo_test.parquet is non-empty."
        )
PY

mkdir -p "${ROLL_OUT_DIR}" "${VAL_DATA_DIR}" "${WANDB_DIR}"
mkdir -p "${HF_HOME}" "${HF_MODULES_CACHE}" "${TRANSFORMERS_CACHE}"
mkdir -p "${RUNTIME_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${OUTLINES_CACHE_DIR}"

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

echo "Using REPO_ROOT=${REPO_ROOT}"
echo "Using PRETRAINED_MODEL=${PRETRAINED_MODEL}"
echo "Using model profile for ${MODEL_BASENAME}"
echo "Using DATASET_DIR=${DATASET_DIR}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (n_gpus_per_node=${N_GPUS_PER_NODE})"
echo "Using W&B project=${WANDB_PROJECT} run=${WANDB_NAME}"
echo "Using CACHE_ROOT=${CACHE_ROOT}"
echo "Using RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT}"
echo "Using RAY_TMPDIR=${RAY_TMPDIR}"
echo "Using RESET_RUNTIME_COMPILE_CACHE=${RESET_RUNTIME_COMPILE_CACHE}"
echo "Using actor_param_offload=${ACTOR_PARAM_OFFLOAD} ref_param_offload=${REF_PARAM_OFFLOAD} actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
echo "Using trust_remote_code=${TRUST_REMOTE_CODE}"

TOTAL_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-${TOTAL_TOKEN_LEN}}"

if [[ "${MODEL_BASENAME}" == *235B* || "${MODEL_BASENAME}" == *A22B* ]] && [[ "${N_NODES}" == "1" ]] && [[ "${USE_KL_LOSS}" == "True" ]]; then
    echo "Warning: single-node 235B FlowRL with USE_KL_LOSS=True loads an extra reference policy and often exceeds host RAM." >&2
    echo "If you still hit Ray node-memory OOM, retry with USE_KL_LOSS=False." >&2
fi

echo "Using use_kl_loss=${USE_KL_LOSS} rollout_n=${ROLLOUT_N} rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN} rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"

CMD=(
    python3
    -m
    verl.trainer.main_ppo
    "algorithm.adv_estimator=grpo"
    "data.train_files=${DATA_TRAIN_PATH}"
    "data.val_files=${VAL_DATA_PATH}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.trust_remote_code=${TRUST_REMOTE_CODE}"
    "data.filter_overlong_prompts=True"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.truncation=left"
    "actor_rollout_ref.model.path=${PRETRAINED_MODEL}"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
    "actor_rollout_ref.actor.optim.weight_decay=0.1"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.temperature=1"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL}"
    "actor_rollout_ref.rollout.enable_prefix_caching=${ENABLE_PREFIX_CACHING}"
    "actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${TOTAL_TOKEN_LEN}"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${TOTAL_TOKEN_LEN}"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${TOTAL_TOKEN_LEN}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.model.trust_remote_code=${TRUST_REMOTE_CODE}"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=${ROLLOUT_LOGPROBS_MODE}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=False"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0"
    "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}"
    "custom_reward_function.path=${CUSTOM_REWARD_PATH}"
    "custom_reward_function.name=${CUSTOM_REWARD_NAME}"
    "algorithm.use_kl_in_reward=False"
    "trainer.critic_warmup=0"
    "trainer.logger=${TRAINER_LOGGERS}"
    "trainer.project_name=${TRAINER_PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=${N_NODES}"
    "trainer.resume_mode=auto"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.rollout_data_dir=${ROLL_OUT_DIR}"
    "trainer.validation_data_dir=${VAL_DATA_DIR}"
)

CMD+=("$@")

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

set -x
"${CMD[@]}"
