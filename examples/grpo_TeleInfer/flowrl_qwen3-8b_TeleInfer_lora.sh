#!/usr/bin/env bash
#
# Dry run:
#   DRY_RUN=1 bash flowrl_qwen3-8b_TeleInfer_lora.sh
#
# Continue training from an existing adapter:
#   LORA_ADAPTER_PATH=/path/to/lora_adapter bash flowrl_qwen3-8b_TeleInfer_lora.sh
#
# Export after training:
#   python3 -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor --target_dir <ckpt>/actor/huggingface
#   The exported adapter will be saved under <target_dir>/lora_adapter.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

# Respect externally provided GPU selection, and fall back to the original 8-GPU setup.
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

PRETRAINED_MODEL="${PRETRAINED_MODEL:-${REPO_ROOT}/model/Qwen2.5-7B-Instruct}"
# DATA_TRAIN_PATH="${DATA_TRAIN_PATH:-${REPO_ROOT}/data/datasets/gsm8k/train.parquet}"
# VAL_DATA_PATH="${VAL_DATA_PATH:-${REPO_ROOT}/data/datasets/gsm8k/test.parquet}"
DATA_TRAIN_PATH="${DATA_TRAIN_PATH:-${REPO_ROOT}/data/GRPO/telemath/train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${REPO_ROOT}/data/GRPO/telemath/test.parquet}"

N_NODES="${N_NODES:-1}"
# Default to TP=2 on 8 GPUs so async rollout only needs 4 replicas instead of 8.
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-2}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-True}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-False}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"

# LoRA defaults follow verl's current guidance for ~7B models:
# - rank >= 32 to avoid hurting convergence
# - target_modules=all-linear
# - learning rate roughly 10x higher than full fine-tuning
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
ACTOR_LR="${ACTOR_LR:-1e-5}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"

USE_SHM="${USE_SHM:-True}"
ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-True}"
USE_REWARD_LOOP="${USE_REWARD_LOOP:-False}"

TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-TeleReasoning_GRPO}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen2.5-7B-Instruct-telemock_lora_r${LORA_RANK}_$(date +%m%d)}"

ROLL_OUT_DIR="${ROLL_OUT_DIR:-${REPO_ROOT}/outputs/eval/rollout/${EXPERIMENT_NAME}}"
VAL_DATA_DIR="${VAL_DATA_DIR:-${REPO_ROOT}/outputs/eval/validation/${EXPERIMENT_NAME}}"

CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-${SCRIPT_DIR}/custom_reward.py}"
CUSTOM_REWARD_NAME="${CUSTOM_REWARD_NAME:-my_math_reward_fn_deepmath_boxed}"

DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a visible_gpu_array <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS_PER_NODE="${#visible_gpu_array[@]}"
if [[ "${N_GPUS_PER_NODE}" -lt 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES resolved to zero GPUs: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi

if (( LORA_RANK <= 0 )); then
    echo "LORA_RANK must be > 0 for this LoRA script, got: ${LORA_RANK}" >&2
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

for required_file in "${DATA_TRAIN_PATH}" "${VAL_DATA_PATH}" "${CUSTOM_REWARD_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -d "${PRETRAINED_MODEL}" ]]; then
    echo "Pretrained model directory not found: ${PRETRAINED_MODEL}" >&2
    exit 1
fi

if [[ -n "${LORA_ADAPTER_PATH}" && ! -d "${LORA_ADAPTER_PATH}" ]]; then
    echo "LORA_ADAPTER_PATH does not exist: ${LORA_ADAPTER_PATH}" >&2
    exit 1
fi

mkdir -p "${ROLL_OUT_DIR}" "${VAL_DATA_DIR}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (n_gpus_per_node=${N_GPUS_PER_NODE})"
echo "Launching LoRA GRPO with rank=${LORA_RANK}, alpha=${LORA_ALPHA}, target_modules=${LORA_TARGET_MODULES}, lr=${ACTOR_LR}"

TOTAL_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-${TOTAL_TOKEN_LEN}}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-${TOTAL_TOKEN_LEN}}"

echo "Low-memory rollout defaults: tp=${TENSOR_MODEL_PARALLEL_SIZE}, rollout_n=${ROLLOUT_N}, train_batch_size=${TRAIN_BATCH_SIZE}, max_model_len=${ROLLOUT_MAX_MODEL_LEN}, max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}, agent_workers=${AGENT_LOOP_NUM_WORKERS}, use_shm=${USE_SHM}"

CMD=(
    python3
    -m
    verl.trainer.main_ppo
    "algorithm.adv_estimator=grpo"
    "data.train_files=${DATA_TRAIN_PATH}"
    "data.val_files=${VAL_DATA_PATH}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
    "data.filter_overlong_prompts=True"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.truncation=left"
    "actor_rollout_ref.model.path=${PRETRAINED_MODEL}"
    "actor_rollout_ref.model.use_shm=${USE_SHM}"
    "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
    "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
    "actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
    "actor_rollout_ref.actor.optim.weight_decay=0.1"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${TOTAL_TOKEN_LEN}"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${TOTAL_TOKEN_LEN}"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${TOTAL_TOKEN_LEN}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=0.01"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL}"
    "actor_rollout_ref.rollout.enable_prefix_caching=${ENABLE_PREFIX_CACHING}"
    "actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE}"
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_NUM_WORKERS}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=processed_logprobs"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.load_format=safetensors"
    "actor_rollout_ref.rollout.layered_summon=${ROLLOUT_LAYERED_SUMMON}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=False"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0"
    "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.ref.fsdp_config.param_offload=False"
    "custom_reward_function.path=${CUSTOM_REWARD_PATH}"
    "custom_reward_function.name=${CUSTOM_REWARD_NAME}"
    "reward_model.use_reward_loop=${USE_REWARD_LOOP}"
    "algorithm.use_kl_in_reward=False"
    "trainer.critic_warmup=0"
    "trainer.logger=['console','wandb']"
    "trainer.project_name=${TRAINER_PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=${N_NODES}"
    "trainer.val_before_train=False"
    "trainer.resume_mode=auto"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.rollout_data_dir=${ROLL_OUT_DIR}"
    "trainer.validation_data_dir=${VAL_DATA_DIR}"
)

if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
    CMD+=("actor_rollout_ref.model.lora_adapter_path=${LORA_ADAPTER_PATH}")
fi

CMD+=("$@")

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

set -x
"${CMD[@]}"
