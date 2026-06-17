#!/usr/bin/env bash
# Qwen3.5-27B + FSDP2 + vLLM 0.18 + plain GRPO on GRPO_mixed_v3/stage1.4.
#
# Adapted from:
#   - examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh : Qwen3.5-27B FSDP2 + vLLM
#     knobs (GEN_TP=4, offload_policy, ulysses, chunked prefill,
#     entropy_from_logits_with_chunking, update_weights_bucket_megabytes=6144).
#   - myverl0.7.0_telelogReward/recipe/grpo_symbolic/run_stage_mixed_v3.sh :
#     dataset layout + telelogs_symbolic reward dispatch.
#
# Reward wiring: verl 0.8 doesn't ship telelogs_symbolic. We use verl's
# BatchRewardManager + custom_reward_function pointing at
# examples/grpo_TeleInfer/telelogs_symbolic_reward.py, which is a port of
# myverl0.7.0_telelogReward/recipe/grpo_symbolic/dapo_reward_entry.py with
# a local TeleMath fallback bound.
#
# dependency: GPU vllm==0.18.0, transformers@<cc7ab9be>
#
# Usage:
#   bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh
#   STAGE_NAME=stage1.4 EXPERIMENT_NAME=grpo_v3 bash examples/grpo_TeleInfer/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

experiment_name=${EXPERIMENT_NAME:-grpo_mixed_v3_stage1_4_qwen35_27b_v2_6_1984}
LOG_DIR="${VERL_ROOT}/${experiment_name}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/launch_$(date +%Y%m%d_%H%M%S)_pid$$.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[INFO] Full output logged to: ${LOG_FILE}"
echo "[INFO] Tail with: tail -f ${LOG_FILE}"

set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

########################### python env resolution ###########################
resolve_python_bin() {
  local candidate
  local candidates=()

  # Priority: explicit PYTHON_BIN -> activated conda env -> qwen35 default -> grpo_py311 -> python3
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("$PYTHON_BIN")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi
  candidates+=("/dpc/kuin0100/conda_env/qwen35/bin/python")
  candidates+=("/dpc/kuin0100/conda_env/grpo_py311/bin/python")
  candidates+=("python3")

  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import ray" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  echo "[ERROR] Could not find a Python with ray installed. Set PYTHON_BIN=/dpc/kuin0100/conda_env/qwen35/bin/python" >&2
  return 1
}

PYTHON_BIN="$(resolve_python_bin)" || exit 1
export PYTHON_BIN

########################### device / parallelism ###########################
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
n_devices_per_node=${NDEVICES_PER_NODE:-8}
fsdp_size=${FSDP_SIZE:-8}
sp_size=${SP_SIZE:-1}
gen_tp=${GEN_TP:-4}
rollout_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}

case "${DEVICE}" in
    gpu) ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        n_devices_per_node=16
        fsdp_size=16
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### data / model paths ###########################
DATA_ROOT=${DATA_ROOT:-/dpc/kuin0100/bohao/202509_InferenceModel/data/GRPO_mixed_v3}
STAGE_NAME=${STAGE_NAME:-stage1.4}
STAGE_DIR=${STAGE_DIR:-$DATA_ROOT/$STAGE_NAME}
VAL_ROOT=${VAL_ROOT:-$DATA_ROOT/val}
SFT_CKPT_DEFAULT=/dpc/kuin0100/bohao/202509_InferenceModel/outputs/model_FT_merged/Qwen3.5-27B-TelecomInstruct_v2.6_qlora_stage3_checkpoint-1984
TELELOGS_TEST=/dpc/kuin0100/bohao/202509_InferenceModel/data/GRPO/telelogs/test.parquet
TELEMATH_TEST=/dpc/kuin0100/bohao/202509_InferenceModel/data/GRPO/telemath/train.parquet
THREEGPP_TEST=$VAL_ROOT/3gpp.parquet
TELETABLE_TEST=$VAL_ROOT/teletable.parquet
TELEQNA_TEST=$VAL_ROOT/teleqna.parquet
ORANBENCH_TEST=$VAL_ROOT/oranbench.parquet
SRSRAN_TEST=$VAL_ROOT/srsran.parquet

is_false_value() {
  case "${1:-}" in
    false|False|FALSE|0|no|No|NO) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -n "${TRAIN_FILES:-}" ]]; then
  train_files="${TRAIN_FILES}"
  train_file_checks=()
elif is_false_value "${INCLUDE_MATHLOGS:-auto}" || [[ "${STAGE_NAME}" == *"wo_mathlogs"* ]]; then
  train_file_checks=("$STAGE_DIR/3gpp.parquet" "$STAGE_DIR/teleqna.parquet" "$STAGE_DIR/oranbench.parquet" "$STAGE_DIR/srsran.parquet" "$STAGE_DIR/teletable.parquet")
  train_files="['${train_file_checks[0]}','${train_file_checks[1]}','${train_file_checks[2]}','${train_file_checks[3]}','${train_file_checks[4]}']"
else
  train_file_checks=("$STAGE_DIR/telemath.parquet" "$STAGE_DIR/telelogs.parquet" "$STAGE_DIR/3gpp.parquet" "$STAGE_DIR/teleqna.parquet" "$STAGE_DIR/oranbench.parquet" "$STAGE_DIR/srsran.parquet" "$STAGE_DIR/teletable.parquet")
  train_files="['${train_file_checks[0]}','${train_file_checks[1]}','${train_file_checks[2]}','${train_file_checks[3]}','${train_file_checks[4]}','${train_file_checks[5]}','${train_file_checks[6]}']"
fi

if [[ -n "${TEST_FILES:-}" ]]; then
  test_files="${TEST_FILES}"
  test_file_checks=()
elif is_false_value "${INCLUDE_MATHLOGS:-auto}" || [[ "${STAGE_NAME}" == *"wo_mathlogs"* ]]; then
  test_file_checks=("$THREEGPP_TEST" "$TELETABLE_TEST" "$TELEQNA_TEST" "$ORANBENCH_TEST" "$SRSRAN_TEST")
  test_files="['${test_file_checks[0]}','${test_file_checks[1]}','${test_file_checks[2]}','${test_file_checks[3]}','${test_file_checks[4]}']"
else
  test_file_checks=("$TELELOGS_TEST" "$TELEMATH_TEST" "$THREEGPP_TEST" "$TELETABLE_TEST" "$TELEQNA_TEST" "$ORANBENCH_TEST" "$SRSRAN_TEST")
  test_files="['${test_file_checks[0]}','${test_file_checks[1]}','${test_file_checks[2]}','${test_file_checks[3]}','${test_file_checks[4]}','${test_file_checks[5]}','${test_file_checks[6]}']"
fi

init_model=${INIT_MODEL:-$SFT_CKPT_DEFAULT}
total_epochs=${TOTAL_EPOCHS:-3}
learning_rate=${LR:-1e-6}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-5}
max_ckpt_to_keep=${MAX_CKPT:-4}
train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}

########################### custom-reward wiring ###########################
# verl 0.8 uses experimental reward_loop managers (naive/dapo/...) which call
# compute_score PER-SAMPLE (not batched). Our shim exposes both:
#   compute_score         -> per-sample dispatcher (for naive/dapo managers)
#   compute_score_batched -> batched (for the legacy BatchRewardManager only)
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-${SCRIPT_DIR}/telelogs_symbolic_reward.py}
CUSTOM_REWARD_NAME=${CUSTOM_REWARD_NAME:-compute_score}

########################### 3gpp reward mode ###########################
# Selects which compute_3gpp_score implementation to use. See
#   logs/qwen3_5_27b_telecominstruct_v2_6_qlora_stage3_checkpoint_1984-full/
#   three_gpp_failure_analysis.md  Part B §16 for the design rationale.
#
#   v1 : legacy binary exact-match reward — reproduces the run that produced
#        ckpt_1984 GRPO. Use this to roll back to the prior behaviour.
#   v2 : multi-component reward (default) — adds confusion-aware partial
#        credit, format-stability bonus, TS-citation bonus, and
#        author-mention penalty. Recommended starting point for the next run.
#
# Quick rollback:   THREEGPP_REWARD_MODE=v1 bash run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh
#
# Component weights can be ablated without code edits via these env vars
# (all read at reward-evaluation time, defaults shown):
#   THREEGPP_R2_NEIGHBOUR_CREDIT=0.30   # partial credit for neighbour-WG misses
#   THREEGPP_R3_FORMAT_BONUS=0.10       # +bonus for ≤200-char clean JSON output
#   THREEGPP_R3_CITATION_BONUS=0.10     # +bonus when TS-series cite matches pred (only if correct)
#   THREEGPP_R3_LONG_PENALTY=0.05       # -penalty for >8000-char total output
#   THREEGPP_AUTHOR_PENALTY=0.30        # -penalty when wrong pred == most-mentioned WG in rationale
export THREEGPP_REWARD_MODE=${THREEGPP_REWARD_MODE:-v2}
echo "[INFO] THREEGPP_REWARD_MODE=${THREEGPP_REWARD_MODE}"

########################### MCQ reward mode ###########################
# Selects which compute_mcq_score implementation to use for the MCQ data
# sources: oranbench / srsran / srsbench / teleqna / teletable.
#
# Default: "binary" (unset) — legacy 0/1 letter-match only. This is the same
# behaviour that produced the ckpt_1984 GRPO run analysed in
# failure_analysis.md §12.2.4 (76-92 % of rollouts had zero gradient).
#
# Set to "v2" to activate auxiliary reward components from
# failure_analysis.md §12.5 GRPO-A:
#     r1 = letter-match correctness (the legacy binary signal — dominant)
#     r2 = format-stability bonus      ([0, 0.3])
#     r3 = anti-hallucination penalty  ([-0.3, 0]) — fires on WRONG answers
#                                                    that contain fabricated
#                                                    O-RAN / RFC / 3GPP TS /
#                                                    srsran-path / Section-x.y
#                                                    citations.
# `acc` stays binary in both modes so validation metrics remain comparable.
#
# Quick rollback: leave the export commented out, or set MCQ_REWARD_MODE=binary.
# Smoke test:    python examples/grpo_TeleInfer/test_mcq_reward_v2.py
#
# export MCQ_REWARD_MODE=v2
#
# Set to "v3_teleqna" to additionally activate the teleqna-specific
# fabrication-template (r4) and hedging-density (r5) penalties on top of v2,
# per teleqna_failure_analysis.md §8. v3_teleqna only changes behaviour for
# samples with data_source == "teleqna"; other MCQ sources keep v2 behaviour.
#
# Recommended A/B path: use the sibling launcher
#   run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_teleqna_v3.sh
# which simply exports MCQ_REWARD_MODE=v3_teleqna and execs this base script.
# Smoke test:    python examples/grpo_TeleInfer/test_mcq_reward_v3.py
#
# export MCQ_REWARD_MODE=v3_teleqna
echo "[INFO] MCQ_REWARD_MODE=${MCQ_REWARD_MODE:-binary (default)}"

########################### TeleMath reward mode ###########################
# Selects which compute_telemath_score implementation to use. See
#   logs/.../telemath_failure_analysis.md Part II §17 for the design rationale.
#
#   binary : legacy binary math_equal reward — reproduces the run that
#            produced ckpt_1984 GRPO. Use this to roll back.
#   v2     : multi-component reward — adds format-stability bonus,
#            anti-pattern penalty for the SFT-identified regressions
#            (Friis NF double-count, M/M/1-on-deterministic, voltage
#            convention, ALOHA variant), and unit-canonicalised partial
#            credit. Recommended starting point.
#
# Quick rollback:   TELEMATH_REWARD_MODE=binary bash run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh
# Activate v2:      TELEMATH_REWARD_MODE=v2     bash run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh
# Smoke test:       python examples/grpo_TeleInfer/test_telemath_reward_v2.py
#
# Component weights are env-overridable (defaults shown):
#   TELEMATH_R2_BOXED_BONUS=0.10
#   TELEMATH_R2_LENGTH_BONUS=0.05
#   TELEMATH_R2_NOREP_BONUS=0.05
#   TELEMATH_R3_NF_DOUBLE_PENALTY=-0.10
#   TELEMATH_R3_MM1_DET_PENALTY=-0.10
#   TELEMATH_R3_VOLTAGE_PENALTY=-0.10
#   TELEMATH_R3_ALOHA_PENALTY=-0.10
#   TELEMATH_R3_PENALTY_CAP=-0.30
#   TELEMATH_UNIT_CREDIT=0.5
#   TELEMATH_R2_LEN_MIN=500
#   TELEMATH_R2_LEN_MAX=10000
#   TELEMATH_R2_REPETITION_NGRAM=30
#   TELEMATH_R2_REPETITION_MAX_HITS=4
#
# Leave unset for the current run's behaviour (binary).
# export TELEMATH_REWARD_MODE=v2
echo "[INFO] TELEMATH_REWARD_MODE=${TELEMATH_REWARD_MODE:-binary (default)}"

########################### CUDA toolkit for JIT ###########################
# flashinfer GDN kernel needs >=12.5 for Qwen3.5 hybrid attention.
# Respect CUDA_HOME already set by `module load cuda/13.0`; only fall back
# to the hard-coded path if neither CUDA_TOOLKIT_HOME nor CUDA_HOME is set.
CUDA_TOOLKIT_HOME=${CUDA_TOOLKIT_HOME:-${CUDA_HOME:-/apps/ku/intel_h200_gpu/cuda/13.0}}
export CUDA_HOME="${CUDA_TOOLKIT_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

########################### caches ###########################
PROJECT_CACHE=${PROJECT_CACHE:-/dpc/kuin0100/bohao/202509_InferenceModel/.cache}
mkdir -p "$PROJECT_CACHE"/{hf,torch,wandb}
export HF_HOME="$PROJECT_CACHE/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$PROJECT_CACHE/torch"
export WANDB_CACHE_DIR="$PROJECT_CACHE/wandb"

export VLLM_CACHE_ROOT=/tmp/vllm
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
export TRITON_CACHE_DIR=/tmp/triton
export TORCH_COMPILE_DEBUG_DIR=/tmp/torch_compile_debug
export CUDA_CACHE_PATH=/tmp/cuda
export RAY_TMPDIR=/tmp/ray
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

########################### W&B ###########################
export WANDB_API_KEY="${WANDB_API_KEY:-1868e6f8bb348a1ea66684c36334162de64864ff}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-grpo_mixed_v3_qwen35_27B}"
export WANDB_NAME="${WANDB_NAME:-${experiment_name}}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_CACHE/wandb}"
mkdir -p "$WANDB_DIR"

########################### algorithm ###########################
adv_estimator=${ADV_ESTIMATOR:-grpo}
reward_manager=${REWARD_MANAGER:-naive}     # v0.8 experimental NaiveRewardManager (only 'batch' lives in the legacy registry, not the experimental one)
project_name=${PROJECT_NAME:-${WANDB_PROJECT}}

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] STAGE_NAME=${STAGE_NAME}"
echo "[INFO] STAGE_DIR=${STAGE_DIR}"
echo "[INFO] VAL_ROOT=${VAL_ROOT}"
echo "[INFO] train_files=${train_files}"
echo "[INFO] test_files=${test_files}"
echo "[INFO] init_model=${init_model}"
echo "[INFO] experiment_name=${experiment_name}"
echo "[INFO] DEVICE=${DEVICE} n_devices=${n_devices_per_node} fsdp_size=${fsdp_size} sp=${sp_size} gen_tp=${gen_tp} mem_util=${rollout_mem_util}"
echo "[INFO] CUSTOM_REWARD: ${CUSTOM_REWARD_PATH}::${CUSTOM_REWARD_NAME}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
"$PYTHON_BIN" -c 'import sys, ray; print(f"[INFO] Python executable={sys.executable}"); print(f"[INFO] ray={ray.__version__}")'

if [[ ! -d "${init_model}" ]]; then
  echo "[ERROR] INIT_MODEL directory not found: ${init_model}" >&2
  exit 1
fi
if [[ ! -f "${CUSTOM_REWARD_PATH}" ]]; then
  echo "[ERROR] Custom reward file not found: ${CUSTOM_REWARD_PATH}" >&2
  exit 1
fi

if [[ -z "${TRAIN_FILES:-}" ]]; then
  for f in "${train_file_checks[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "[ERROR] Missing train parquet: $f" >&2
      exit 1
    fi
  done
fi

if [[ -z "${TEST_FILES:-}" ]]; then
  for f in "${test_file_checks[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "[ERROR] Missing validation parquet: $f" >&2
      exit 1
    fi
  done
fi

########################### launch ###########################
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=3072 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.trust_remote_code=True \
    reward.reward_manager.name=${reward_manager} \
    reward.custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    reward.custom_reward_function.name="${CUSTOM_REWARD_NAME}" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.validation_data_dir=./${experiment_name}/validation_data \
    trainer.rollout_data_dir=./${experiment_name}/rollout_data \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=${n_devices_per_node} \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.val_before_train=True \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.max_actor_ckpt_to_keep=${max_ckpt_to_keep} \
    trainer.max_critic_ckpt_to_keep=${max_ckpt_to_keep} \
    actor_rollout_ref.model.path=${init_model} \
    +actor_rollout_ref.model.override_config.max_position_embeddings=16384 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.offload_policy=False \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=${INFER_BACKEND} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_mem_util} \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.7} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.95} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K:--1} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE:-True} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-1} \
    "$@"
