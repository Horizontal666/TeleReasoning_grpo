# 跑完run_qwen3-8b训练模型以后运行：
# cd /workspace/wbh/202509_InferenceModel/Inference/verl
# python3 -m verl.model_merger merge --backend fsdp --local_dir /workspace/wbh/202509_InferenceModel/outputs/grpo/checkpoints/TeleReasoning_GRPO/Qwen2.5-7B-Instruct-gsm8k_0320/global_step_580/actor --target_dir /workspace/wbh/202509_InferenceModel/outputs/grpo/checkpoints/TeleReasoning_GRPO/Qwen2.5-7B-Instruct-gsm8k_0320/global_step_580/actor/huggingface


#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Respect externally provided GPU selection, and fall back to the original 8-GPU setup.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
export CUDA_VISIBLE_DEVICES

export VLLM_USE_V1=1

# UPDATED: use your local Qwen2.5-7B-Instruct path (change to Qwen3-8B if you actually use that)
PRETRAINED_MODEL=/workspace/wbh/202509_InferenceModel/model/Qwen2.5-7B-Instruct
# PRETRAINED_MODEL=/workspace/wbh/202509_InferenceModel/outputs/model_FT_merged/Qwen2.5-7B-Instruct-telemath_self_gen_v0_peft_checkpoint-10

n_nodes=1
IFS=',' read -r -a visible_gpu_array <<< "$CUDA_VISIBLE_DEVICES"
n_gpus_per_node="${#visible_gpu_array[@]}"
if [[ "$n_gpus_per_node" -lt 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES resolved to zero GPUs: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (n_gpus_per_node=${n_gpus_per_node})"

tensor_model_parallel_size=1
save_freq=50

# UPDATED: point math data to your /workspace tree (adjust if your paths differ)
# data_train_path=/workspace/wbh/202509_InferenceModel/data/GRPO/telemath_self_gen_v0/train.parquet
# r1_test_path=/workspace/wbh/202509_InferenceModel/data/GRPO/telemath_self_gen_v0/test.parquet
data_train_path=/workspace/wbh/202509_InferenceModel/data/datasets/gsm8k/train.parquet
r1_test_path=/workspace/wbh/202509_InferenceModel/data/datasets/gsm8k/test.parquet

# UPDATED: new experiment name for TeleInfer-style FlowRL run
# experiment_name="Qwen2.5-7B-Instruct-telemath_self_gen_v0_peft_checkpoint-10_telemath_self_gen_v0_0313_flowrl"
# experiment_name="Qwen2.5-7B-Instruct_FFW_datafilter_v0.1_newnewComputeScore_rollout8_0120_flowrl"
experiment_name="Qwen2.5-7B-Instruct-gsm8k_0320"

ROLL_OUT_DIR=/workspace/wbh/202509_InferenceModel/outputs/eval/rollout/${experiment_name}
VAL_DATA_DIR=/workspace/wbh/202509_InferenceModel/outputs/eval/validation/${experiment_name}
# ROLL_OUT_DIR=/workspace/wbh/202509_InferenceModel/outputs/eval/rollout/Qwen2.5-7B-Instruct_FFW_datafilter_v0.1_newnewComputeScore_rollout8_0120_flowrl

CUSTOM_REWARD_PATH="${SCRIPT_DIR}/custom_reward.py"
# Default to the new DeepMath-like boxed + equivalence reward.
# Quick rollback:
#   CUSTOM_REWARD_NAME=my_math_reward_fn bash flowrl_qwen3-8b_TeleInfer.sh
CUSTOM_REWARD_NAME="${CUSTOM_REWARD_NAME:-my_math_reward_fn_deepmath_boxed}"

max_prompt_length=2048 #2048
max_response_length=8000 #8000

set -euo pipefail

# Preflight checks to avoid empty-path failures in dataset loader.
: "${data_train_path:?data_train_path is empty}"
: "${r1_test_path:?r1_test_path is empty}"
if [[ ! -f "$data_train_path" ]]; then
    echo "Train parquet not found: $data_train_path" >&2
    exit 1
fi
if [[ ! -f "$r1_test_path" ]]; then
    echo "Validation parquet not found: $r1_test_path" >&2
    exit 1
fi

mkdir -p "$ROLL_OUT_DIR" "$VAL_DATA_DIR"

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_train_path \
    data.val_files=$r1_test_path \
    data.train_batch_size=128 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='left' \
    actor_rollout_ref.model.path=$PRETRAINED_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=processed_logprobs \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    custom_reward_function.path=$CUSTOM_REWARD_PATH \
    custom_reward_function.name=$CUSTOM_REWARD_NAME \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='TeleReasoning_GRPO' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.resume_mode=auto \
    trainer.save_freq=$save_freq \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.rollout_data_dir=$ROLL_OUT_DIR \
    trainer.validation_data_dir=$VAL_DATA_DIR \
    "$@"
