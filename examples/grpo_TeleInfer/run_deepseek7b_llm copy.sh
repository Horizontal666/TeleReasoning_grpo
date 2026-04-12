#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck source=/dev/null
. "${REPO_ROOT}/scripts/use_project_cache.sh"

set -x

# python3 -m verl.trainer.main_ppo \
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_trainer_TeleInfer_PPO.yaml' \
    algorithm.adv_estimator=gae \
    data.train_files=/dpc/kuin0100/bohao/202509_InferenceModel/data/telechat_normalized/verl/train.parquet \
    data.val_files=/dpc/kuin0100/bohao/202509_InferenceModel/data/telechat_normalized/verl/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/dpc/kuin0100/bohao/202509_InferenceModel/model/deepseek-math-7b-instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/dpc/kuin0100/bohao/202509_InferenceModel/model/deepseek-math-7b-instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_example_wirelessMathBench' \
    trainer.experiment_name='deepseek_llm_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=1 \
    trainer.use_legacy_worker_impl=auto \
    trainer.total_epochs=2 $@
