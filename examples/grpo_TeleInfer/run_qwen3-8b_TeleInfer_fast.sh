# set -x

# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=/workspace/wbh/202509_InferenceModel/data/GRPO/CT_from_finefineweb_commu_5000/train.parquet \
#     data.val_files=/workspace/wbh/202509_InferenceModel/data/GRPO/CT_from_finefineweb_commu_5000/test.parquet \
#     data.train_batch_size=240 \
#     data.max_prompt_length=512 \
#     data.max_response_length=1024 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=/workspace/wbh/202509_InferenceModel/outputs/model_FT_merged/Qwen3-8B-TelecomInstruct_v0.1_peft \
#     trainer.nnodes=1 \
#     trainer.n_gpus_per_node=6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=True \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#     actor_rollout_ref.actor.use_dynamic_bsz=True \
#     actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=2 \
#     actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
#     actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
#     actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
#     actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.project_name='TeleReasoning_GRPO' \
#     trainer.experiment_name='qwen3_8b_TelecomInstruct_v0.1_FFWcommu5000_0107_fast' \
#     trainer.logger='["console","wandb"]' \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=5 \
#     "$@"

set -x

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/wbh/202509_InferenceModel/data/GRPO/telemath/train.parquet \
    data.val_files=/workspace/wbh/202509_InferenceModel/data/GRPO/telemath/test.parquet \
    data.train_batch_size=240 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/workspace/wbh/202509_InferenceModel/model/Qwen3-8B \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.project_name='TeleReasoning_GRPO' \
    trainer.experiment_name='qwen3_8b_onlyTelemath_newComputeScore_0107_fast' \
    trainer.logger='["console","wandb"]' \
    trainer.rollout_data_dir="/workspace/wbh/202509_InferenceModel/outputs/eval/rollout" \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 \
    "$@"
