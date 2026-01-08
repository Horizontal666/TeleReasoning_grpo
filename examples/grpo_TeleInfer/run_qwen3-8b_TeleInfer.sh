# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

# 跑完run_qwen3-8b训练模型以后运行：
# cd /workspace/wbh/202509_InferenceModel/Inference/verl
# python3 -m verl.model_merger merge --backend fsdp --local_dir //workspace/wbh/202509_InferenceModel/outputs/grpo/checkpoints/TeleReasoning_GRPO/qwen3_8b_FFWcommu5000_0106_fast/global_step_100/actor --target_dir /workspace/wbh/202509_InferenceModel/outputs/grpo/checkpoints/TeleReasoning_GRPO/qwen3_8b_FFWcommu5000_0106_fast/global_step_100/actor/huggingface

# 执行完后，你这个 actor/huggingface 目录里就会有完整的 HF 权重（通常是若干 model-*.safetensors 或 pytorch_model.bin），再加上原来的 config / tokenizer。这一步已经隐含地“合并了 base 模型 Qwen3-8B + 你的 GRPO 更新”，不需要再额外拿Qwen3-8B 做任何事情。

# data.train_files=/workspace/wbh/202509_InferenceModel/data/math/CT_from_finefineweb_all_3000/train.parquet \
# data.val_files=/workspace/wbh/202509_InferenceModel/data/math/CT_from_finefineweb_all_3000/test.parquet \

# /workspace/wbh/202509_InferenceModel/model/Qwen3-8B
# 


# set -x

# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=/workspace/wbh/202509_InferenceModel/data/math/telemath_train_chattemplate.parquet \
#     data.val_files=/workspace/wbh/202509_InferenceModel/data/math/telemath_train_chattemplate.parquet \
#     actor_rollout_ref.model.path=/workspace/wbh/202509_InferenceModel/outputs/model_FT_merged/Qwen3-8B-TelecomInstruct_v0.1_peft \
#     trainer.n_gpus_per_node=6 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     trainer.project_name='TeleReasoning_GRPO' \
#     trainer.experiment_name='qwen3_8b_TelecomInstructv0.1_onlyTelemath_0101' \
#     data.train_batch_size=256 \
#     data.max_prompt_length=512 \
#     data.max_response_length=512 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=True \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=3 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console","wandb"]' \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=10 \
#     "$@"



set -x

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/wbh/202509_InferenceModel/data/GRPO/telemath/train.parquet \
    data.val_files=/workspace/wbh/202509_InferenceModel/data/GRPO/telemath/test.parquet \
    actor_rollout_ref.model.path=/workspace/wbh/202509_InferenceModel/model/Qwen3-8B \
    trainer.n_gpus_per_node=6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.project_name='TeleReasoning_GRPO' \
    trainer.experiment_name='qwen3_8b_onlyTelemath_0105' \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    "$@"