#!/usr/bin/env bash
# Convenience wrapper: same as run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh
# but with MCQ reward v2 (auxiliary r2/r3 active for oranbench / srsran /
# srsbench / teleqna / teletable).
#
# Implements recommendation GRPO-A from
#   logs/qwen3_5_27b_telecominstruct_v2_6_qlora_stage3_checkpoint_1984-full/
#   failure_analysis.md §12.5
#
# To revert: just use the original run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh
# (which defaults to MCQ_REWARD_MODE=binary, byte-identical to the ckpt_1984
# training trajectory).
#
# Smoke test:
#   /dpc/kuin0100/conda_env/grpo_py311/bin/python \
#       examples/grpo_TeleInfer/test_mcq_reward_v2.py
export MCQ_REWARD_MODE=v2
exec "$(dirname "$0")/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh" "$@"
