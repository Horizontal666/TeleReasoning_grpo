#!/usr/bin/env bash
# Convenience wrapper: same as run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh
# but with MCQ reward v3_teleqna — adds two teleqna-only penalty terms on
# top of v2 (fabrication-template r4 and hedging-density r5).
#
# Behaviour summary:
#   - data_source == "teleqna"  → v2 r1/r2/r3 + new r4/r5 penalties
#   - other MCQ sources         → identical to v2 (r4/r5 do NOT fire)
#   - non-MCQ sources           → unaffected
#
# Implements the recommendation in
#   logs/qwen3_5_27b_telecominstruct_v2_6_qlora_stage3_checkpoint_1984-full/
#   teleqna_failure_analysis.md §8 "Recommendations for GRPO data preparation".
#
# Rollback paths:
#   - Pure v2 (no teleqna penalties): use run_qwen3_5_27b_fsdp_mixed_v3_stage1_4_mcqv2.sh
#   - Original ckpt_1984 binary reward: use run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh
#     (MCQ_REWARD_MODE unset → binary).
#
# Smoke test:
#   /dpc/kuin0100/conda_env/grpo_py311/bin/python \
#       examples/grpo_TeleInfer/test_mcq_reward_v3.py
export MCQ_REWARD_MODE=v3_teleqna
exec "$(dirname "$0")/run_qwen3_5_27b_fsdp_mixed_v3_stage1_4.sh" "$@"
