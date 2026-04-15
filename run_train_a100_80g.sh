#!/usr/bin/env bash
set -euo pipefail

python3 train_sft.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_path data/processed \
  --output_dir outputs/qwen3-8b-thinking-whoandwhen-a100-80g \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --max_seq_length 4096 \
  --use_bf16 \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --save_total_limit 2 \
  --dataloader_num_workers 8
