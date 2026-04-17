#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen3-8B}"

python3 eval_accuracy.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_path data/processed \
  --split validation \
  --output_dir outputs/eval-qwen3-8b-thinking \
  --batch_size 4 \
  --max_new_tokens 256 \
  --use_bf16 \
  --attn_implementation flash_attention_2
