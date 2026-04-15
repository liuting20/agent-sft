#!/usr/bin/env bash
set -euo pipefail

python3 scripts/prepare_dataset.py \
  --dataset_name Kevin355/Who_and_When \
  --dataset_config Algorithm-Generated \
  --output_dir data/processed
