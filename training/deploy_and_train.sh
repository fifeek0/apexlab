#!/bin/bash
# Deploy training files to DGX Spark and run the training.
# Usage: bash training/deploy_and_train.sh [--smoke N]
#
# Prerequisites: ssh dgx works, pip deps installed on DGX
# (transformers, peft, trl, datasets, bitsandbytes, accelerate)

set -euo pipefail
DGX="dgx"
REMOTE_DIR="~/racing-coach-training"
SMOKE_STEPS=""

for arg in "$@"; do
  case "$arg" in
    --smoke) shift; SMOKE_STEPS="--max-steps ${1:-10}" ;;
  esac
done

echo "=== Deploying to DGX ==="
ssh $DGX "mkdir -p $REMOTE_DIR/{configs,data,training}"

# Transfer files
scp configs/gemma4_e2b_racing_lora_v1.yaml $DGX:$REMOTE_DIR/configs/
scp training/train_racing_coach.py $DGX:$REMOTE_DIR/training/
scp data/racing_sft_train.jsonl $DGX:$REMOTE_DIR/data/
scp data/racing_sft_val.jsonl $DGX:$REMOTE_DIR/data/ 2>/dev/null || echo "(no val data yet)"

echo "=== Files deployed ==="
ssh $DGX "wc -l $REMOTE_DIR/data/*.jsonl"

echo "=== Installing deps (if needed) ==="
ssh $DGX "pip3 install --break-system-packages -q transformers peft trl datasets accelerate bitsandbytes pyyaml 2>&1 | tail -2"

echo "=== Starting training ==="
ssh $DGX "cd $REMOTE_DIR && python3 training/train_racing_coach.py \
  --config configs/gemma4_e2b_racing_lora_v1.yaml \
  --train-data data/racing_sft_train.jsonl \
  --val-data data/racing_sft_val.jsonl \
  $SMOKE_STEPS"
