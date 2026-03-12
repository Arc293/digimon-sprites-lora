#!/usr/bin/env bash
set -euo pipefail

# OOM-safe preset for Kaggle multi-GPU (e.g., T4 x2).
# You can still override any variable before calling this script.

export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export RESOLUTION="${RESOLUTION:-384}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export RANK="${RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-8}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1200}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/kaggle_train_flux2_lora_full.sh"
