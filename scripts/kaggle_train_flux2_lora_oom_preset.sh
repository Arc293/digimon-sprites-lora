#!/usr/bin/env bash
set -euo pipefail

# OOM-safe preset for Kaggle T4-class GPUs.
# You can still override any variable before calling this script.

export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export RESOLUTION="${RESOLUTION:-320}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export RANK="${RANK:-4}"
export LORA_ALPHA="${LORA_ALPHA:-4}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1200}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-256}"
export ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-1}"
export USE_CACHE_LATENTS="${USE_CACHE_LATENTS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/kaggle_train_flux2_lora_full.sh"
