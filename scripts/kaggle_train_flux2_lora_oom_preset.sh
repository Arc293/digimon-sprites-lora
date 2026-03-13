#!/usr/bin/env bash
set -euo pipefail

# OOM-safe preset for Kaggle T4-class GPUs.
# This preset intentionally overrides common memory-heavy env vars so prior notebook
# exports do not accidentally force high-memory settings.
# If you need custom values, use *_OVERRIDE env vars listed below.

export NUM_PROCESSES="${NUM_PROCESSES_OVERRIDE:-1}"
export RESOLUTION="${RESOLUTION_OVERRIDE:-256}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE_OVERRIDE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM_OVERRIDE:-4}"
export MIXED_PRECISION="${MIXED_PRECISION_OVERRIDE:-fp16}"
export LEARNING_RATE="${LEARNING_RATE_OVERRIDE:-5e-5}"
export RANK="${RANK_OVERRIDE:-4}"
export LORA_ALPHA="${LORA_ALPHA_OVERRIDE:-4}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS_OVERRIDE:-1200}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY_OVERRIDE:-100}"
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH_OVERRIDE:-128}"
export ENABLE_OFFLOAD="${ENABLE_OFFLOAD_OVERRIDE:-1}"
export USE_CACHE_LATENTS="${USE_CACHE_LATENTS_OVERRIDE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_OVERRIDE:-expandable_segments:True,max_split_size_mb:64}"

echo "[oom-preset] NUM_PROCESSES=$NUM_PROCESSES RESOLUTION=$RESOLUTION MAX_SEQUENCE_LENGTH=$MAX_SEQUENCE_LENGTH"
echo "[oom-preset] RANK=$RANK ALPHA=$LORA_ALPHA OFFLOAD=$ENABLE_OFFLOAD CACHE_LATENTS=$USE_CACHE_LATENTS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/kaggle_train_flux2_lora_full.sh"
