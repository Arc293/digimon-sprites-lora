#!/usr/bin/env bash
set -euo pipefail

# Kaggle-oriented full LoRA training launcher (resume-safe).
# Run this inside a Kaggle notebook terminal/cell with `bash scripts/kaggle_train_flux2_lora_full.sh`.

REPO_NAME="${REPO_NAME:-digimon-sprites-lora}"
REPO_DIR="${REPO_DIR:-/kaggle/working/$REPO_NAME}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/scripts/third_party/train_dreambooth_lora_flux2_klein.py}"

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.2-klein-base-4B}"
INSTANCE_DATA_DIR="${INSTANCE_DATA_DIR:-/kaggle/input/vpet-lora-train-images/train_images}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/outputs/lora_vpet_flux2_klein4b_full}"
RESUME_INPUT_DIR="${RESUME_INPUT_DIR:-}"

INSTANCE_PROMPT="${INSTANCE_PROMPT:-vpet_style, digimon, full body, sprite, pixel art, clean outline, limited color palette}"

RESOLUTION="${RESOLUTION:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
RANK="${RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
SEED="${SEED:-42}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
NUM_PROCESSES="${NUM_PROCESSES:-auto}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
USE_CACHE_LATENTS="${USE_CACHE_LATENTS:-1}"
ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-0}"

install_deps() {
  python3 -m pip install -q --upgrade \
    "diffusers @ git+https://github.com/huggingface/diffusers.git" \
    accelerate peft datasets ftfy sentencepiece tensorboard
}

hf_auth_if_available() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
  fi
}

seed_output_dir() {
  mkdir -p "$OUTPUT_DIR"
  if [[ -n "$RESUME_INPUT_DIR" && -d "$RESUME_INPUT_DIR" ]]; then
    rsync -a "$RESUME_INPUT_DIR"/ "$OUTPUT_DIR"/
  fi
}

find_latest_checkpoint() {
  local root="$1"
  local latest
  latest="$(ls -d "$root"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  echo "$latest"
}

detect_num_processes() {
  if [[ "$NUM_PROCESSES" != "auto" ]]; then
    echo "$NUM_PROCESSES"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -n "$count" && "$count" -ge 1 ]]; then
      echo "$count"
      return
    fi
  fi

  echo "1"
}

main() {
  echo "[kaggle] repo_dir=$REPO_DIR"
  echo "[kaggle] model=$MODEL_NAME"
  echo "[kaggle] data=$INSTANCE_DATA_DIR"
  echo "[kaggle] output=$OUTPUT_DIR"
  local num_procs
  num_procs="$(detect_num_processes)"
  if ! [[ "$num_procs" =~ ^[0-9]+$ ]] || [[ "$num_procs" -lt 1 ]]; then
    echo "Invalid NUM_PROCESSES resolved value: $num_procs" >&2
    exit 1
  fi
  echo "[kaggle] num_processes=$num_procs (NUM_PROCESSES=$NUM_PROCESSES)"
  echo "[kaggle] max_sequence_length=$MAX_SEQUENCE_LENGTH use_cache_latents=$USE_CACHE_LATENTS offload=$ENABLE_OFFLOAD"

  if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
  fi
  if [[ ! -d "$INSTANCE_DATA_DIR" ]]; then
    echo "Dataset dir not found: $INSTANCE_DATA_DIR" >&2
    exit 1
  fi

  install_deps
  hf_auth_if_available
  seed_output_dir

  local resume_arg=()
  local latest_ckpt
  latest_ckpt="$(find_latest_checkpoint "$OUTPUT_DIR")"
  if [[ -n "$latest_ckpt" ]]; then
    echo "[kaggle] resuming from: $latest_ckpt"
    resume_arg=(--resume_from_checkpoint "$latest_ckpt")
  else
    echo "[kaggle] no checkpoint found, starting fresh"
  fi

  local accelerate_args=(
    launch
    --num_processes "$num_procs"
    --mixed_precision "$MIXED_PRECISION"
  )
  if [[ "$num_procs" -gt 1 ]]; then
    accelerate_args+=(--multi_gpu)
  fi

  local training_flags=()
  if [[ "$USE_CACHE_LATENTS" == "1" ]]; then
    training_flags+=(--cache_latents)
  fi
  if [[ "$ENABLE_OFFLOAD" == "1" ]]; then
    training_flags+=(--offload)
  fi

  accelerate "${accelerate_args[@]}" "$TRAIN_SCRIPT" \
    --pretrained_model_name_or_path "$MODEL_NAME" \
    --instance_data_dir "$INSTANCE_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --instance_prompt "$INSTANCE_PROMPT" \
    --resolution "$RESOLUTION" \
    --max_sequence_length "$MAX_SEQUENCE_LENGTH" \
    --center_crop \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --gradient_checkpointing \
    --optimizer AdamW \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler constant \
    --lr_warmup_steps 50 \
    --rank "$RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --max_train_steps "$MAX_TRAIN_STEPS" \
    --checkpointing_steps "$CHECKPOINT_EVERY" \
    --seed "$SEED" \
    --report_to tensorboard \
    --skip_final_inference \
    "${training_flags[@]}" \
    "${resume_arg[@]}"

  echo "[kaggle] done"
  echo "[kaggle] final LoRA: $OUTPUT_DIR/pytorch_lora_weights.safetensors"
}

main "$@"
