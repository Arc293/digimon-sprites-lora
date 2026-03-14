#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# High-end launcher for SDXL LoRA training on multi-GPU systems (e.g. 2x H200).
# - Robust preflight checks
# - Caption-aware dataset mode (HF imagefolder metadata)
# - Resume-safe checkpoint handling
# - Optional auto-fetch of official SDXL trainer script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/scripts/third_party/train_dreambooth_lora_sdxl.py}"
TRAIN_SCRIPT_URL="${TRAIN_SCRIPT_URL:-https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py}"
AUTO_FETCH_TRAINER="${AUTO_FETCH_TRAINER:-1}"
PREP_SCRIPT="${PREP_SCRIPT:-$REPO_DIR/scripts/prepare_hf_imagefolder_dataset.py}"

MODEL_NAME="${MODEL_NAME:-stabilityai/stable-diffusion-xl-base-1.0}"
PRETRAINED_VAE_MODEL="${PRETRAINED_VAE_MODEL:-madebyollin/sdxl-vae-fp16-fix}"

# Source data (png+txt pairs) and caption-aware dataset build output.
TRAIN_PAIR_DIR="${TRAIN_PAIR_DIR:-$REPO_DIR/datasets/vpet_lora/train}"
HF_DATASET_DIR="${HF_DATASET_DIR:-$REPO_DIR/datasets/vpet_lora/hf_imagefolder_train_sdxl}"
USE_CAPTION_DATASET="${USE_CAPTION_DATASET:-1}"
REFRESH_CAPTION_DATASET="${REFRESH_CAPTION_DATASET:-0}"
LINK_MODE="${LINK_MODE:-symlink}"

# Fallback mode (single shared prompt for all images).
INSTANCE_DATA_DIR="${INSTANCE_DATA_DIR:-$REPO_DIR/datasets/vpet_lora/train_images}"
INSTANCE_PROMPT="${INSTANCE_PROMPT:-vpet_style, vpet_left_pose, digimon, full body, partial left-facing, three-quarter view, sprite, pixel art, clean outline, limited color palette}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/lora_vpet_sdxl_h200_nocompromise}"
RESUME_INPUT_DIR="${RESUME_INPUT_DIR:-}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Training recipe (quality-first defaults tuned for large VRAM systems).
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
RESOLUTION="${RESOLUTION:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RANK="${RANK:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-80}"
SEED="${SEED:-3407}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"

USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-1}"
ALLOW_TF32="${ALLOW_TF32:-1}"
CENTER_CROP="${CENTER_CROP:-1}"
RANDOM_FLIP="${RANDOM_FLIP:-0}"
TRAIN_TEXT_ENCODER="${TRAIN_TEXT_ENCODER:-0}"
ENABLE_XFORMERS="${ENABLE_XFORMERS:-0}"

VALIDATION_PROMPT="${VALIDATION_PROMPT:-vpet_style, vpet_left_pose, digimon, full body, partial left-facing, three-quarter view, sprite, pixel art, clean outline, simple background}"
VALIDATION_EPOCHS="${VALIDATION_EPOCHS:-2}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-4}"

INSTALL_DEPS="${INSTALL_DEPS:-0}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_DIR/.cache/hf_datasets}"
HF_HOME="${HF_HOME:-$REPO_DIR/.cache/hf_home}"

# Runtime environment hardening.
export TOKENIZERS_PARALLELISM="false"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export HF_HOME
export HF_DATASETS_CACHE

LOG_FILE=""
TRAINER_HELP=""
TRAIN_ARGS=()

log() {
  printf '[h200-sdxl] %s\n' "$*"
}

die() {
  printf '[h200-sdxl] ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local line="$1"
  local code="$2"
  printf '[h200-sdxl] ERROR: command failed at line %s (exit=%s)\n' "$line" "$code" >&2
  if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    printf '[h200-sdxl] Last 80 log lines from %s:\n' "$LOG_FILE" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
  fi
  exit "$code"
}
trap 'on_error "$LINENO" "$?"' ERR

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

require_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be an integer, got: $value"
}

is_true() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" || "$1" == "yes" || "$1" == "YES" ]]
}

count_images() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.bmp' \) | wc -l | tr -d ' '
}

count_txt() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -iname '*.txt' | wc -l | tr -d ' '
}

count_metadata_rows() {
  local dir="$1"
  [[ -f "$dir/metadata.jsonl" ]] || {
    echo 0
    return
  }
  wc -l < "$dir/metadata.jsonl" | tr -d ' '
}

install_deps() {
  if ! is_true "$INSTALL_DEPS"; then
    return
  fi

  log "Installing/upgrading SDXL training dependencies (INSTALL_DEPS=1)..."
  python3 -m pip install -U \
    "diffusers @ git+https://github.com/huggingface/diffusers.git" \
    accelerate peft datasets ftfy sentencepiece tensorboard transformers
}

python_preflight_imports() {
  python3 - <<'PY'
import importlib
mods = [
    "torch",
    "accelerate",
    "diffusers",
    "transformers",
    "datasets",
    "peft",
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing python modules: {', '.join(missing)}")
print("python_preflight_ok")
PY
}

print_gpu_summary() {
  log "GPU summary:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
}

check_gpu_count() {
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  [[ "$gpu_count" =~ ^[0-9]+$ ]] || die "Could not determine GPU count"
  (( gpu_count >= 1 )) || die "No GPUs detected"
  if (( NUM_PROCESSES > gpu_count )); then
    die "NUM_PROCESSES=$NUM_PROCESSES exceeds detected GPU count=$gpu_count"
  fi
}

check_numeric_config() {
  require_integer "NUM_PROCESSES" "$NUM_PROCESSES"
  require_integer "RESOLUTION" "$RESOLUTION"
  require_integer "TRAIN_BATCH_SIZE" "$TRAIN_BATCH_SIZE"
  require_integer "GRAD_ACCUM" "$GRAD_ACCUM"
  require_integer "RANK" "$RANK"
  require_integer "MAX_TRAIN_STEPS" "$MAX_TRAIN_STEPS"
  require_integer "CHECKPOINT_EVERY" "$CHECKPOINT_EVERY"
  require_integer "CHECKPOINTS_TOTAL_LIMIT" "$CHECKPOINTS_TOTAL_LIMIT"
  require_integer "SEED" "$SEED"
  require_integer "DATALOADER_NUM_WORKERS" "$DATALOADER_NUM_WORKERS"
  require_integer "LR_WARMUP_STEPS" "$LR_WARMUP_STEPS"
  require_integer "VALIDATION_EPOCHS" "$VALIDATION_EPOCHS"
  require_integer "NUM_VALIDATION_IMAGES" "$NUM_VALIDATION_IMAGES"
  (( NUM_PROCESSES >= 1 )) || die "NUM_PROCESSES must be >= 1"
  (( TRAIN_BATCH_SIZE >= 1 )) || die "TRAIN_BATCH_SIZE must be >= 1"
  (( GRAD_ACCUM >= 1 )) || die "GRAD_ACCUM must be >= 1"
  (( RANK >= 1 )) || die "RANK must be >= 1"
  (( CHECKPOINT_EVERY >= 1 )) || die "CHECKPOINT_EVERY must be >= 1"
  (( MAX_TRAIN_STEPS >= CHECKPOINT_EVERY )) || die "MAX_TRAIN_STEPS must be >= CHECKPOINT_EVERY"
}

fetch_trainer_if_needed() {
  if [[ -f "$TRAIN_SCRIPT" ]]; then
    return
  fi

  if ! is_true "$AUTO_FETCH_TRAINER"; then
    die "SDXL trainer not found at $TRAIN_SCRIPT (set AUTO_FETCH_TRAINER=1 or provide TRAIN_SCRIPT)"
  fi

  require_cmd curl
  mkdir -p "$(dirname "$TRAIN_SCRIPT")"

  log "Fetching official SDXL trainer into $TRAIN_SCRIPT"
  local tmp_file
  tmp_file="${TRAIN_SCRIPT}.tmp"
  curl -fL "$TRAIN_SCRIPT_URL" -o "$tmp_file"
  mv "$tmp_file" "$TRAIN_SCRIPT"
  chmod +x "$TRAIN_SCRIPT"
}

load_trainer_help() {
  TRAINER_HELP="$(python3 "$TRAIN_SCRIPT" --help 2>&1 || true)"
  [[ -n "$TRAINER_HELP" ]] || die "Could not read trainer help output from $TRAIN_SCRIPT"
}

supports_arg() {
  local flag="$1"
  grep -Fq -- "$flag" <<<"$TRAINER_HELP"
}

require_trainer_arg() {
  local flag="$1"
  supports_arg "$flag" || die "Trainer does not support required arg: $flag"
}

append_arg_value() {
  local flag="$1"
  local value="$2"
  if supports_arg "$flag"; then
    TRAIN_ARGS+=("$flag" "$value")
  else
    log "trainer missing $flag, skipping"
  fi
}

append_flag_if_true() {
  local flag="$1"
  local cond="$2"
  if is_true "$cond"; then
    if supports_arg "$flag"; then
      TRAIN_ARGS+=("$flag")
    else
      log "trainer missing $flag, skipping"
    fi
  fi
}

caption_dataset_is_stale() {
  [[ -f "$HF_DATASET_DIR/metadata.jsonl" ]] || return 0

  local newer_input=""
  newer_input="$(find "$TRAIN_PAIR_DIR" -maxdepth 1 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.txt' \) \
    -newer "$HF_DATASET_DIR/metadata.jsonl" -print -quit)"
  [[ -n "$newer_input" ]]
}

prepare_caption_dataset() {
  mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

  local need_build="0"
  if [[ ! -d "$HF_DATASET_DIR" ]]; then
    need_build="1"
  fi
  if is_true "$REFRESH_CAPTION_DATASET"; then
    need_build="1"
  fi
  if [[ "$need_build" == "0" ]] && caption_dataset_is_stale; then
    log "Caption dataset is stale; rebuilding from updated pair dataset."
    need_build="1"
  fi

  if [[ "$need_build" == "1" ]]; then
    log "Preparing caption-aware imagefolder dataset at: $HF_DATASET_DIR"
    python3 "$PREP_SCRIPT" \
      --input-dir "$TRAIN_PAIR_DIR" \
      --output-dir "$HF_DATASET_DIR" \
      --mode "$LINK_MODE" \
      --overwrite
  else
    log "Using existing caption dataset: $HF_DATASET_DIR"
  fi

  [[ -f "$HF_DATASET_DIR/metadata.jsonl" ]] || die "metadata.jsonl not found in $HF_DATASET_DIR"

  HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" python3 - <<PY
from datasets import load_dataset
path = r"$HF_DATASET_DIR"
ds = load_dataset(path)
cols = ds["train"].column_names
if "image" not in cols or "text" not in cols:
    raise SystemExit(f"Unexpected columns: {cols}")
if len(ds["train"]) < 50:
    raise SystemExit(f"Dataset too small: {len(ds['train'])}")
print(f"dataset_ok rows={len(ds['train'])} cols={cols}")
PY

  local hf_rows
  hf_rows="$(count_metadata_rows "$HF_DATASET_DIR")"
  log "hf imagefolder rows=$hf_rows dir=$HF_DATASET_DIR"
}

seed_output_dir() {
  mkdir -p "$OUTPUT_DIR"
  if [[ -n "$RESUME_INPUT_DIR" ]]; then
    [[ -d "$RESUME_INPUT_DIR" ]] || die "RESUME_INPUT_DIR does not exist: $RESUME_INPUT_DIR"
    require_cmd rsync
    log "Seeding output dir from RESUME_INPUT_DIR: $RESUME_INPUT_DIR"
    rsync -a "$RESUME_INPUT_DIR"/ "$OUTPUT_DIR"/
  fi
}

find_latest_checkpoint() {
  local latest
  latest="$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  echo "$latest"
}

record_run_metadata() {
  local run_meta="$OUTPUT_DIR/run_metadata_sdxl.json"
  local git_commit
  local train_pair_image_count="0"
  local train_pair_caption_count="0"
  local hf_dataset_row_count="0"
  git_commit="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ -d "$TRAIN_PAIR_DIR" ]]; then
    train_pair_image_count="$(count_images "$TRAIN_PAIR_DIR")"
    train_pair_caption_count="$(count_txt "$TRAIN_PAIR_DIR")"
  fi
  if [[ -d "$HF_DATASET_DIR" ]]; then
    hf_dataset_row_count="$(count_metadata_rows "$HF_DATASET_DIR")"
  fi

  RUN_META_PATH="$run_meta" \
  GIT_COMMIT="$git_commit" \
  MODEL_NAME="$MODEL_NAME" \
  PRETRAINED_VAE_MODEL="$PRETRAINED_VAE_MODEL" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  USE_CAPTION_DATASET="$USE_CAPTION_DATASET" \
  TRAIN_PAIR_DIR="$TRAIN_PAIR_DIR" \
  HF_DATASET_DIR="$HF_DATASET_DIR" \
  INSTANCE_DATA_DIR="$INSTANCE_DATA_DIR" \
  INSTANCE_PROMPT="$INSTANCE_PROMPT" \
  NUM_PROCESSES="$NUM_PROCESSES" \
  MIXED_PRECISION="$MIXED_PRECISION" \
  RESOLUTION="$RESOLUTION" \
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  GRAD_ACCUM="$GRAD_ACCUM" \
  RANK="$RANK" \
  LEARNING_RATE="$LEARNING_RATE" \
  MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
  CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  CHECKPOINTS_TOTAL_LIMIT="$CHECKPOINTS_TOTAL_LIMIT" \
  SEED="$SEED" \
  DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" \
  LR_SCHEDULER="$LR_SCHEDULER" \
  LR_WARMUP_STEPS="$LR_WARMUP_STEPS" \
  USE_8BIT_ADAM="$USE_8BIT_ADAM" \
  USE_GRADIENT_CHECKPOINTING="$USE_GRADIENT_CHECKPOINTING" \
  TRAIN_TEXT_ENCODER="$TRAIN_TEXT_ENCODER" \
  ENABLE_XFORMERS="$ENABLE_XFORMERS" \
  ALLOW_TF32="$ALLOW_TF32" \
  CENTER_CROP="$CENTER_CROP" \
  RANDOM_FLIP="$RANDOM_FLIP" \
  VALIDATION_PROMPT="$VALIDATION_PROMPT" \
  VALIDATION_EPOCHS="$VALIDATION_EPOCHS" \
  NUM_VALIDATION_IMAGES="$NUM_VALIDATION_IMAGES" \
  TRAIN_PAIR_IMAGE_COUNT="$train_pair_image_count" \
  TRAIN_PAIR_CAPTION_COUNT="$train_pair_caption_count" \
  HF_DATASET_ROW_COUNT="$hf_dataset_row_count" \
  python3 - <<'PY'
import json
import os

meta = {
    "model_name": os.environ["MODEL_NAME"],
    "pretrained_vae_model": os.environ["PRETRAINED_VAE_MODEL"],
    "output_dir": os.environ["OUTPUT_DIR"],
    "use_caption_dataset": os.environ["USE_CAPTION_DATASET"],
    "train_pair_dir": os.environ["TRAIN_PAIR_DIR"],
    "train_pair_image_count": int(os.environ["TRAIN_PAIR_IMAGE_COUNT"]),
    "train_pair_caption_count": int(os.environ["TRAIN_PAIR_CAPTION_COUNT"]),
    "hf_dataset_dir": os.environ["HF_DATASET_DIR"],
    "hf_dataset_row_count": int(os.environ["HF_DATASET_ROW_COUNT"]),
    "instance_data_dir": os.environ["INSTANCE_DATA_DIR"],
    "instance_prompt": os.environ["INSTANCE_PROMPT"],
    "num_processes": int(os.environ["NUM_PROCESSES"]),
    "mixed_precision": os.environ["MIXED_PRECISION"],
    "resolution": int(os.environ["RESOLUTION"]),
    "train_batch_size": int(os.environ["TRAIN_BATCH_SIZE"]),
    "gradient_accumulation_steps": int(os.environ["GRAD_ACCUM"]),
    "rank": int(os.environ["RANK"]),
    "learning_rate": os.environ["LEARNING_RATE"],
    "max_train_steps": int(os.environ["MAX_TRAIN_STEPS"]),
    "checkpointing_steps": int(os.environ["CHECKPOINT_EVERY"]),
    "checkpoints_total_limit": int(os.environ["CHECKPOINTS_TOTAL_LIMIT"]),
    "seed": int(os.environ["SEED"]),
    "dataloader_num_workers": int(os.environ["DATALOADER_NUM_WORKERS"]),
    "lr_scheduler": os.environ["LR_SCHEDULER"],
    "lr_warmup_steps": int(os.environ["LR_WARMUP_STEPS"]),
    "use_8bit_adam": os.environ["USE_8BIT_ADAM"],
    "use_gradient_checkpointing": os.environ["USE_GRADIENT_CHECKPOINTING"],
    "train_text_encoder": os.environ["TRAIN_TEXT_ENCODER"],
    "enable_xformers": os.environ["ENABLE_XFORMERS"],
    "allow_tf32": os.environ["ALLOW_TF32"],
    "center_crop": os.environ["CENTER_CROP"],
    "random_flip": os.environ["RANDOM_FLIP"],
    "validation_prompt": os.environ["VALIDATION_PROMPT"],
    "validation_epochs": int(os.environ["VALIDATION_EPOCHS"]),
    "num_validation_images": int(os.environ["NUM_VALIDATION_IMAGES"]),
    "git_commit": os.environ["GIT_COMMIT"],
}

with open(os.environ["RUN_META_PATH"], "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=True, indent=2)
PY
}

main() {
  require_cmd python3
  require_cmd nvidia-smi
  require_cmd accelerate

  [[ -f "$PREP_SCRIPT" ]] || die "Prep script not found: $PREP_SCRIPT"

  fetch_trainer_if_needed
  [[ -f "$TRAIN_SCRIPT" ]] || die "Training script not found: $TRAIN_SCRIPT"

  install_deps
  python_preflight_imports
  check_numeric_config

  print_gpu_summary
  check_gpu_count
  load_trainer_help
  require_trainer_arg --pretrained_model_name_or_path
  require_trainer_arg --output_dir
  require_trainer_arg --instance_prompt

  if is_true "$USE_CAPTION_DATASET"; then
    require_trainer_arg --dataset_name
    [[ -d "$TRAIN_PAIR_DIR" ]] || die "TRAIN_PAIR_DIR not found: $TRAIN_PAIR_DIR"
    local pair_images pair_txt
    pair_images="$(count_images "$TRAIN_PAIR_DIR")"
    pair_txt="$(count_txt "$TRAIN_PAIR_DIR")"
    log "pair dataset images=$pair_images captions=$pair_txt dir=$TRAIN_PAIR_DIR"
    (( pair_images >= 100 )) || die "Need at least 100 images for full training"
    (( pair_txt >= pair_images )) || die "Caption files fewer than images in $TRAIN_PAIR_DIR"
    prepare_caption_dataset
  else
    require_trainer_arg --instance_data_dir
    [[ -d "$INSTANCE_DATA_DIR" ]] || die "INSTANCE_DATA_DIR not found: $INSTANCE_DATA_DIR"
    local num_images
    num_images="$(count_images "$INSTANCE_DATA_DIR")"
    log "instance_data images=$num_images dir=$INSTANCE_DATA_DIR"
    (( num_images >= 100 )) || die "Need at least 100 images for full training"
  fi

  seed_output_dir
  record_run_metadata

  local resume_arg=()
  if is_true "$RESUME_FROM_LATEST"; then
    local latest_ckpt
    latest_ckpt="$(find_latest_checkpoint)"
    if [[ -n "$latest_ckpt" ]]; then
      log "Resuming from latest checkpoint: $latest_ckpt"
      resume_arg=(--resume_from_checkpoint "$latest_ckpt")
    else
      log "No checkpoint found; starting fresh"
    fi
  fi

  local accelerate_args=(
    launch
    --num_processes "$NUM_PROCESSES"
    --mixed_precision "$MIXED_PRECISION"
  )
  if (( NUM_PROCESSES > 1 )); then
    accelerate_args+=(--multi_gpu)
  fi

  TRAIN_ARGS=()

  append_arg_value --pretrained_model_name_or_path "$MODEL_NAME"
  append_arg_value --output_dir "$OUTPUT_DIR"
  append_arg_value --instance_prompt "$INSTANCE_PROMPT"
  append_arg_value --resolution "$RESOLUTION"
  append_arg_value --train_batch_size "$TRAIN_BATCH_SIZE"
  append_arg_value --gradient_accumulation_steps "$GRAD_ACCUM"
  append_arg_value --learning_rate "$LEARNING_RATE"
  append_arg_value --lr_scheduler "$LR_SCHEDULER"
  append_arg_value --lr_warmup_steps "$LR_WARMUP_STEPS"
  append_arg_value --rank "$RANK"
  append_arg_value --max_train_steps "$MAX_TRAIN_STEPS"
  append_arg_value --checkpointing_steps "$CHECKPOINT_EVERY"
  append_arg_value --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
  append_arg_value --seed "$SEED"
  append_arg_value --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  append_arg_value --report_to tensorboard
  append_arg_value --mixed_precision "$MIXED_PRECISION"

  if [[ -n "$PRETRAINED_VAE_MODEL" ]]; then
    append_arg_value --pretrained_vae_model_name_or_path "$PRETRAINED_VAE_MODEL"
  fi

  if is_true "$USE_CAPTION_DATASET"; then
    append_arg_value --dataset_name "$HF_DATASET_DIR"
    append_arg_value --image_column image
    append_arg_value --caption_column text
  else
    append_arg_value --instance_data_dir "$INSTANCE_DATA_DIR"
  fi

  append_flag_if_true --center_crop "$CENTER_CROP"
  append_flag_if_true --random_flip "$RANDOM_FLIP"
  append_flag_if_true --gradient_checkpointing "$USE_GRADIENT_CHECKPOINTING"
  append_flag_if_true --use_8bit_adam "$USE_8BIT_ADAM"
  append_flag_if_true --allow_tf32 "$ALLOW_TF32"
  append_flag_if_true --train_text_encoder "$TRAIN_TEXT_ENCODER"
  append_flag_if_true --enable_xformers_memory_efficient_attention "$ENABLE_XFORMERS"

  if [[ -n "$VALIDATION_PROMPT" ]]; then
    append_arg_value --validation_prompt "$VALIDATION_PROMPT"
    append_arg_value --validation_epochs "$VALIDATION_EPOCHS"
    append_arg_value --num_validation_images "$NUM_VALIDATION_IMAGES"
  fi

  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "Authenticating to Hugging Face Hub (token provided)"
    if command -v hf >/dev/null 2>&1; then
      hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
    elif command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
    else
      log "HF_TOKEN set but no 'hf' or 'huggingface-cli' command found; continuing without CLI login"
    fi
  fi

  local now
  now="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="$OUTPUT_DIR/train_sdxl_${now}.log"

  log "Starting SDXL LoRA training"
  log "trainer=$TRAIN_SCRIPT"
  log "log_file=$LOG_FILE"
  log "output_dir=$OUTPUT_DIR"

  {
    echo "accelerate ${accelerate_args[*]} $TRAIN_SCRIPT ${TRAIN_ARGS[*]} ${resume_arg[*]}"
  } > "$OUTPUT_DIR/launch_command_sdxl_${now}.txt"

  if is_true "$DRY_RUN"; then
    log "DRY_RUN=1, preflight completed and launch command written; skipping training."
    exit 0
  fi

  accelerate "${accelerate_args[@]}" "$TRAIN_SCRIPT" \
    "${TRAIN_ARGS[@]}" \
    "${resume_arg[@]}" \
    2>&1 | tee -a "$LOG_FILE"

  [[ -f "$OUTPUT_DIR/pytorch_lora_weights.safetensors" ]] || die "Training finished but final LoRA file not found"

  log "Training completed successfully"
  log "LoRA: $OUTPUT_DIR/pytorch_lora_weights.safetensors"
}

main "$@"
