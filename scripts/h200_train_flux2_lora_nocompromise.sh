#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# High-end launcher for FLUX.2-klein LoRA training on multi-GPU systems (e.g. 2x H200).
# - Robust preflight checks
# - Caption-aware dataset mode (HF imagefolder metadata)
# - Resume-safe checkpoint handling
# - Deterministic logging and reproducible run metadata

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/scripts/third_party/train_dreambooth_lora_flux2_klein.py}"
PREP_SCRIPT="${PREP_SCRIPT:-$REPO_DIR/scripts/prepare_hf_imagefolder_dataset.py}"

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.2-klein-base-4B}"

# Source data (png+txt pairs) and caption-aware dataset build output.
TRAIN_PAIR_DIR="${TRAIN_PAIR_DIR:-$REPO_DIR/datasets/vpet_lora/train}"
HF_DATASET_DIR="${HF_DATASET_DIR:-$REPO_DIR/datasets/vpet_lora/hf_imagefolder_train}"
USE_CAPTION_DATASET="${USE_CAPTION_DATASET:-1}"
REFRESH_CAPTION_DATASET="${REFRESH_CAPTION_DATASET:-0}"
LINK_MODE="${LINK_MODE:-symlink}"

# Fallback mode (single shared prompt for all images).
INSTANCE_DATA_DIR="${INSTANCE_DATA_DIR:-$REPO_DIR/datasets/vpet_lora/train_images}"
INSTANCE_PROMPT="${INSTANCE_PROMPT:-vpet_style, vpet_left_pose, digimon, full body, partial left-facing, three-quarter view, sprite, pixel art, clean outline, limited color palette}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/lora_vpet_flux2_klein4b_h200_nocompromise}"
RESUME_INPUT_DIR="${RESUME_INPUT_DIR:-}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Training recipe (quality-first defaults tuned for large VRAM systems).
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
RESOLUTION="${RESOLUTION:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RANK="${RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-7e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-4000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-60}"
SEED="${SEED:-3407}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
WEIGHTING_SCHEME="${WEIGHTING_SCHEME:-none}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
VALIDATION_PROMPT="${VALIDATION_PROMPT:-}"
VALIDATION_EPOCHS="${VALIDATION_EPOCHS:-2}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-4}"

USE_CACHE_LATENTS="${USE_CACHE_LATENTS:-1}"
ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-0}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-0}"
ALLOW_TF32="${ALLOW_TF32:-1}"
CENTER_CROP="${CENTER_CROP:-1}"
RANDOM_FLIP="${RANDOM_FLIP:-0}"

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

log() {
  printf '[h200-train] %s\n' "$*"
}

die() {
  printf '[h200-train] ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local line="$1"
  local code="$2"
  printf '[h200-train] ERROR: command failed at line %s (exit=%s)\n' "$line" "$code" >&2
  if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    printf '[h200-train] Last 80 log lines from %s:\n' "$LOG_FILE" >&2
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

  log "Installing/upgrading training dependencies (INSTALL_DEPS=1)..."
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
  require_integer "LORA_ALPHA" "$LORA_ALPHA"
  require_integer "MAX_TRAIN_STEPS" "$MAX_TRAIN_STEPS"
  require_integer "CHECKPOINT_EVERY" "$CHECKPOINT_EVERY"
  require_integer "CHECKPOINTS_TOTAL_LIMIT" "$CHECKPOINTS_TOTAL_LIMIT"
  require_integer "SEED" "$SEED"
  require_integer "MAX_SEQUENCE_LENGTH" "$MAX_SEQUENCE_LENGTH"
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

prepare_caption_dataset() {
  mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

  local need_build="0"
  if [[ ! -d "$HF_DATASET_DIR" ]]; then
    need_build="1"
  fi
  if is_true "$REFRESH_CAPTION_DATASET"; then
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
  local run_meta="$OUTPUT_DIR/run_metadata.json"
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
  LORA_ALPHA="$LORA_ALPHA" \
  LORA_DROPOUT="$LORA_DROPOUT" \
  LEARNING_RATE="$LEARNING_RATE" \
  MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
  CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  CHECKPOINTS_TOTAL_LIMIT="$CHECKPOINTS_TOTAL_LIMIT" \
  SEED="$SEED" \
  MAX_SEQUENCE_LENGTH="$MAX_SEQUENCE_LENGTH" \
  DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" \
  LR_SCHEDULER="$LR_SCHEDULER" \
  LR_WARMUP_STEPS="$LR_WARMUP_STEPS" \
  WEIGHTING_SCHEME="$WEIGHTING_SCHEME" \
  GUIDANCE_SCALE="$GUIDANCE_SCALE" \
  USE_CACHE_LATENTS="$USE_CACHE_LATENTS" \
  ENABLE_OFFLOAD="$ENABLE_OFFLOAD" \
  USE_GRADIENT_CHECKPOINTING="$USE_GRADIENT_CHECKPOINTING" \
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
    "lora_alpha": int(os.environ["LORA_ALPHA"]),
    "lora_dropout": float(os.environ["LORA_DROPOUT"]),
    "learning_rate": os.environ["LEARNING_RATE"],
    "max_train_steps": int(os.environ["MAX_TRAIN_STEPS"]),
    "checkpointing_steps": int(os.environ["CHECKPOINT_EVERY"]),
    "checkpoints_total_limit": int(os.environ["CHECKPOINTS_TOTAL_LIMIT"]),
    "seed": int(os.environ["SEED"]),
    "max_sequence_length": int(os.environ["MAX_SEQUENCE_LENGTH"]),
    "dataloader_num_workers": int(os.environ["DATALOADER_NUM_WORKERS"]),
    "lr_scheduler": os.environ["LR_SCHEDULER"],
    "lr_warmup_steps": int(os.environ["LR_WARMUP_STEPS"]),
    "weighting_scheme": os.environ["WEIGHTING_SCHEME"],
    "guidance_scale": os.environ["GUIDANCE_SCALE"],
    "use_cache_latents": os.environ["USE_CACHE_LATENTS"],
    "enable_offload": os.environ["ENABLE_OFFLOAD"],
    "use_gradient_checkpointing": os.environ["USE_GRADIENT_CHECKPOINTING"],
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

  [[ -f "$TRAIN_SCRIPT" ]] || die "Training script not found: $TRAIN_SCRIPT"
  [[ -f "$PREP_SCRIPT" ]] || die "Prep script not found: $PREP_SCRIPT"

  install_deps
  python_preflight_imports
  check_numeric_config

  print_gpu_summary
  check_gpu_count

  if is_true "$USE_CAPTION_DATASET"; then
    [[ -d "$TRAIN_PAIR_DIR" ]] || die "TRAIN_PAIR_DIR not found: $TRAIN_PAIR_DIR"
    local pair_images pair_txt
    pair_images="$(count_images "$TRAIN_PAIR_DIR")"
    pair_txt="$(count_txt "$TRAIN_PAIR_DIR")"
    log "pair dataset images=$pair_images captions=$pair_txt dir=$TRAIN_PAIR_DIR"
    (( pair_images >= 100 )) || die "Need at least 100 images for full training"
    (( pair_txt >= pair_images )) || die "Caption files fewer than images in $TRAIN_PAIR_DIR"
    prepare_caption_dataset
  else
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

  local train_args=(
    --pretrained_model_name_or_path "$MODEL_NAME"
    --output_dir "$OUTPUT_DIR"
    --instance_prompt "$INSTANCE_PROMPT"
    --resolution "$RESOLUTION"
    --max_sequence_length "$MAX_SEQUENCE_LENGTH"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --optimizer AdamW
    --learning_rate "$LEARNING_RATE"
    --lr_scheduler "$LR_SCHEDULER"
    --lr_warmup_steps "$LR_WARMUP_STEPS"
    --rank "$RANK"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --max_train_steps "$MAX_TRAIN_STEPS"
    --checkpointing_steps "$CHECKPOINT_EVERY"
    --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
    --seed "$SEED"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --weighting_scheme "$WEIGHTING_SCHEME"
    --guidance_scale "$GUIDANCE_SCALE"
    --report_to tensorboard
    --skip_final_inference
  )

  if is_true "$USE_CAPTION_DATASET"; then
    train_args+=(
      --dataset_name "$HF_DATASET_DIR"
      --image_column image
      --caption_column text
    )
  else
    train_args+=(--instance_data_dir "$INSTANCE_DATA_DIR")
  fi
  if [[ -n "$VALIDATION_PROMPT" ]]; then
    train_args+=(
      --validation_prompt "$VALIDATION_PROMPT"
      --validation_epochs "$VALIDATION_EPOCHS"
      --num_validation_images "$NUM_VALIDATION_IMAGES"
    )
  fi

  if is_true "$CENTER_CROP"; then
    train_args+=(--center_crop)
  fi
  if is_true "$RANDOM_FLIP"; then
    train_args+=(--random_flip)
  fi
  if is_true "$USE_CACHE_LATENTS"; then
    train_args+=(--cache_latents)
  fi
  if is_true "$ENABLE_OFFLOAD"; then
    train_args+=(--offload)
  fi
  if is_true "$USE_GRADIENT_CHECKPOINTING"; then
    train_args+=(--gradient_checkpointing)
  fi
  if is_true "$ALLOW_TF32"; then
    train_args+=(--allow_tf32)
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
  LOG_FILE="$OUTPUT_DIR/train_${now}.log"

  log "Starting training"
  log "log_file=$LOG_FILE"
  log "output_dir=$OUTPUT_DIR"

  {
    echo "accelerate ${accelerate_args[*]} $TRAIN_SCRIPT ${train_args[*]} ${resume_arg[*]}"
  } > "$OUTPUT_DIR/launch_command_${now}.txt"

  if is_true "$DRY_RUN"; then
    log "DRY_RUN=1, preflight completed and launch command written; skipping training."
    exit 0
  fi

  accelerate "${accelerate_args[@]}" "$TRAIN_SCRIPT" \
    "${train_args[@]}" \
    "${resume_arg[@]}" \
    2>&1 | tee -a "$LOG_FILE"

  [[ -f "$OUTPUT_DIR/pytorch_lora_weights.safetensors" ]] || die "Training finished but final LoRA file not found"

  log "Training completed successfully"
  log "LoRA: $OUTPUT_DIR/pytorch_lora_weights.safetensors"
}

main "$@"
