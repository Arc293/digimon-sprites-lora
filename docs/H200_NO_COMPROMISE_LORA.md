# H200 No-Compromise FLUX.2 LoRA Training

This runbook trains a high-capacity style LoRA from your full VPet dataset using:
- Base model: `black-forest-labs/FLUX.2-klein-base-4B`
- Trainer: `scripts/third_party/train_dreambooth_lora_flux2_klein.py`
- Launcher: `scripts/h200_train_flux2_lora_nocompromise.sh`
- Dataset mode: local Hugging Face `imagefolder` with per-image captions from `.txt`
- Public trigger tokens: `vpet_style`, `vpet_left_pose`

## Current Dataset Snapshot
As of March 14, 2026, the refreshed training set is:
- `datasets/vpet_lora/train`: `1203` image+caption pairs
- `datasets/vpet_lora/hf_imagefolder_train_sdxl`: `1203` `metadata.jsonl` rows
- Composition:
  - `688` pre-existing curated V-Pet pairs
  - `515` retained DragonRod imports after duplicate archiving and outline cleanup
- DragonRod archive state:
  - `299` files currently in `training/dragonrod_archived_duplicates`
  - manual spot-check shortlist: `training/dragonrod_archived_review_shortlist.md`

Quick verification commands:
```bash
find datasets/vpet_lora/train -maxdepth 1 -type f -name '*.png' | wc -l
wc -l datasets/vpet_lora/hf_imagefolder_train_sdxl/metadata.jsonl
find training/dragonrod -maxdepth 1 -type f -name '*.png' | wc -l
find training/dragonrod_archived_duplicates -maxdepth 1 -type f -name '*.png' | wc -l
```

Expected output on the current repo state:
- `1203`
- `1203`
- `515`
- `299`

## What This LoRA Is
- Goal: transfer the Digimon VPet sprite style while preserving subject identity in img2img workflows.
- Training set used: `datasets/vpet_lora/train` (expected image+caption pairs).
- Current pair-count target: `1203` images with matching captions.
- Capacity profile:
  - `rank=64`
  - `lora_alpha=64`
  - `lora_dropout=0.05`
- Optimization profile:
  - `optimizer=AdamW`
  - `lr=7e-5`
  - `scheduler=cosine`
  - `warmup=200`
  - `weighting_scheme=none`
- Core run profile:
  - `resolution=512`
  - `max_train_steps=4000`
  - `checkpointing_steps=100`
  - `checkpoints_total_limit=60`
  - `mixed_precision=bf16`
  - `num_processes=2` (for 2x H200)

## Why This Setup
- Uses all captions via `metadata.jsonl` dataset mode (not single shared prompt mode).
- Supports robust resume from latest checkpoint.
- Writes reproducibility metadata and exact launch command per run.
- Validates dataset schema and minimum size before launching.

## Preflight On Remote H200 Host
```bash
cd /path/to/digimon-sprites-lora
bash -n scripts/h200_train_flux2_lora_nocompromise.sh
python3 scripts/prepare_hf_imagefolder_dataset.py --help
```

Optional dry run (checks only, no training):
```bash
DRY_RUN=1 bash scripts/h200_train_flux2_lora_nocompromise.sh
```

Current dry-run preflight should report:
- `pair dataset images=1203 captions=1203`
- `hf imagefolder rows=1203`

## One-Command Full Training
```bash
cd /path/to/digimon-sprites-lora
export HF_TOKEN="<your_hf_token_if_needed>"
export TRAIN_PAIR_DIR="$PWD/datasets/vpet_lora/train"
export OUTPUT_DIR="$PWD/outputs/lora_vpet_flux2_klein4b_h200_nocompromise"
export NUM_PROCESSES=2
bash scripts/h200_train_flux2_lora_nocompromise.sh
```

## Resume Training (If Interrupted)
Keep same `OUTPUT_DIR` and re-run:
```bash
cd /path/to/digimon-sprites-lora
export OUTPUT_DIR="$PWD/outputs/lora_vpet_flux2_klein4b_h200_nocompromise"
export RESUME_FROM_LATEST=1
bash scripts/h200_train_flux2_lora_nocompromise.sh
```

If you trained in another directory and want to continue in a fresh output dir:
```bash
cd /path/to/digimon-sprites-lora
export RESUME_INPUT_DIR="/path/to/old/output_dir"
export OUTPUT_DIR="$PWD/outputs/lora_vpet_flux2_klein4b_h200_nocompromise_resume"
bash scripts/h200_train_flux2_lora_nocompromise.sh
```

## Outputs You Will Get
Inside `OUTPUT_DIR`:
- `pytorch_lora_weights.safetensors` (final LoRA)
- `checkpoint-*` (intermediate checkpoints every 100 steps)
- `train_YYYYmmdd_HHMMSS.log` (full training logs)
- `run_metadata.json` (full run config)
- `launch_command_YYYYmmdd_HHMMSS.txt` (exact launch command)

## Automated Checkpoint Scoring
Use this after training to rank checkpoints automatically:
- script: `scripts/score_flux2_lora_checkpoints.py`
- output: ranked JSON + CSV in your chosen scoring output dir

Quick pass (latest 10 checkpoints, faster):
```bash
cd /path/to/digimon-sprites-lora
python3 scripts/score_flux2_lora_checkpoints.py \
  --train-output-dir "$PWD/outputs/lora_vpet_flux2_klein4b_h200_nocompromise" \
  --checkpoints 10 \
  --prompts 12 \
  --steps 20 \
  --output-dir "$PWD/outputs/lora_checkpoint_scores_quick" \
  --include-final
```

Full pass (more reliable ranking):
```bash
cd /path/to/digimon-sprites-lora
python3 scripts/score_flux2_lora_checkpoints.py \
  --train-output-dir "$PWD/outputs/lora_vpet_flux2_klein4b_h200_nocompromise" \
  --checkpoints 0 \
  --prompts 24 \
  --style-refs 160 \
  --steps 24 \
  --save-preview-images \
  --top-k-preview 4 \
  --output-dir "$PWD/outputs/lora_checkpoint_scores_full" \
  --include-final
```

Scoring formula:
- `combined = 0.45 * prompt_alignment + 0.35 * style_similarity + 0.20 * palette_score`
- `prompt_alignment`: CLIP text-image similarity
- `style_similarity`: CLIP image similarity to VPet style centroid
- `palette_score`: closeness to training-set sprite color-count profile

## Choosing Best LoRA
- Do not assume last checkpoint is best.
- Compare:
  - final `pytorch_lora_weights.safetensors`
  - selected checkpoints around late/mid training (`checkpoint-2800`, `checkpoint-3200`, `checkpoint-3600`, `checkpoint-4000`)
- Evaluate with identical seeds/prompts and your real `working/` samples.

## High-Quality Optional Tweaks
Use only if you want slower but potentially better fitting:
```bash
# Higher capacity (more expressive, risk of overfit if pushed too far)
export RANK=96
export LORA_ALPHA=96

# Longer run
export MAX_TRAIN_STEPS=6000
export CHECKPOINT_EVERY=100

# Add periodic validation renders from trainer
export VALIDATION_PROMPT="vpet_style, pixel sprite, full body digimon, clean outline"
export VALIDATION_EPOCHS=2
export NUM_VALIDATION_IMAGES=4
```

## Safety Notes
- The launcher is strict (`set -Eeuo pipefail`) and fails fast on missing prerequisites.
- If `HF_TOKEN` is set, it tries `hf` then `huggingface-cli`; otherwise it continues without CLI login.
- Caption dataset is rebuilt when needed and verified to include `image` + `text` columns.

## SDXL Alternative Training Path
Alternative launcher:
- `scripts/h200_train_sdxl_lora_nocompromise.sh`

What it does:
- uses SDXL base (`stabilityai/stable-diffusion-xl-base-1.0`) + optional VAE fix
- auto-fetches official diffusers SDXL DreamBooth-LoRA trainer if missing
- reuses same caption dataset pipeline (`datasets/vpet_lora/train` -> `metadata.jsonl`)
- validates supported trainer arguments before launch

Quick SDXL preflight:
```bash
cd /path/to/digimon-sprites-lora
DRY_RUN=1 bash scripts/h200_train_sdxl_lora_nocompromise.sh
```

Full SDXL run:
```bash
cd /path/to/digimon-sprites-lora
export OUTPUT_DIR="$PWD/outputs/lora_vpet_sdxl_h200_nocompromise"
export NUM_PROCESSES=2
bash scripts/h200_train_sdxl_lora_nocompromise.sh
```

Current SDXL preflight should report:
- `pair dataset images=1203 captions=1203`
- `hf imagefolder rows=1203`

## Two-Stage SDXL Inference Path
Canonical left-facing conversion is implemented as:
- Stage 1: pose normalization with LoRA
- Stage 2: sprite refinement with the same LoRA

Runner:
- `scripts/run_sdxl_two_stage_vpet.py`

Example:
```bash
cd /path/to/digimon-sprites-lora
python3 scripts/run_sdxl_two_stage_vpet.py \
  --input-dir working \
  --checkpoint your_sdxl_checkpoint.safetensors \
  --lora your_vpet_sdxl_lora.safetensors \
  --comfy-input-dir /path/to/ComfyUI/input \
  --comfy-output-dir /path/to/ComfyUI/output
```
