# Kaggle Detour: Full LoRA Training (FLUX.2-klein-4B)

This runbook is for training a stronger LoRA on Kaggle GPU, then using the final LoRA locally in ComfyUI.

## 1) Prepare a Kaggle upload bundle locally

From repo root:

```bash
python3 scripts/prepare_kaggle_train_images_bundle.py \
  --input-dir datasets/vpet_lora/train_images \
  --output-dir dist/kaggle/vpet_lora_train_images/train_images \
  --overwrite
```

This creates:
- `dist/kaggle/vpet_lora_train_images/train_images/*.png`
- `dist/kaggle/vpet_lora_train_images/manifest.csv`

Upload `dist/kaggle/vpet_lora_train_images/` to Kaggle as a dataset.

Recommended Kaggle dataset slug: `vpet-lora-train-images`

## 2) Kaggle notebook setup

In Kaggle notebook settings:
- Enable GPU.
- Add your uploaded dataset input.
- Add a secret `HF_TOKEN` (for better Hub download reliability).

First notebook cell:

```bash
cd /kaggle/working
git clone https://github.com/Arc293/digimon-sprites-lora.git
cd digimon-sprites-lora
```

## 3) Start full training

```bash
cd /kaggle/working/digimon-sprites-lora
export HF_TOKEN="$HF_TOKEN"
export INSTANCE_DATA_DIR="/kaggle/input/vpet-lora-train-images/train_images"
export OUTPUT_DIR="/kaggle/working/outputs/lora_vpet_flux2_klein4b_full"

# Tune if needed:
export MAX_TRAIN_STEPS=1200
export CHECKPOINT_EVERY=100
export RESOLUTION=512
export LEARNING_RATE=1e-4
export RANK=16
export LORA_ALPHA=16
export TRAIN_BATCH_SIZE=1
export GRAD_ACCUM=2
export MIXED_PRECISION=fp16
# Optional override; default is auto-detect from nvidia-smi.
# For T4 x2:
export NUM_PROCESSES=2

bash scripts/kaggle_train_flux2_lora_full.sh
```

OOM-safe preset (recommended first run on Kaggle `T4 x2`):

```bash
cd /kaggle/working/digimon-sprites-lora
export HF_TOKEN="$HF_TOKEN"
export INSTANCE_DATA_DIR="/kaggle/input/vpet-lora-train-images/train_images"
export OUTPUT_DIR="/kaggle/working/outputs/lora_vpet_flux2_klein4b_full"
bash scripts/kaggle_train_flux2_lora_oom_preset.sh
```

The OOM preset now force-sets safe values to avoid stale notebook env vars:
- `NUM_PROCESSES=1`
- `RESOLUTION=256`
- `MAX_SEQUENCE_LENGTH=128`
- `ENABLE_OFFLOAD=1`
- `USE_CACHE_LATENTS=0`

If you want to override those, use `*_OVERRIDE` names, for example:

```bash
export RESOLUTION_OVERRIDE=320
export MAX_SEQUENCE_LENGTH_OVERRIDE=256
bash scripts/kaggle_train_flux2_lora_oom_preset.sh
```

Why this preset is single-GPU:
- `accelerate --multi_gpu` is data-parallel, so each GPU still loads the full model.
- FLUX2-klein + Qwen text encoder can exceed per-GPU VRAM on T4 16GB, even with 2 GPUs available.
- This preset uses `NUM_PROCESSES=1`, `RESOLUTION=256`, `MAX_SEQUENCE_LENGTH=128`, and `--offload`.

Notes:
- The launcher auto-detects GPU count and uses `accelerate --multi_gpu` when `NUM_PROCESSES > 1`.
- Effective batch size is `TRAIN_BATCH_SIZE * GRAD_ACCUM * NUM_PROCESSES`.
- For Kaggle `T4 x2`, `TRAIN_BATCH_SIZE=1` and `GRAD_ACCUM=2` is a good starting point.

## 4) Resume in later Kaggle sessions

Kaggle sessions are time-limited. Save checkpoint outputs as a Kaggle dataset version, then attach it next run.

When resuming, set:

```bash
export RESUME_INPUT_DIR="/kaggle/input/<your-checkpoint-dataset-path>"
bash scripts/kaggle_train_flux2_lora_full.sh
```

The script auto-detects latest `checkpoint-*` and passes `--resume_from_checkpoint`.

## 5) Export final LoRA back to local ComfyUI

After training completes, get:

- `/kaggle/working/outputs/lora_vpet_flux2_klein4b_full/pytorch_lora_weights.safetensors`

Copy it locally to:

- `~/Coding/ComfyUI/models/loras/vpet_flux2_style_full.safetensors`

Then use in workflow with:
- `--flux-lora vpet_flux2_style_full.safetensors`

## 6) Stability knobs (if training fails)

- OOM: reduce `RESOLUTION` to `384`, then `320`; increase `GRAD_ACCUM` to keep effective batch.
- OOM at model load (`text_encoder.to`): set `NUM_PROCESSES=1`, `ENABLE_OFFLOAD=1`, `MAX_SEQUENCE_LENGTH=256`.
- If OOM persists at model load even with the OOM preset, Kaggle T4 VRAM is insufficient for this trainer/model combo in your run context. At that point switch to a 24GB+ GPU (L4/A10/A5000/3090/4090/A100).
- NaNs: reduce `LEARNING_RATE` to `5e-5` or `2e-5`.
- Too weak style: increase `MAX_TRAIN_STEPS` to `1600+`.
- Overfitting/weird outputs: lower `RANK` from `16` to `8`, or stop at an earlier checkpoint.
- If OOM persists on `NUM_PROCESSES=1`, keep `ENABLE_OFFLOAD=1` and drop `RESOLUTION` to `256`.
