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
git clone <YOUR_REPO_URL> Digimon_vpets
cd Digimon_vpets
```

## 3) Start full training

```bash
cd /kaggle/working/Digimon_vpets
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
export GRAD_ACCUM=4
export MIXED_PRECISION=fp16

bash scripts/kaggle_train_flux2_lora_full.sh
```

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
- NaNs: reduce `LEARNING_RATE` to `5e-5` or `2e-5`.
- Too weak style: increase `MAX_TRAIN_STEPS` to `1600+`.
- Overfitting/weird outputs: lower `RANK` from `16` to `8`, or stop at an earlier checkpoint.
