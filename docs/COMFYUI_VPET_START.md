# VPet Style Conversion - FLUX.2 Klein 4B Setup

This repo is now configured for a **FLUX.2 img2img pipeline** in ComfyUI, including a LoRA-first option.

## 1) Prepare training data for style LoRA (optional now)

```bash
python3 scripts/prepare_vpet_lora_dataset.py \
  --input training \
  --output datasets/vpet_lora/train \
  --target-size 512 \
  --trigger-token vpet_style
```

Output:
- `datasets/vpet_lora/train/*.png` (normalized images)
- `datasets/vpet_lora/train/*.txt` (captions)
- `datasets/vpet_lora/train/manifest.csv`

## 2) Prepare `working/` images for img2img

```bash
python3 scripts/prepare_working_inputs.py \
  --input working \
  --output datasets/working_prepped \
  --target-size 768
```

Output:
- `datasets/working_prepped/*.png`
- `datasets/working_prepped/manifest.csv`

## 3) Install required FLUX.2 model files in ComfyUI

Place these files in your ComfyUI model folders:

- `models/diffusion_models/`
  - `flux-2-klein-4b.safetensors` (or your installed FLUX.2 Klein 4B filename)
- `models/text_encoders/`
  - `qwen_3_4b.safetensors`
- `models/vae/`
  - `flux2-vae.safetensors`
- `models/loras/` (for LoRA-first workflow)
  - your FLUX style LoRA (example: `vpet_flux2_style.safetensors`)

Then restart ComfyUI.

## 4) Workflow template

Template path:
- `workflows/vpet_img2img_api.json`
- `workflows/vpet_flux2_lora_api.json` (recommended first path when style transfer drifts identity)
- `workflows/vpet_flux2_lora_controlnet_canny_api.json` (LoRA + Canny ControlNet for stronger structure/identity lock)
- `workflows/vpet_img2img_api_stylepush.json` (stronger style transfer, weaker structure lock)
- `workflows/vpet_img2img_api_identity.json` (IP-Adapter identity lock; best when character changes)

Node IDs used by batch script:
- `4`: `UNETLoader`
- `12`: `CLIPLoader` (`type=flux2`)
- `13`: `VAELoader`
- `16`: `LoraLoaderModelOnly` (LoRA application node in LoRA workflow)
- `10`: input image
- `6`: positive prompt
- `7`: negative prompt
- `14`, `15`: `ReferenceLatent` (forces stronger input-image structure retention)
- `20`: `ControlNetLoader` (ControlNet model)
- `21`: `Canny` (edge map from input image)
- `22`: `ControlNetApplyAdvanced` (injects control into positive/negative conditioning)
- `3`: sampler
- `9`: save image

## 5) Run batch conversion

LoRA-first pass for **style + identity balance**:

```bash
python3 scripts/run_comfy_batch_img2img.py \
  --workflow workflows/vpet_flux2_lora_api.json \
  --input-dir datasets/working_prepped \
  --output-dir outputs/vpet_batch_lora \
  --comfy-input-dir /ABS/PATH/TO/ComfyUI/input \
  --comfy-output-dir /ABS/PATH/TO/ComfyUI/output \
  --unet flux-2-klein-4b.safetensors \
  --text-encoder qwen_3_4b.safetensors \
  --vae flux2-vae.safetensors \
  --flux-lora vpet_flux2_style.safetensors \
  --flux-lora-strength 0.9 \
  --steps 18 \
  --cfg 4.1 \
  --denoise 0.36
```

Then optional non-LoRA fidelity/style two-pass:

```bash
python3 scripts/run_comfy_batch_img2img.py \
  --workflow workflows/vpet_img2img_api.json \
  --input-dir datasets/working_prepped \
  --output-dir outputs/vpet_batch_fidelity \
  --comfy-input-dir /ABS/PATH/TO/ComfyUI/input \
  --comfy-output-dir /ABS/PATH/TO/ComfyUI/output \
  --unet flux-2-klein-4b.safetensors \
  --text-encoder qwen_3_4b.safetensors \
  --vae flux2-vae.safetensors \
  --steps 16 \
  --cfg 3.8 \
  --denoise 0.25
```

Second pass for style push (from pass-1 outputs):

```bash
python3 scripts/run_comfy_batch_img2img.py \
  --workflow workflows/vpet_img2img_api_stylepush.json \
  --input-dir outputs/vpet_batch_fidelity \
  --output-dir outputs/vpet_batch \
  --comfy-input-dir /ABS/PATH/TO/ComfyUI/input \
  --comfy-output-dir /ABS/PATH/TO/ComfyUI/output \
  --unet flux-2-klein-4b.safetensors \
  --text-encoder qwen_3_4b.safetensors \
  --vae flux2-vae.safetensors \
  --steps 20 \
  --cfg 4.2 \
  --denoise 0.48 \
  --postprocess-sprite \
  --sprite-size 192 \
  --sprite-colors 40 \
  --sprite-output-size 192
```

Identity lock test (recommended when output changes species/face):

```bash
python3 scripts/run_comfy_batch_img2img.py \
  --workflow workflows/vpet_img2img_api_identity.json \
  --input-dir datasets/working_prepped \
  --output-dir outputs/vpet_batch_identity \
  --comfy-input-dir /ABS/PATH/TO/ComfyUI/input \
  --comfy-output-dir /ABS/PATH/TO/ComfyUI/output \
  --unet flux-2-klein-4b.safetensors \
  --text-encoder qwen_3_4b.safetensors \
  --vae flux2-vae.safetensors \
  --steps 18 \
  --cfg 4.0 \
  --denoise 0.42 \
  --ip-scale 0.94 \
  --postprocess-sprite \
  --sprite-size 192 \
  --sprite-colors 40 \
  --sprite-output-size 192
```

LoRA + ControlNet Canny smoke test (single image):

```bash
python3 scripts/run_comfy_batch_img2img.py \
  --workflow workflows/vpet_flux2_lora_controlnet_canny_api.json \
  --input-dir datasets/working_prepped \
  --output-dir outputs/vpet_batch_lora_controlnet \
  --comfy-input-dir /ABS/PATH/TO/ComfyUI/input \
  --comfy-output-dir /ABS/PATH/TO/ComfyUI/output \
  --unet flux-2-klein-4b.safetensors \
  --text-encoder qwen_3_4b.safetensors \
  --vae flux2-vae.safetensors \
  --flux-lora YOUR_PIXEL_FLUX_LORA.safetensors \
  --flux-lora-strength 0.7 \
  --controlnet YOUR_FLUX_CONTROLNET_CANNY.safetensors \
  --control-strength 0.75 \
  --control-start 0.0 \
  --control-end 0.88 \
  --canny-low 0.35 \
  --canny-high 0.85 \
  --steps 20 \
  --cfg 4.1 \
  --denoise 0.38 \
  --limit 1
```

Compatibility note:
- A FLUX workflow requires a FLUX-compatible LoRA and FLUX-compatible ControlNet.
- SDXL/SD1.5 LoRAs cannot be used directly in this FLUX.2 workflow.

## 6) Recommended tuning range

- Steps: `14` to `22`
- CFG: `3.5` to `4.8`
- Denoise: `0.25` to `0.45`
- FLUX LoRA strength: `0.7` to `1.1`

If output stops matching the source image, reduce denoise first.
If style is too weak after fidelity is correct, increase denoise in small increments (`+0.03`).
If identity still drifts in LoRA mode, reduce LoRA strength by `0.1` before increasing denoise.

## 7) NoobAI Civitai LoRA Integration

Model you shared:
- https://civitai.com/models/378602/pokemon-sprite-xl-pixelart-lora

Important compatibility note:
- This is a **NoobAI/SDXL-family LoRA**, not a FLUX LoRA.
- Do not use it with the FLUX workflow files above.

Files to install in ComfyUI:
- `models/checkpoints/`: a compatible NoobAI (or the exact base family shown on the LoRA page/version tag)
- `models/loras/`: the downloaded `Pokemon Sprite XL PixelArt LoRA` file

Workflow template for this path:
- `workflows/noobai_pokemon_sprite_xl_img2img_api.json`
- `workflows/noobai_pokemon_sprite_xl_img2img_identity_api.json` (conservative identity-preserving preset)

One-image smoke test:

```bash
python3 scripts/run_comfy_batch_img2img.py \
  --workflow workflows/noobai_pokemon_sprite_xl_img2img_api.json \
  --input-dir datasets/working_prepped \
  --output-dir outputs/noobai_sprite_smoke \
  --comfy-input-dir /ABS/PATH/TO/ComfyUI/input \
  --comfy-output-dir /ABS/PATH/TO/ComfyUI/output \
  --checkpoint YOUR_NOOBAI_CHECKPOINT.safetensors \
  --lora YOUR_POKEMON_SPRITE_XL_LORA.safetensors \
  --lora-strength 0.85 \
  --steps 24 \
  --cfg 6.0 \
  --denoise 0.52 \
  --limit 1
```

Tuning:
- Too close to input: increase `--denoise` to `0.58` and `--lora-strength` to `0.95`.
- Identity drift: lower `--denoise` to `0.45`.

Conservative identity-first smoke test:

```bash
python3 scripts/run_comfy_batch_img2img.py \
  --workflow workflows/noobai_pokemon_sprite_xl_img2img_identity_api.json \
  --input-dir datasets/working_prepped \
  --output-dir outputs/noobai_sprite_identity_smoke \
  --comfy-input-dir /ABS/PATH/TO/ComfyUI/input \
  --comfy-output-dir /ABS/PATH/TO/ComfyUI/output \
  --checkpoint YOUR_NOOBAI_CHECKPOINT.safetensors \
  --lora YOUR_POKEMON_SPRITE_XL_LORA.safetensors \
  --lora-strength 0.6 \
  --steps 18 \
  --cfg 4.2 \
  --denoise 0.34 \
  --limit 1
```
