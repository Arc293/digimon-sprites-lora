#!/usr/bin/env python3
"""Run batch img2img through a ComfyUI API workflow template.

Default workflow targets a standard FLUX.2 pipeline and can optionally apply
a FLUX style LoRA if present in the workflow graph.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib import error, request

from PIL import Image


DEFAULT_PROMPT_TEMPLATE = (
    "{name}, same character as input image, preserve pose and silhouette, preserve body proportions, "
    "digimon, full body centered, virtual pet sprite style, pixel art, 1px dark outline, limited color palette"
)
DEFAULT_NEGATIVE = (
    "different character, changed pose, changed silhouette, different anatomy, extra limbs, missing limbs, "
    "blurry, noisy, jpeg artifacts, text, watermark, logo, signature, photorealistic, 3d render, painterly"
)


NODE_UNET = "4"
NODE_CLIP = "12"
NODE_VAE = "13"
NODE_LOAD_IMAGE = "10"
NODE_POSITIVE = "6"
NODE_NEGATIVE = "7"
NODE_SAMPLER = "3"
NODE_SAVE = "9"
NODE_IP_ADAPTER_APPLY = "31"
LORA_NODE_TYPES = {
    "LoraLoader",
    "LoraLoaderModelOnly",
    "FluxLoraLoader",
    "FluxLoraLoaderModelOnly",
}


class PromptLostError(RuntimeError):
    """Raised when a queued prompt disappears before history/output is available."""


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            yield path


def sanitize_name(stem: str) -> str:
    cleaned = stem.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"^\s*\d+\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "digimon"


def apply_sprite_postprocess(
    image_path: Path,
    sprite_size: int,
    palette_colors: int,
    output_size: int,
) -> None:
    """Pixelize and palette-limit image to better match classic VPet sprite look."""
    im = Image.open(image_path).convert("RGB")
    small = im.resize((sprite_size, sprite_size), resample=Image.Resampling.LANCZOS)
    quant = small.quantize(colors=palette_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
    if output_size != sprite_size:
        quant = quant.resize((output_size, output_size), resample=Image.Resampling.NEAREST)
    quant.save(image_path, format="PNG", optimize=True)


def post_json(url: str, payload: Dict) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


def get_json(url: str) -> Dict:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


def queue_prompt(comfy_url: str, prompt: Dict, client_id: str) -> str:
    result = post_json(f"{comfy_url}/prompt", {"prompt": prompt, "client_id": client_id})
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {result}")
    return prompt_id


def poll_history(comfy_url: str, prompt_id: str, timeout_s: int = 900, poll_s: float = 1.5) -> Dict:
    start = time.time()
    while True:
        history = get_json(f"{comfy_url}/history/{prompt_id}")

        if prompt_id in history:
            entry = history[prompt_id]
        else:
            entry = history

        if entry:
            status = entry.get("status", {}) or {}
            status_str = status.get("status_str")
            completed = status.get("completed", False)

            if entry.get("outputs"):
                return entry

            if completed and status_str == "error":
                error_message = "ComfyUI prompt failed"
                for message in status.get("messages", []):
                    if message and message[0] == "execution_error":
                        payload = message[1]
                        node_id = payload.get("node_id")
                        node_type = payload.get("node_type")
                        exc_type = payload.get("exception_type")
                        exc_msg = (payload.get("exception_message") or "").strip()
                        error_message = (
                            f"ComfyUI execution_error at node {node_id} ({node_type}): "
                            f"{exc_type}: {exc_msg}"
                        )
                        break
                raise RuntimeError(error_message)

        # If prompt vanished from queue and we still have no history, fail fast.
        queue = get_json(f"{comfy_url}/queue")
        running = queue.get("queue_running", []) if isinstance(queue, dict) else []
        pending = queue.get("queue_pending", []) if isinstance(queue, dict) else []

        def has_prompt(items: List) -> bool:
            for item in items:
                if isinstance(item, list) and len(item) > 1 and item[1] == prompt_id:
                    return True
            return False

        if not entry and not has_prompt(running) and not has_prompt(pending):
            raise PromptLostError(
                f"Prompt {prompt_id} is no longer in ComfyUI queue and no history entry is available."
            )

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")

        time.sleep(poll_s)


def first_output_image(entry: Dict, comfy_output_dir: Path) -> Optional[Path]:
    outputs = entry.get("outputs", {})
    for _, node_out in outputs.items():
        images = node_out.get("images", [])
        if not images:
            continue
        img = images[0]
        filename = img.get("filename")
        subfolder = img.get("subfolder", "")
        if not filename:
            continue
        return comfy_output_dir / subfolder / filename
    return None


def fallback_output_by_prefix(comfy_output_dir: Path, prefix: str, after_ts: float) -> Optional[Path]:
    """Find newest generated image matching SaveImage prefix after this job started."""
    candidates: List[Path] = []
    for path in comfy_output_dir.rglob(f"{prefix}*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            continue
        try:
            if path.stat().st_mtime + 1.0 >= after_ts:
                candidates.append(path)
        except OSError:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def fetch_model_list(comfy_url: str, model_type: str) -> List[str]:
    payload = get_json(f"{comfy_url}/models/{model_type}")
    if isinstance(payload, list):
        return payload
    return []


def assert_model_exists(comfy_url: str, model_type: str, model_name: str) -> None:
    available = fetch_model_list(comfy_url, model_type)
    if not available:
        raise RuntimeError(
            f"No models detected at /models/{model_type}. "
            f"Install '{model_name}' in the corresponding ComfyUI model folder and restart ComfyUI."
        )
    if model_name not in available:
        preview = ", ".join(available[:10])
        raise RuntimeError(
            f"{model_type} model '{model_name}' not found in ComfyUI. "
            f"Detected {len(available)} entries. First entries: {preview}"
        )


def find_lora_nodes(workflow: Dict) -> List[str]:
    lora_nodes: List[str] = []
    for node_id, node in workflow.items():
        if node.get("class_type") in LORA_NODE_TYPES:
            lora_nodes.append(node_id)
    return lora_nodes


def read_lora_name_from_workflow(workflow: Dict) -> Optional[str]:
    for node_id in find_lora_nodes(workflow):
        inputs = workflow[node_id].get("inputs", {})
        for key in ("lora_name", "lora", "name"):
            value = inputs.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def apply_overrides(
    workflow: Dict,
    input_filename: str,
    prompt: str,
    negative_prompt: str,
    save_prefix: str,
    unet_name: Optional[str],
    text_encoder_name: Optional[str],
    vae_name: Optional[str],
    clip_type: Optional[str],
    steps: Optional[int],
    cfg: Optional[float],
    denoise: Optional[float],
    seed: Optional[int],
    ip_scale: Optional[float],
    flux_lora: Optional[str],
    flux_lora_strength: Optional[float],
) -> Dict:
    wf = copy.deepcopy(workflow)

    wf[NODE_LOAD_IMAGE]["inputs"]["image"] = input_filename
    wf[NODE_POSITIVE]["inputs"]["text"] = prompt
    wf[NODE_NEGATIVE]["inputs"]["text"] = negative_prompt
    wf[NODE_SAVE]["inputs"]["filename_prefix"] = save_prefix

    if unet_name:
        wf[NODE_UNET]["inputs"]["unet_name"] = unet_name

    if text_encoder_name:
        wf[NODE_CLIP]["inputs"]["clip_name"] = text_encoder_name

    if vae_name:
        wf[NODE_VAE]["inputs"]["vae_name"] = vae_name

    if clip_type:
        wf[NODE_CLIP]["inputs"]["type"] = clip_type

    sampler_inputs = wf[NODE_SAMPLER]["inputs"]
    if steps is not None:
        sampler_inputs["steps"] = steps
    if cfg is not None:
        sampler_inputs["cfg"] = cfg
    if denoise is not None:
        sampler_inputs["denoise"] = denoise
    if seed is not None:
        sampler_inputs["seed"] = seed

    # Optional: only applied in workflows that include Flux IP-Adapter node 31.
    if ip_scale is not None and NODE_IP_ADAPTER_APPLY in wf:
        wf[NODE_IP_ADAPTER_APPLY]["inputs"]["ip_scale"] = ip_scale

    if flux_lora is not None or flux_lora_strength is not None:
        lora_nodes = find_lora_nodes(wf)
        if not lora_nodes:
            raise RuntimeError(
                "LoRA override requested but no LoRA loader node was found in workflow. "
                f"Supported node types: {', '.join(sorted(LORA_NODE_TYPES))}"
            )

        for node_id in lora_nodes:
            inputs = wf[node_id]["inputs"]
            if flux_lora is not None:
                if "lora_name" in inputs:
                    inputs["lora_name"] = flux_lora
                elif "lora" in inputs:
                    inputs["lora"] = flux_lora
                else:
                    inputs["name"] = flux_lora

            if flux_lora_strength is not None:
                if "strength_model" in inputs:
                    inputs["strength_model"] = flux_lora_strength
                elif "strength" in inputs:
                    inputs["strength"] = flux_lora_strength
                elif "scale" in inputs:
                    inputs["scale"] = flux_lora_strength
                else:
                    inputs["strength_model"] = flux_lora_strength

                if "strength_clip" in inputs:
                    inputs["strength_clip"] = flux_lora_strength

    return wf


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run ComfyUI img2img workflow over a folder.")
    parser.add_argument("--workflow", default="workflows/vpet_img2img_api.json", help="Workflow JSON path.")
    parser.add_argument("--input-dir", default="datasets/working_prepped", help="Prepared input folder.")
    parser.add_argument("--output-dir", default="outputs/vpet_batch", help="Final output folder in this repo.")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI API base URL.")
    parser.add_argument("--comfy-input-dir", required=True, help="ComfyUI input directory path.")
    parser.add_argument("--comfy-output-dir", required=True, help="ComfyUI output directory path.")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE, help="Positive prompt template with {name}.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE, help="Negative prompt text.")
    parser.add_argument("--unet", default=None, help="Override diffusion model filename (models/diffusion_models).")
    parser.add_argument(
        "--text-encoder",
        default=None,
        help="Override text encoder filename (models/text_encoders), e.g. qwen_3_4b.safetensors.",
    )
    parser.add_argument("--vae", default=None, help="Override VAE filename (models/vae), e.g. flux2-vae.safetensors.")
    parser.add_argument("--clip-type", default=None, help="Override CLIPLoader type, e.g. flux2.")
    parser.add_argument("--steps", type=int, default=None, help="Override sampler steps.")
    parser.add_argument("--cfg", type=float, default=None, help="Override CFG.")
    parser.add_argument("--denoise", type=float, default=None, help="Override denoise amount.")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducibility.")
    parser.add_argument("--ip-scale", type=float, default=None, help="Override Flux IP-Adapter scale (workflow with node 31).")
    parser.add_argument("--flux-lora", default=None, help="Override FLUX LoRA filename (models/loras).")
    parser.add_argument(
        "--flux-lora-strength",
        type=float,
        default=None,
        help="Override FLUX LoRA strength (maps to strength_model/strength depending on loader node).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only N images (0 = all).")
    parser.add_argument(
        "--postprocess-sprite",
        action="store_true",
        help="Apply pixelization + palette limiting to outputs for closer VPet sprite look.",
    )
    parser.add_argument("--sprite-size", type=int, default=192, help="Internal sprite resolution for postprocess.")
    parser.add_argument("--sprite-colors", type=int, default=40, help="Palette color count for postprocess.")
    parser.add_argument(
        "--sprite-output-size",
        type=int,
        default=192,
        help="Final output size after postprocess (use 192 for true sprite size, or 768 for enlarged preview).",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip API validation that diffusion model/text encoder/VAE exist in ComfyUI model lists.",
    )
    parser.add_argument(
        "--prompt-timeout",
        type=int,
        default=900,
        help="Max seconds to wait for each ComfyUI prompt before timeout.",
    )
    args = parser.parse_args()

    workflow_path = Path(args.workflow)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    comfy_input_dir = Path(args.comfy_input_dir)
    comfy_output_dir = Path(args.comfy_output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    comfy_input_dir.mkdir(parents=True, exist_ok=True)

    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    client_id = str(uuid.uuid4())

    selected_unet = args.unet or workflow[NODE_UNET]["inputs"]["unet_name"]
    selected_text_encoder = args.text_encoder or workflow[NODE_CLIP]["inputs"]["clip_name"]
    selected_vae = args.vae or workflow[NODE_VAE]["inputs"]["vae_name"]
    selected_lora = args.flux_lora or read_lora_name_from_workflow(workflow)

    if not args.skip_model_check:
        assert_model_exists(args.comfy_url, "diffusion_models", selected_unet)
        assert_model_exists(args.comfy_url, "text_encoders", selected_text_encoder)
        assert_model_exists(args.comfy_url, "vae", selected_vae)
        if selected_lora:
            assert_model_exists(args.comfy_url, "loras", selected_lora)

    images = list(iter_images(input_dir))
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        raise RuntimeError(f"No images found in: {input_dir}")

    print(f"Processing {len(images)} images")

    for idx, image_path in enumerate(images, start=1):
        safe_key = f"batch_{idx:05d}_{uuid.uuid4().hex[:8]}_{image_path.stem}.png"
        comfy_image_path = comfy_input_dir / safe_key
        shutil.copy2(image_path, comfy_image_path)

        prompt = args.prompt_template.format(name=sanitize_name(image_path.stem))
        save_prefix = f"vpet_{image_path.stem}"
        prompt_start_ts = time.time()

        prompt_graph = apply_overrides(
            workflow=workflow,
            input_filename=safe_key,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            save_prefix=save_prefix,
            unet_name=args.unet,
            text_encoder_name=args.text_encoder,
            vae_name=args.vae,
            clip_type=args.clip_type,
            steps=args.steps,
            cfg=args.cfg,
            denoise=args.denoise,
            seed=args.seed,
            ip_scale=args.ip_scale,
            flux_lora=args.flux_lora,
            flux_lora_strength=args.flux_lora_strength,
        )

        prompt_id = queue_prompt(args.comfy_url, prompt_graph, client_id=client_id)
        generated: Optional[Path] = None

        try:
            entry = poll_history(args.comfy_url, prompt_id, timeout_s=args.prompt_timeout)
            generated = first_output_image(entry, comfy_output_dir=comfy_output_dir)
        except (TimeoutError, PromptLostError):
            generated = fallback_output_by_prefix(
                comfy_output_dir=comfy_output_dir,
                prefix=save_prefix,
                after_ts=prompt_start_ts,
            )
            if generated:
                print(
                    f"[warn] Recovered output by filename prefix for prompt {prompt_id}: {generated.name}"
                )
            else:
                raise

        if not generated or not generated.exists():
            generated = fallback_output_by_prefix(
                comfy_output_dir=comfy_output_dir,
                prefix=save_prefix,
                after_ts=prompt_start_ts,
            )
        if not generated or not generated.exists():
            raise RuntimeError(f"No output image found for prompt {prompt_id}")

        final_out = output_dir / f"{image_path.stem}.png"
        shutil.copy2(generated, final_out)
        if args.postprocess_sprite:
            apply_sprite_postprocess(
                image_path=final_out,
                sprite_size=args.sprite_size,
                palette_colors=args.sprite_colors,
                output_size=args.sprite_output_size,
            )
        print(f"[{idx}/{len(images)}] {image_path.name} -> {final_out.name}")

    print(f"Done. Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
