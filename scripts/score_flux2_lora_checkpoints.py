#!/usr/bin/env python3
"""Rank FLUX.2-klein LoRA checkpoints using automated CLIP-based evaluation.

This script generates deterministic samples for each checkpoint LoRA and scores
outputs by combining:
- Prompt alignment score (CLIP text-image cosine similarity)
- Style similarity score (CLIP image-image cosine similarity to style centroid)
- Palette score (how close unique-color count is to sprite-style references)

Outputs:
- ranked JSON report
- ranked CSV table

Notes:
- Requires a recent `diffusers` build with `Flux2KleinPipeline`.
- Designed for checkpoint directories produced by
  `train_dreambooth_lora_flux2_klein.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"


@dataclass
class Candidate:
    name: str
    step: int
    lora_path: Path


class ScoringError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score and rank FLUX.2 LoRA checkpoints.")
    parser.add_argument("--train-output-dir", required=True, help="Training output dir containing checkpoint-*.")
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base FLUX.2 model id/path used to train LoRA.",
    )
    parser.add_argument(
        "--caption-source-dir",
        default="datasets/vpet_lora/train",
        help="Directory containing .txt captions to sample evaluation prompts from.",
    )
    parser.add_argument(
        "--style-ref-dir",
        default="datasets/vpet_lora/train",
        help="Directory containing style reference images.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lora_checkpoint_scores",
        help="Directory for reports and optional preview images.",
    )
    parser.add_argument("--prompts", type=int, default=24, help="Number of prompts per checkpoint.")
    parser.add_argument("--style-refs", type=int, default=160, help="Number of style reference images.")
    parser.add_argument("--steps", type=int, default=24, help="Inference steps per sample.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--width", type=int, default=512, help="Generation width.")
    parser.add_argument("--height", type=int, default=512, help="Generation height.")
    parser.add_argument("--max-sequence-length", type=int, default=512, help="Tokenizer max sequence length.")
    parser.add_argument("--seed", type=int, default=3407, help="Base seed for prompt sampling and generation.")
    parser.add_argument(
        "--checkpoints",
        type=int,
        default=0,
        help="Max number of checkpoint-* dirs to score (0 = all). Latest checkpoints are preferred.",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Also score train-output-dir/pytorch_lora_weights.safetensors if present.",
    )
    parser.add_argument(
        "--clip-model",
        default=DEFAULT_CLIP_MODEL,
        help="CLIP model used for scoring.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Flux pipeline dtype.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Flux inference device (cuda recommended).",
    )
    parser.add_argument(
        "--clip-device",
        default="cuda",
        help="CLIP scoring device. Use cpu if you want to reserve VRAM for generation.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF cache dir for model downloads.",
    )
    parser.add_argument(
        "--save-preview-images",
        action="store_true",
        help="Save generated images per checkpoint for manual review.",
    )
    parser.add_argument(
        "--top-k-preview",
        type=int,
        default=4,
        help="Keep preview images only for top K checkpoints (0 = keep all).",
    )
    parser.add_argument(
        "--report-prefix",
        default="flux2_checkpoint_scores",
        help="Prefix for output report files.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise ScoringError(f"{label} not found: {path}")


def iter_images(path: Path) -> Iterable[Path]:
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def load_captions(caption_dir: Path) -> list[str]:
    caps = []
    for txt in sorted(caption_dir.glob("*.txt")):
        text = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            caps.append(text)
    if not caps:
        raise ScoringError(f"No non-empty .txt captions found in {caption_dir}")
    return caps


def sample_list(items: list, n: int, seed: int) -> list:
    if n <= 0 or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:n]]


def discover_candidates(train_output_dir: Path, include_final: bool, max_checkpoints: int) -> list[Candidate]:
    candidates: list[Candidate] = []

    checkpoint_dirs: list[tuple[int, Path]] = []
    for ckpt_dir in train_output_dir.glob("checkpoint-*"):
        if not ckpt_dir.is_dir():
            continue
        suffix = ckpt_dir.name.split("-")[-1]
        if not suffix.isdigit():
            continue
        checkpoint_dirs.append((int(suffix), ckpt_dir))

    for step, ckpt_dir in sorted(checkpoint_dirs, key=lambda x: x[0]):
        if not ckpt_dir.is_dir():
            continue
        lora = ckpt_dir / "pytorch_lora_weights.safetensors"
        if lora.is_file():
            candidates.append(Candidate(name=ckpt_dir.name, step=step, lora_path=lora))

    if max_checkpoints > 0 and len(candidates) > max_checkpoints:
        candidates = candidates[-max_checkpoints:]

    if include_final:
        final_lora = train_output_dir / "pytorch_lora_weights.safetensors"
        if final_lora.is_file():
            max_step = max([c.step for c in candidates], default=0)
            candidates.append(Candidate(name="final", step=max_step + 1, lora_path=final_lora))

    if not candidates:
        raise ScoringError(
            "No candidate LoRAs found. Expected checkpoint-*/pytorch_lora_weights.safetensors (and/or final LoRA)."
        )

    return candidates


def torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def normalize_embeddings(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def estimate_palette_target(images: list[Path], sample_side: int = 128) -> float:
    vals = []
    for p in images:
        im = Image.open(p).convert("RGB").resize((sample_side, sample_side), Image.Resampling.NEAREST)
        arr = np.asarray(im)
        uniq = np.unique(arr.reshape(-1, 3), axis=0).shape[0]
        vals.append(float(uniq))
    return float(np.median(vals)) if vals else 128.0


def palette_score(image: Image.Image, target_colors: float, sample_side: int = 128) -> tuple[float, float]:
    rgb = image.convert("RGB").resize((sample_side, sample_side), Image.Resampling.NEAREST)
    arr = np.asarray(rgb)
    uniq = float(np.unique(arr.reshape(-1, 3), axis=0).shape[0])
    # Gaussian score around target color count.
    sigma = max(16.0, target_colors * 0.35)
    score = math.exp(-((uniq - target_colors) ** 2) / (2.0 * sigma * sigma))
    return score, uniq


def build_style_centroid(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    style_images: list[Path],
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    embs: list[torch.Tensor] = []
    for i in range(0, len(style_images), batch_size):
        batch_paths = style_images[i : i + batch_size]
        batch_pil = [Image.open(p).convert("RGB") for p in batch_paths]
        clip_in = clip_processor(images=batch_pil, return_tensors="pt")
        clip_in = {k: v.to(device) for k, v in clip_in.items()}
        with torch.inference_mode():
            feats = clip_model.get_image_features(**clip_in)
        embs.append(normalize_embeddings(feats).detach().cpu())
    all_emb = torch.cat(embs, dim=0)
    centroid = normalize_embeddings(all_emb.mean(dim=0, keepdim=True))[0]
    return centroid


def build_prompt_embeddings(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    prompts: list[str],
    device: torch.device,
) -> torch.Tensor:
    clip_in = clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    clip_in = {k: v.to(device) for k, v in clip_in.items()}
    with torch.inference_mode():
        txt = clip_model.get_text_features(**clip_in)
    return normalize_embeddings(txt).detach().cpu()


def score_image(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image: Image.Image,
    prompt_emb: torch.Tensor,
    style_centroid: torch.Tensor,
    clip_device: torch.device,
    palette_target: float,
) -> tuple[float, float, float, float]:
    clip_in = clip_processor(images=[image.convert("RGB")], return_tensors="pt")
    clip_in = {k: v.to(clip_device) for k, v in clip_in.items()}
    with torch.inference_mode():
        img_feat = clip_model.get_image_features(**clip_in)
    img_feat = normalize_embeddings(img_feat).detach().cpu()[0]

    prompt_alignment = float(torch.dot(img_feat, prompt_emb))
    style_similarity = float(torch.dot(img_feat, style_centroid))
    palette_sim, unique_colors = palette_score(image=image, target_colors=palette_target)
    return prompt_alignment, style_similarity, palette_sim, unique_colors


def load_flux_pipeline(base_model: str, dtype: torch.dtype, device: torch.device, cache_dir: Optional[str]):
    try:
        from diffusers import Flux2KleinPipeline
    except Exception as exc:  # pragma: no cover
        raise ScoringError(
            "Could not import Flux2KleinPipeline from diffusers. Install a recent diffusers build."
        ) from exc

    kwargs = {"torch_dtype": dtype}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    pipe = Flux2KleinPipeline.from_pretrained(base_model, **kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_image(pipe, prompt: str, seed: int, steps: int, guidance: float, width: int, height: int, max_seq_len: int):
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else None
    if device is None:
        device = next(pipe.transformer.parameters()).device

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            width=width,
            height=height,
            max_sequence_length=max_seq_len,
            generator=generator,
        )
    if not hasattr(out, "images") or not out.images:
        raise ScoringError("Pipeline returned no images")
    return out.images[0]


def write_reports(rows: list[dict], output_dir: Path, prefix: str) -> tuple[Path, Path]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{prefix}_{ts}.json"
    csv_path = output_dir / f"{prefix}_{ts}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)

    fieldnames = [
        "rank",
        "name",
        "step",
        "lora_path",
        "combined_score",
        "prompt_alignment_mean",
        "style_similarity_mean",
        "palette_score_mean",
        "unique_colors_mean",
        "num_prompts",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = i
            writer.writerow(out)

    return json_path, csv_path


def main() -> None:
    args = parse_args()

    train_output_dir = Path(args.train_output_dir).resolve()
    caption_source_dir = Path(args.caption_source_dir).resolve()
    style_ref_dir = Path(args.style_ref_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_exists(train_output_dir, "train output dir")
    ensure_exists(caption_source_dir, "caption source dir")
    ensure_exists(style_ref_dir, "style ref dir")

    device = torch.device(args.device)
    clip_device = torch.device(args.clip_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ScoringError("--device=cuda requested but CUDA is not available")
    if clip_device.type == "cuda" and not torch.cuda.is_available():
        raise ScoringError("--clip-device=cuda requested but CUDA is not available")

    candidates = discover_candidates(
        train_output_dir=train_output_dir,
        include_final=args.include_final,
        max_checkpoints=args.checkpoints,
    )

    all_captions = load_captions(caption_source_dir)
    prompts = sample_list(all_captions, args.prompts, args.seed)

    style_images_all = list(iter_images(style_ref_dir))
    if not style_images_all:
        raise ScoringError(f"No images found in style_ref_dir={style_ref_dir}")
    style_images = sample_list(style_images_all, args.style_refs, args.seed + 1)

    print(f"[score] candidates={len(candidates)} prompts={len(prompts)} style_refs={len(style_images)}")

    clip_model = CLIPModel.from_pretrained(args.clip_model, cache_dir=args.cache_dir).to(clip_device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model, cache_dir=args.cache_dir)

    style_centroid = build_style_centroid(
        clip_model=clip_model,
        clip_processor=clip_processor,
        style_images=style_images,
        device=clip_device,
    )
    prompt_embs = build_prompt_embeddings(
        clip_model=clip_model,
        clip_processor=clip_processor,
        prompts=prompts,
        device=clip_device,
    )

    palette_target = estimate_palette_target(style_images)
    print(f"[score] palette_target_unique_colors={palette_target:.2f}")

    pipe = load_flux_pipeline(
        base_model=args.base_model,
        dtype=torch_dtype(args.dtype),
        device=device,
        cache_dir=args.cache_dir,
    )

    rows: list[dict] = []
    for cand in candidates:
        print(f"[score] checkpoint={cand.name} step={cand.step}")

        if hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

        pipe.load_lora_weights(str(cand.lora_path.parent), weight_name=cand.lora_path.name)

        preview_dir = output_dir / "preview" / cand.name
        if args.save_preview_images:
            preview_dir.mkdir(parents=True, exist_ok=True)

        prompt_scores = []
        style_scores = []
        palette_scores = []
        unique_colors = []

        for i, prompt in enumerate(prompts):
            seed = args.seed + (cand.step * 1000) + i
            image = generate_image(
                pipe=pipe,
                prompt=prompt,
                seed=seed,
                steps=args.steps,
                guidance=args.guidance,
                width=args.width,
                height=args.height,
                max_seq_len=args.max_sequence_length,
            )

            p_align, s_sim, p_sim, uniq = score_image(
                clip_model=clip_model,
                clip_processor=clip_processor,
                image=image,
                prompt_emb=prompt_embs[i],
                style_centroid=style_centroid,
                clip_device=clip_device,
                palette_target=palette_target,
            )
            prompt_scores.append(p_align)
            style_scores.append(s_sim)
            palette_scores.append(p_sim)
            unique_colors.append(uniq)

            if args.save_preview_images:
                image.save(preview_dir / f"{i:03d}.png")

        prompt_mean = float(np.mean(prompt_scores))
        style_mean = float(np.mean(style_scores))
        palette_mean = float(np.mean(palette_scores))
        uniq_mean = float(np.mean(unique_colors))

        combined = (0.45 * prompt_mean) + (0.35 * style_mean) + (0.20 * palette_mean)

        rows.append(
            {
                "name": cand.name,
                "step": cand.step,
                "lora_path": str(cand.lora_path),
                "combined_score": round(combined, 6),
                "prompt_alignment_mean": round(prompt_mean, 6),
                "style_similarity_mean": round(style_mean, 6),
                "palette_score_mean": round(palette_mean, 6),
                "unique_colors_mean": round(uniq_mean, 3),
                "num_prompts": len(prompts),
            }
        )

    rows.sort(key=lambda x: x["combined_score"], reverse=True)

    if args.save_preview_images and args.top_k_preview > 0:
        keep = {r["name"] for r in rows[: args.top_k_preview]}
        preview_root = output_dir / "preview"
        if preview_root.exists():
            for p in preview_root.iterdir():
                if p.is_dir() and p.name not in keep:
                    shutil.rmtree(p, ignore_errors=True)

    json_path, csv_path = write_reports(rows=rows, output_dir=output_dir, prefix=args.report_prefix)

    print("[score] ranking complete")
    for i, row in enumerate(rows[:10], start=1):
        print(
            f"  {i:02d}. {row['name']:<16} score={row['combined_score']:.6f} "
            f"prompt={row['prompt_alignment_mean']:.6f} style={row['style_similarity_mean']:.6f} "
            f"palette={row['palette_score_mean']:.6f}"
        )
    print(f"[score] json={json_path}")
    print(f"[score] csv={csv_path}")


if __name__ == "__main__":
    main()
