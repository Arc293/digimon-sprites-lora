#!/usr/bin/env python3
"""Prepare a LoRA-style dataset from VPet sprite images.

This script keeps source files untouched and writes a normalized dataset with
captions suitable for style-LoRA training.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image


STOP_WORDS = {
    "vpet",
    "vb",
    "vbbe",
    "alysion",
    "raid",
    "boss",
    "perfect",
    "adult",
    "child",
    "baby",
    "black",
    "mode",
}


def parse_hex_color(value: str) -> Tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected 6-char hex color, got: {value}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def clean_name(stem: str) -> str:
    text = stem.lower()
    text = re.sub(r"[()\[\]{}]", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\d+", " ", text)
    tokens = [t for t in re.split(r"\s+", text) if t and t not in STOP_WORDS]
    if not tokens:
        return "digimon"
    return " ".join(tokens)


def alpha_bbox(rgba: Image.Image, alpha_threshold: int) -> Tuple[int, int, int, int]:
    alpha = rgba.getchannel("A")
    mask = alpha.point(lambda px: 255 if px > alpha_threshold else 0)
    box = mask.getbbox()
    if box is None:
        return (0, 0, rgba.width, rgba.height)
    return box


def padded_box(box: Tuple[int, int, int, int], image_size: Tuple[int, int], pad: int) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    width, height = image_size
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def make_caption(name: str, style_trigger_token: str, pose_trigger_token: str, pose_description: str) -> str:
    return (
        f"{style_trigger_token}, {pose_trigger_token}, digimon, {name}, full body, {pose_description}, "
        "sprite, pixel art, clean outline, limited color palette"
    )


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            yield path


def process_image(
    source_path: Path,
    out_path: Path,
    target_size: int,
    sprite_fill: float,
    alpha_threshold: int,
    bbox_pad: int,
    bg_color: Tuple[int, int, int],
) -> None:
    rgba = Image.open(source_path).convert("RGBA")
    box = alpha_bbox(rgba, alpha_threshold=alpha_threshold)
    box = padded_box(box, rgba.size, pad=bbox_pad)
    sprite = rgba.crop(box)

    max_dim = max(sprite.width, sprite.height)
    fit = int(target_size * sprite_fill)
    scale = fit / max_dim if max_dim > 0 else 1.0
    new_w = max(1, int(round(sprite.width * scale)))
    new_h = max(1, int(round(sprite.height * scale)))

    # Nearest-neighbor preserves sprite edges and palette feel.
    sprite = sprite.resize((new_w, new_h), resample=Image.NEAREST)

    canvas = Image.new("RGBA", (target_size, target_size), bg_color + (255,))
    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2
    canvas.alpha_composite(sprite, (x, y))

    # RGB output is generally preferred by common LoRA trainers.
    canvas.convert("RGB").save(out_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VPet images for style LoRA training.")
    parser.add_argument("--input", default="training", help="Source folder with VPet sprites.")
    parser.add_argument("--output", default="datasets/vpet_lora/train", help="Output dataset folder.")
    parser.add_argument("--target-size", type=int, default=512, help="Square output size in pixels.")
    parser.add_argument("--sprite-fill", type=float, default=0.84, help="Sprite max fill ratio within canvas.")
    parser.add_argument("--alpha-threshold", type=int, default=8, help="Alpha threshold for bounding box.")
    parser.add_argument("--bbox-pad", type=int, default=4, help="Padding around detected sprite bbox.")
    parser.add_argument("--bg-color", default="f2f4f8", help="Canvas background hex color, e.g. f2f4f8.")
    parser.add_argument("--trigger-token", default="vpet_style", help="Backward-compatible alias for --style-trigger-token.")
    parser.add_argument("--style-trigger-token", default=None, help="Style trigger token used in captions.")
    parser.add_argument("--pose-trigger-token", default="vpet_left_pose", help="Pose trigger token used in captions.")
    parser.add_argument(
        "--pose-description",
        default="partial left-facing, three-quarter view",
        help="Canonical pose wording to include in captions.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only N images (0 = all).")
    args = parser.parse_args()

    src_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_color = parse_hex_color(args.bg_color)
    style_trigger_token = args.style_trigger_token or args.trigger_token

    rows = []
    count = 0
    for idx, source in enumerate(iter_images(src_dir), start=1):
        if args.limit > 0 and idx > args.limit:
            break
        out_base = f"{idx:05d}_{source.stem}"
        out_png = out_dir / f"{out_base}.png"
        out_txt = out_dir / f"{out_base}.txt"

        process_image(
            source_path=source,
            out_path=out_png,
            target_size=args.target_size,
            sprite_fill=args.sprite_fill,
            alpha_threshold=args.alpha_threshold,
            bbox_pad=args.bbox_pad,
            bg_color=bg_color,
        )

        name = clean_name(source.stem)
        caption = make_caption(
            name=name,
            style_trigger_token=style_trigger_token,
            pose_trigger_token=args.pose_trigger_token,
            pose_description=args.pose_description,
        )
        out_txt.write_text(caption + "\n", encoding="utf-8")

        rows.append((str(source), str(out_png), caption))
        count += 1

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "prepared_image", "caption"])
        writer.writerows(rows)

    print(f"Prepared {count} images -> {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
