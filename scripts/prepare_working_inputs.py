#!/usr/bin/env python3
"""Normalize working images for ComfyUI img2img conversion.

- Finds foreground region (alpha-aware or border-color subtraction)
- Crops and centers subject on a square canvas
- Writes PNG files ready for batch img2img
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageChops


def parse_hex_color(value: str) -> Tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected 6-char hex color, got: {value}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            yield path


def estimate_border_color(rgb: Image.Image) -> Tuple[int, int, int]:
    w, h = rgb.size
    if w == 0 or h == 0:
        return (255, 255, 255)

    samples = []
    stride_x = max(1, w // 50)
    stride_y = max(1, h // 50)

    for x in range(0, w, stride_x):
        samples.append(rgb.getpixel((x, 0)))
        samples.append(rgb.getpixel((x, h - 1)))
    for y in range(0, h, stride_y):
        samples.append(rgb.getpixel((0, y)))
        samples.append(rgb.getpixel((w - 1, y)))

    rs = [p[0] for p in samples]
    gs = [p[1] for p in samples]
    bs = [p[2] for p in samples]
    return (int(statistics.median(rs)), int(statistics.median(gs)), int(statistics.median(bs)))


def find_foreground_bbox(image: Image.Image, threshold: int, alpha_threshold: int, min_area_ratio: float) -> Tuple[int, int, int, int]:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    alpha_box = alpha.point(lambda px: 255 if px > alpha_threshold else 0).getbbox()

    if alpha_box is not None and alpha_box != (0, 0, rgba.width, rgba.height):
        return alpha_box

    rgb = rgba.convert("RGB")
    bg_color = estimate_border_color(rgb)
    bg = Image.new("RGB", rgb.size, bg_color)
    diff = ImageChops.difference(rgb, bg).convert("L")
    mask = diff.point(lambda px: 255 if px > threshold else 0)
    box = mask.getbbox()

    if box is None:
        return (0, 0, rgb.width, rgb.height)

    bw = box[2] - box[0]
    bh = box[3] - box[1]
    if (bw * bh) / float(rgb.width * rgb.height) < min_area_ratio:
        return (0, 0, rgb.width, rgb.height)

    return box


def pad_box(box: Tuple[int, int, int, int], image_size: Tuple[int, int], pad: int) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    width, height = image_size
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def process_one(
    source: Path,
    output: Path,
    target_size: int,
    fill_ratio: float,
    bg_color: Tuple[int, int, int],
    threshold: int,
    alpha_threshold: int,
    min_area_ratio: float,
    bbox_pad: int,
) -> Tuple[int, int, int, int]:
    im = Image.open(source).convert("RGBA")
    box = find_foreground_bbox(im, threshold=threshold, alpha_threshold=alpha_threshold, min_area_ratio=min_area_ratio)
    box = pad_box(box, im.size, pad=bbox_pad)
    crop = im.crop(box)

    max_dim = max(crop.width, crop.height)
    fit = int(target_size * fill_ratio)
    scale = fit / max_dim if max_dim > 0 else 1.0
    new_w = max(1, int(round(crop.width * scale)))
    new_h = max(1, int(round(crop.height * scale)))
    resized = crop.resize((new_w, new_h), resample=Image.LANCZOS)

    canvas = Image.new("RGBA", (target_size, target_size), bg_color + (255,))
    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2
    canvas.alpha_composite(resized, (x, y))
    canvas.convert("RGB").save(output, format="PNG", optimize=True)

    return box


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare working images for sprite-style img2img.")
    parser.add_argument("--input", default="working", help="Source working folder.")
    parser.add_argument("--output", default="datasets/working_prepped", help="Output folder.")
    parser.add_argument("--target-size", type=int, default=768, help="Square output size in pixels.")
    parser.add_argument("--fill-ratio", type=float, default=0.88, help="Subject max fill ratio in output.")
    parser.add_argument("--bg-color", default="f2f4f8", help="Canvas background hex color.")
    parser.add_argument("--threshold", type=int, default=22, help="Foreground threshold for border subtraction.")
    parser.add_argument("--alpha-threshold", type=int, default=8, help="Alpha threshold for transparent inputs.")
    parser.add_argument("--min-area-ratio", type=float, default=0.02, help="Fallback to full frame under this area.")
    parser.add_argument("--bbox-pad", type=int, default=8, help="Padding around detected subject bbox.")
    parser.add_argument("--limit", type=int, default=0, help="Process only N images (0 = all).")
    args = parser.parse_args()

    src_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_color = parse_hex_color(args.bg_color)

    rows = []
    total = 0
    for idx, source in enumerate(iter_images(src_dir), start=1):
        if args.limit > 0 and idx > args.limit:
            break
        out_name = f"{idx:05d}_{source.stem}.png"
        out_path = out_dir / out_name
        box = process_one(
            source=source,
            output=out_path,
            target_size=args.target_size,
            fill_ratio=args.fill_ratio,
            bg_color=bg_color,
            threshold=args.threshold,
            alpha_threshold=args.alpha_threshold,
            min_area_ratio=args.min_area_ratio,
            bbox_pad=args.bbox_pad,
        )
        rows.append((str(source), str(out_path), box))
        total += 1

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "prepared_image", "bbox"])
        writer.writerows(rows)

    print(f"Prepared {total} inputs -> {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
