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


def make_foreground_mask(image: Image.Image, threshold: int, alpha_threshold: int) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    alpha_mask = alpha.point(lambda px: 255 if px > alpha_threshold else 0)
    alpha_box = alpha_mask.getbbox()
    if alpha_box is not None and alpha_box != (0, 0, rgba.width, rgba.height):
        return alpha_mask

    rgb = rgba.convert("RGB")
    bg_color = estimate_border_color(rgb)
    bg = Image.new("RGB", rgb.size, bg_color)
    diff = ImageChops.difference(rgb, bg).convert("L")
    return diff.point(lambda px: 255 if px > threshold else 0)


def find_foreground_bbox(image: Image.Image, threshold: int, alpha_threshold: int, min_area_ratio: float) -> Tuple[int, int, int, int]:
    mask = make_foreground_mask(image=image, threshold=threshold, alpha_threshold=alpha_threshold)
    box = mask.getbbox()

    if box is None:
        return (0, 0, image.width, image.height)

    bw = box[2] - box[0]
    bh = box[3] - box[1]
    if (bw * bh) / float(image.width * image.height) < min_area_ratio:
        return (0, 0, image.width, image.height)

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


def mean_x(mask: Image.Image, top: int, bottom: int) -> float | None:
    total_x = 0
    total_n = 0
    px = mask.load()
    width, _ = mask.size
    for y in range(top, bottom):
        for x in range(width):
            if px[x, y] > 0:
                total_x += x
                total_n += 1
    if total_n == 0:
        return None
    return total_x / float(total_n)


def side_protrusion(mask: Image.Image, top: int, bottom: int) -> Tuple[float | None, float | None]:
    px = mask.load()
    width, _ = mask.size
    lefts = []
    rights = []
    for y in range(top, bottom):
        row = [x for x in range(width) if px[x, y] > 0]
        if not row:
            continue
        lefts.append(min(row))
        rights.append(max(row))
    if not lefts or not rights:
        return (None, None)
    return (sum(lefts) / float(len(lefts)), sum(rights) / float(len(rights)))


def detect_horizontal_facing(
    crop: Image.Image,
    threshold: int,
    alpha_threshold: int,
    score_threshold: float,
) -> Tuple[str, float]:
    mask = make_foreground_mask(image=crop, threshold=threshold, alpha_threshold=alpha_threshold)
    mask = mask.resize((128, 128), resample=Image.NEAREST)

    split = int(128 * 0.58)
    upper_mean = mean_x(mask, 0, split)
    lower_mean = mean_x(mask, split, 128)
    upper_left, upper_right = side_protrusion(mask, 0, split)
    lower_left, lower_right = side_protrusion(mask, split, 128)

    if None in (upper_mean, lower_mean, upper_left, upper_right, lower_left, lower_right):
        return ("unknown", 0.0)

    centroid_shift = float(upper_mean - lower_mean)
    left_protrusion = float(lower_left - upper_left)
    right_protrusion = float(upper_right - lower_right)
    score = ((0.6 * centroid_shift) + (0.4 * (right_protrusion - left_protrusion))) / 128.0

    if score >= score_threshold:
        return ("right", score)
    if score <= -score_threshold:
        return ("left", score)
    return ("unknown", score)


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
    auto_flip_right_facing: bool,
    flip_score_threshold: float,
) -> Tuple[Tuple[int, int, int, int], str, float, bool]:
    im = Image.open(source).convert("RGBA")
    box = find_foreground_bbox(im, threshold=threshold, alpha_threshold=alpha_threshold, min_area_ratio=min_area_ratio)
    box = pad_box(box, im.size, pad=bbox_pad)
    crop = im.crop(box)
    facing_guess, flip_score = detect_horizontal_facing(
        crop=crop,
        threshold=threshold,
        alpha_threshold=alpha_threshold,
        score_threshold=flip_score_threshold,
    )
    flip_applied = auto_flip_right_facing and facing_guess == "right"
    if flip_applied:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

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

    return (box, facing_guess, flip_score, flip_applied)


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
    parser.add_argument(
        "--auto-flip-right-facing",
        action="store_true",
        help="Flip obviously right-facing inputs so Stage 1 starts closer to canonical left-facing V-Pet pose.",
    )
    parser.add_argument(
        "--flip-score-threshold",
        type=float,
        default=0.06,
        help="Confidence threshold for the simple left/right facing heuristic.",
    )
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
        box, facing_guess, flip_score, flip_applied = process_one(
            source=source,
            output=out_path,
            target_size=args.target_size,
            fill_ratio=args.fill_ratio,
            bg_color=bg_color,
            threshold=args.threshold,
            alpha_threshold=args.alpha_threshold,
            min_area_ratio=args.min_area_ratio,
            bbox_pad=args.bbox_pad,
            auto_flip_right_facing=args.auto_flip_right_facing,
            flip_score_threshold=args.flip_score_threshold,
        )
        rows.append((str(source), str(out_path), box, facing_guess, round(flip_score, 4), flip_applied))
        total += 1

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "prepared_image", "bbox", "facing_guess", "flip_score", "flip_applied"])
        writer.writerows(rows)

    print(f"Prepared {total} inputs -> {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
