#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class CleanupConfig:
    white_threshold: int
    warm_blue_threshold: int
    alpha_background_threshold: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove border-connected white or off-white halo pixels from sprite PNGs "
            "while leaving interior highlights intact."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("training/dragonrod"),
        help="Directory containing sprite PNGs to normalize.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training/dragonrod_outline_cleanup_manifest.json"),
        help="Path to write the cleanup manifest JSON.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=235,
        help="Minimum red and green channel value for a pixel to count as near-white.",
    )
    parser.add_argument(
        "--warm-blue-threshold",
        type=int,
        default=220,
        help="Minimum blue channel value for warm off-white pixels such as 255,255,240.",
    )
    parser.add_argument(
        "--alpha-background-threshold",
        type=int,
        default=16,
        help="Alpha threshold below which a pixel is treated as transparent background.",
    )
    return parser.parse_args()


def is_candidate_background(pixel: tuple[int, int, int, int], cfg: CleanupConfig) -> bool:
    r, g, b, a = pixel
    if a <= cfg.alpha_background_threshold:
        return True
    return r >= cfg.white_threshold and g >= cfg.white_threshold and b >= cfg.warm_blue_threshold


def cleanup_image(path: Path, cfg: CleanupConfig) -> dict[str, int | str | bool]:
    image = Image.open(path).convert("RGBA")
    pixels = image.load()
    width, height = image.size

    queue: deque[tuple[int, int]] = deque()
    visited: set[tuple[int, int]] = set()

    def enqueue_if_background(x: int, y: int) -> None:
        if (x, y) in visited:
            return
        if not is_candidate_background(pixels[x, y], cfg):
            return
        visited.add((x, y))
        queue.append((x, y))

    for x in range(width):
        enqueue_if_background(x, 0)
        enqueue_if_background(x, height - 1)
    for y in range(height):
        enqueue_if_background(0, y)
        enqueue_if_background(width - 1, y)

    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height:
                enqueue_if_background(nx, ny)

    removed_pixels = 0
    for x, y in visited:
        r, g, b, a = pixels[x, y]
        if a > cfg.alpha_background_threshold and is_candidate_background((r, g, b, a), cfg):
            pixels[x, y] = (r, g, b, 0)
            removed_pixels += 1

    changed = removed_pixels > 0
    if changed:
        image.save(path)

    return {
        "file": str(path),
        "width": width,
        "height": height,
        "changed": changed,
        "removed_pixels": removed_pixels,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    cfg = CleanupConfig(
        white_threshold=args.white_threshold,
        warm_blue_threshold=args.warm_blue_threshold,
        alpha_background_threshold=args.alpha_background_threshold,
    )

    results = []
    total_removed = 0
    changed_files = 0
    png_files = sorted(input_dir.glob("*.png"))

    for png_path in png_files:
        result = cleanup_image(png_path, cfg)
        results.append(result)
        total_removed += int(result["removed_pixels"])
        if result["changed"]:
            changed_files += 1

    manifest = {
        "input_dir": str(input_dir),
        "file_count": len(png_files),
        "changed_files": changed_files,
        "total_removed_pixels": total_removed,
        "config": {
            "white_threshold": cfg.white_threshold,
            "warm_blue_threshold": cfg.warm_blue_threshold,
            "alpha_background_threshold": cfg.alpha_background_threshold,
        },
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        json.dumps(
            {
                "file_count": len(png_files),
                "changed_files": changed_files,
                "total_removed_pixels": total_removed,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
