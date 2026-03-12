#!/usr/bin/env python3
"""Prepare an image-only Kaggle upload bundle for VPet LoRA training."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Kaggle-ready image-only training bundle.")
    parser.add_argument("--input-dir", default="datasets/vpet_lora/train_images", help="Input image folder.")
    parser.add_argument(
        "--output-dir",
        default="dist/kaggle/vpet_lora_train_images/train_images",
        help="Output folder (train_images subdir recommended).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output-dir before writing.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise RuntimeError(f"Input folder not found: {input_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Output already exists: {output_dir}. Re-run with --overwrite to recreate."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir.parent / "manifest.csv"
    readme_path = output_dir.parent / "README.txt"

    rows = []
    for src in iter_images(input_dir):
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        rows.append((src.name, str(dst.relative_to(output_dir.parent))))

    if not rows:
        raise RuntimeError(f"No images found in: {input_dir}")

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "bundle_path"])
        writer.writerows(rows)

    readme_path.write_text(
        "\n".join(
            [
                "VPet LoRA training bundle for Kaggle",
                f"image_count={len(rows)}",
                "contents:",
                "- train_images/*.png",
                "- manifest.csv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Copied {len(rows)} images")
    print(f"Bundle root: {output_dir.parent}")
    print(f"Upload this folder to Kaggle as a dataset input")


if __name__ == "__main__":
    main()
