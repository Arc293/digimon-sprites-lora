#!/usr/bin/env python3
"""Build a Hugging Face imagefolder-style dataset from image+caption pairs.

Input directory is expected to contain image files with matching .txt files
sharing the same basename.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare imagefolder dataset with metadata.jsonl captions.")
    parser.add_argument("--input-dir", required=True, help="Directory with image and .txt caption pairs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for imagefolder dataset.")
    parser.add_argument("--mode", choices=["symlink", "copy", "hardlink"], default="symlink")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory.")
    parser.add_argument(
        "--default-caption",
        default=None,
        help="Fallback caption when .txt is missing/empty. If omitted, missing captions are treated as errors.",
    )
    return parser.parse_args()


def ensure_clean_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Output exists: {path}. Re-run with --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_images(input_dir: Path) -> list[Path]:
    images = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")
    return images


def read_caption(caption_path: Path, default_caption: str | None) -> str:
    candidates = [caption_path]
    if caption_path.suffix == ".txt":
        candidates.append(caption_path.with_suffix(".TXT"))

    for candidate in candidates:
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8").strip()
        if text:
            return text

    if default_caption is not None:
        return default_caption
    raise RuntimeError(f"Missing or empty caption: {caption_path}")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        dst.hardlink_to(src)
        return
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory not found: {input_dir}")
    if input_dir == output_dir:
        raise RuntimeError("Input and output directories must be different.")

    ensure_clean_output(output_dir, args.overwrite)
    images = collect_images(input_dir)

    metadata_path = output_dir / "metadata.jsonl"
    rows = []

    for image_path in images:
        caption_path = image_path.with_suffix(".txt")
        caption = read_caption(caption_path, args.default_caption)

        out_image = output_dir / image_path.name
        link_or_copy(image_path, out_image, args.mode)

        rows.append({"file_name": image_path.name, "text": caption})

    with metadata_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_images": len(rows),
        "mode": args.mode,
        "metadata": str(metadata_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
