#!/usr/bin/env python3
"""Normalize DragonRod sprites and import them into the SDXL LoRA pair dataset.

The output format matches the existing `datasets/vpet_lora/train` convention:
- `NNNNN_<name>.png`
- `NNNNN_<name>.txt`
- shared `manifest.csv`

The script is rerun-safe. It stores a manifest of imported source files and
reuses the previous output mapping when possible instead of generating
duplicate entries.
"""

from __future__ import annotations

import argparse
import csv
import json
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

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
INDEXED_NAME_RE = re.compile(r"^(?P<index>\d{5})_(?P<stem>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import normalized DragonRod sprites into datasets/vpet_lora/train."
    )
    parser.add_argument("--source-dir", default="training/dragonrod", help="Directory of cleaned DragonRod sprites.")
    parser.add_argument("--output-dir", default="datasets/vpet_lora/train", help="Output pair dataset directory.")
    parser.add_argument(
        "--import-manifest",
        default=None,
        help="Optional import manifest path. Defaults to <output-dir>/dragonrod_import_manifest.json.",
    )
    parser.add_argument(
        "--dataset-manifest",
        default=None,
        help="Optional dataset manifest CSV path. Defaults to <output-dir>/manifest.csv.",
    )
    parser.add_argument("--target-size", type=int, default=512, help="Square output size in pixels.")
    parser.add_argument("--sprite-fill", type=float, default=0.84, help="Sprite max fill ratio within canvas.")
    parser.add_argument("--alpha-threshold", type=int, default=8, help="Alpha threshold for bounding box.")
    parser.add_argument("--bbox-pad", type=int, default=4, help="Padding around detected sprite bbox.")
    parser.add_argument("--bg-color", default="f2f4f8", help="Canvas background hex color, e.g. f2f4f8.")
    parser.add_argument("--style-trigger-token", default="vpet_style", help="Style trigger token used in captions.")
    parser.add_argument("--pose-trigger-token", default="vpet_left_pose", help="Pose trigger token used in captions.")
    parser.add_argument(
        "--pose-description",
        default="partial left-facing, three-quarter view",
        help="Canonical pose wording to include in captions.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only N images (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported without writing files.")
    return parser.parse_args()


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
    sprite = sprite.resize((new_w, new_h), resample=Image.NEAREST)

    canvas = Image.new("RGBA", (target_size, target_size), bg_color + (255,))
    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2
    canvas.alpha_composite(sprite, (x, y))
    canvas.convert("RGB").save(out_path, format="PNG", optimize=True)


def make_caption(name: str, style_trigger_token: str, pose_trigger_token: str, pose_description: str) -> str:
    return (
        f"{style_trigger_token}, {pose_trigger_token}, digimon, {name}, full body, {pose_description}, "
        "sprite, pixel art, clean outline, limited color palette"
    )


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def detect_next_index(output_dir: Path) -> int:
    highest = 0
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = INDEXED_NAME_RE.match(path.stem)
        if match:
            highest = max(highest, int(match.group("index")))
    return highest + 1


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_dataset_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_dataset_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "prepared_image", "caption"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    import_manifest_path = Path(args.import_manifest).resolve() if args.import_manifest else output_dir / "dragonrod_import_manifest.json"
    dataset_manifest_path = Path(args.dataset_manifest).resolve() if args.dataset_manifest else output_dir / "manifest.csv"

    if not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")

    ensure_parent(import_manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    bg_color = parse_hex_color(args.bg_color)
    repo_root = Path.cwd().resolve()

    state = load_json(import_manifest_path) or {}
    previous_imports = state.get("imports", [])
    imports_by_source = {entry["source"]: entry for entry in previous_imports}

    dataset_rows = load_dataset_manifest(dataset_manifest_path)
    dataset_rows_by_prepared = {row["prepared_image"]: row for row in dataset_rows if row.get("prepared_image")}

    current_sources = {str(path.resolve()) for path in iter_images(source_dir)}
    stale_imports = [entry for entry in previous_imports if entry.get("source") not in current_sources]
    stale_prepared_images = {entry.get("prepared_image_relative") for entry in stale_imports if entry.get("prepared_image_relative")}

    for entry in stale_imports:
        prepared_image = Path(entry["prepared_image"])
        prepared_caption = Path(entry["prepared_caption"])
        if not args.dry_run:
            if prepared_image.exists():
                prepared_image.unlink()
            if prepared_caption.exists():
                prepared_caption.unlink()

    for prepared_rel in stale_prepared_images:
        dataset_rows_by_prepared.pop(prepared_rel, None)

    next_index = detect_next_index(output_dir)
    processed = 0
    created = 0
    reused = 0
    new_dataset_rows: list[dict[str, str]] = []
    updated_imports: list[dict[str, str]] = []

    for ordinal, source in enumerate(iter_images(source_dir), start=1):
        if args.limit > 0 and ordinal > args.limit:
            break

        source_str = str(source)
        existing = imports_by_source.get(source_str)
        if existing:
            prepared_image = Path(existing["prepared_image"])
            prepared_caption = Path(existing["prepared_caption"])
            if prepared_image.exists() and prepared_caption.exists():
                reused += 1
                updated_imports.append(existing)
                processed += 1
                continue

        output_base = f"{next_index:05d}_{source.stem}"
        out_png = output_dir / f"{output_base}.png"
        out_txt = output_dir / f"{output_base}.txt"

        name = clean_name(source.stem)
        caption = make_caption(
            name=name,
            style_trigger_token=args.style_trigger_token,
            pose_trigger_token=args.pose_trigger_token,
            pose_description=args.pose_description,
        )

        if not args.dry_run:
            process_image(
                source_path=source,
                out_path=out_png,
                target_size=args.target_size,
                sprite_fill=args.sprite_fill,
                alpha_threshold=args.alpha_threshold,
                bbox_pad=args.bbox_pad,
                bg_color=bg_color,
            )
            out_txt.write_text(caption + "\n", encoding="utf-8")

        prepared_rel = str(out_png.relative_to(repo_root))
        source_rel = str(source.relative_to(repo_root))
        row = {
            "source": source_rel,
            "prepared_image": prepared_rel,
            "caption": caption,
        }
        new_dataset_rows.append(row)

        updated_imports.append(
            {
                "source": source_str,
                "source_relative": source_rel,
                "prepared_image": str(out_png),
                "prepared_caption": str(out_txt),
                "prepared_image_relative": prepared_rel,
                "caption": caption,
                "index": next_index,
            }
        )

        created += 1
        processed += 1
        next_index += 1

    if not args.dry_run:
        for row in new_dataset_rows:
            dataset_rows_by_prepared[row["prepared_image"]] = row
        merged_rows = sorted(
            dataset_rows_by_prepared.values(),
            key=lambda row: row["prepared_image"],
        )
        write_dataset_manifest(dataset_manifest_path, merged_rows)

        state = {
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "target_size": args.target_size,
            "sprite_fill": args.sprite_fill,
            "alpha_threshold": args.alpha_threshold,
            "bbox_pad": args.bbox_pad,
            "bg_color": args.bg_color,
            "style_trigger_token": args.style_trigger_token,
            "pose_trigger_token": args.pose_trigger_token,
            "pose_description": args.pose_description,
            "imports": sorted(updated_imports, key=lambda entry: entry["index"]),
        }
        import_manifest_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "processed": processed,
        "created": created,
        "reused_existing_imports": reused,
        "pruned_stale_imports": len(stale_imports),
        "next_available_index": next_index,
        "import_manifest": str(import_manifest_path),
        "dataset_manifest": str(dataset_manifest_path),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
