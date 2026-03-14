#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
TRAINING_SUFFIX_PATTERNS = [
    re.compile(r"\bvpet\s+vbbe\b"),
    re.compile(r"\bvpet\s+vb\b"),
    re.compile(r"\bvpet\s+alysion\b"),
    re.compile(r"\bvpet\b"),
    re.compile(r"\bforthbattle\b"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive DragonRod sprites that duplicate top-level training sprites by canonicalized name."
    )
    parser.add_argument("--dragonrod-dir", default="training/dragonrod")
    parser.add_argument("--training-dir", default="training")
    parser.add_argument("--archive-dir", default="training/dragonrod_archived_duplicates")
    parser.add_argument("--report", default="training/dragonrod_duplicate_report.json")
    parser.add_argument("--review", default="training/dragonrod_duplicate_review.txt")
    parser.add_argument("--archive-manifest", default="training/dragonrod_archived_duplicates/archive_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    value = value.lower()
    value = value.replace("&", " and ")
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = value.replace("'", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def canonical_training_stem(stem: str) -> str:
    text = normalize_text(stem)
    for pattern in TRAINING_SUFFIX_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w]+", "", text)
    return text


def canonical_dragonrod_stem(stem: str) -> str:
    text = normalize_text(stem)
    text = re.sub(r"[^\w]+", "", text)
    return text


def iter_training_images(training_dir: Path, dragonrod_dir: Path, archive_dir: Path) -> list[Path]:
    images = []
    for path in sorted(training_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if dragonrod_dir in path.parents or archive_dir in path.parents:
            continue
        images.append(path)
    return images


def main() -> None:
    args = parse_args()
    dragonrod_dir = Path(args.dragonrod_dir).resolve()
    training_dir = Path(args.training_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()
    report_path = Path(args.report).resolve()
    review_path = Path(args.review).resolve()
    archive_manifest_path = Path(args.archive_manifest).resolve()

    if not dragonrod_dir.is_dir():
        raise SystemExit(f"DragonRod directory not found: {dragonrod_dir}")
    if not training_dir.is_dir():
        raise SystemExit(f"Training directory not found: {training_dir}")

    archive_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    training_images = iter_training_images(training_dir, dragonrod_dir, archive_dir)
    training_by_key: dict[str, list[str]] = {}
    for path in training_images:
        key = canonical_training_stem(path.stem)
        training_by_key.setdefault(key, []).append(str(path.relative_to(training_dir.parent)))

    dragonrod_images = sorted(path for path in dragonrod_dir.glob("*.png"))
    duplicates: list[dict[str, str | list[str]]] = []
    unique: list[str] = []
    archive_moves: list[dict[str, str]] = []

    for path in dragonrod_images:
        key = canonical_dragonrod_stem(path.stem)
        matches = training_by_key.get(key, [])
        relative = str(path.relative_to(training_dir.parent))
        if matches:
            duplicates.append(
                {
                    "dragonrod": relative,
                    "canonical_key": key,
                    "matching_training": matches,
                }
            )
            destination = archive_dir / path.name
            suffix = 2
            while destination.exists():
                destination = archive_dir / f"{path.stem} ({suffix}){path.suffix}"
                suffix += 1
            archive_moves.append(
                {
                    "from": str(path),
                    "to": str(destination),
                }
            )
        else:
            unique.append(relative)

    if not args.dry_run:
        for move in archive_moves:
            src = Path(move["from"])
            dst = Path(move["to"])
            if src.exists():
                shutil.move(str(src), str(dst))

    report = {
        "dragonrod_png_count": len(dragonrod_images),
        "existing_training_image_count": len(training_images),
        "normalized_name_match_count": len(duplicates),
        "exact_hash_match_count": 0,
        "normalized_name_matches": duplicates,
        "exact_hash_matches": [],
        "unique_by_normalized_name": unique,
        "dry_run": args.dry_run,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    review_lines = [
        f"DragonRod PNGs: {len(dragonrod_images)}",
        f"Existing training images: {len(training_images)}",
        f"Candidate duplicates by normalized name: {len(duplicates)}",
        "Exact pixel duplicates: 0",
        "",
        "Candidate duplicates:",
    ]
    for entry in duplicates:
        review_lines.append(f"- {entry['dragonrod']}")
        for match in entry["matching_training"]:
            review_lines.append(f"  -> {match}")
    review_path.write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    archive_manifest = {
        "dry_run": args.dry_run,
        "archived_count": len(archive_moves),
        "moves": archive_moves,
    }
    if not args.dry_run or not archive_manifest_path.exists():
        archive_manifest_path.write_text(json.dumps(archive_manifest, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "dragonrod_png_count": len(dragonrod_images),
                "training_image_count": len(training_images),
                "duplicate_count": len(duplicates),
                "archived_count": len(archive_moves),
                "dry_run": args.dry_run,
                "report": str(report_path),
                "review": str(review_path),
                "archive_manifest": str(archive_manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
