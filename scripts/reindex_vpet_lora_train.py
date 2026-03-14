#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path


PREFIX_LEN = 6  # 00001_


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reindex datasets/vpet_lora/train pairs into global alphabetical order and repopulate train_images."
    )
    parser.add_argument("--input-dir", default="datasets/vpet_lora/train")
    parser.add_argument("--manifest", default="datasets/vpet_lora/train/manifest.csv")
    parser.add_argument("--dragonrod-manifest", default="datasets/vpet_lora/train/dragonrod_import_manifest.json")
    parser.add_argument("--train-images-dir", default="datasets/vpet_lora/train_images")
    parser.add_argument("--symlink-train-images", action="store_true", help="Populate train_images using symlinks instead of copies.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def logical_stem(path: Path) -> str:
    stem = path.stem
    if len(stem) > PREFIX_LEN and stem[:5].isdigit() and stem[5] == "_":
        return stem[PREFIX_LEN:]
    return stem


def collect_pairs(input_dir: Path) -> list[dict[str, Path | str]]:
    pairs = []
    for image_path in sorted(input_dir.glob("*.png")):
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            raise SystemExit(f"Missing caption for {image_path}")
        pairs.append(
            {
                "image": image_path,
                "caption": caption_path,
                "logical_stem": logical_stem(image_path),
            }
        )
    if not pairs:
        raise SystemExit(f"No PNG pairs found in {input_dir}")
    return pairs


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "prepared_image", "caption"])
        writer.writeheader()
        writer.writerows(rows)


def clean_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    input_dir = Path(args.input_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    dragonrod_manifest_path = Path(args.dragonrod_manifest).resolve()
    train_images_dir = Path(args.train_images_dir).resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    pairs = collect_pairs(input_dir)
    sorted_pairs = sorted(
        pairs,
        key=lambda pair: (str(pair["logical_stem"]).casefold(), str(Path(pair["image"]).name).casefold()),
    )

    rename_map: dict[str, dict[str, str | int]] = {}
    plan = []
    for index, pair in enumerate(sorted_pairs, start=1):
        image_path = Path(pair["image"])
        caption_path = Path(pair["caption"])
        logical = str(pair["logical_stem"])
        new_base = f"{index:05d}_{logical}"
        new_image = input_dir / f"{new_base}.png"
        new_caption = input_dir / f"{new_base}.txt"
        rename_map[str(image_path)] = {
            "index": index,
            "new_image_abs": str(new_image),
            "new_caption_abs": str(new_caption),
            "new_image_rel": str(new_image.relative_to(repo_root)),
        }
        plan.append(
            {
                "old_image": str(image_path),
                "old_caption": str(caption_path),
                "new_image": str(new_image),
                "new_caption": str(new_caption),
            }
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "pair_count": len(sorted_pairs),
                    "train_images_dir": str(train_images_dir),
                    "symlink_train_images": args.symlink_train_images,
                    "sample_moves": plan[:10],
                },
                indent=2,
            )
        )
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="vpet_reindex_", dir=str(input_dir.parent)))
    try:
        for item in plan:
            shutil.move(item["old_image"], temp_dir / Path(item["new_image"]).name)
            shutil.move(item["old_caption"], temp_dir / Path(item["new_caption"]).name)

        for moved in sorted(temp_dir.iterdir()):
            shutil.move(str(moved), str(input_dir / moved.name))
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    manifest_rows = load_manifest_rows(manifest_path)
    updated_rows = []
    for row in manifest_rows:
        prepared_abs = repo_root / row["prepared_image"]
        mapped = rename_map.get(str(prepared_abs))
        if mapped is None:
            raise SystemExit(f"Manifest row not found in rename map: {row['prepared_image']}")
        updated_rows.append(
            {
                "source": row["source"],
                "prepared_image": str(mapped["new_image_rel"]),
                "caption": row["caption"],
            }
        )
    updated_rows.sort(key=lambda row: row["prepared_image"])
    write_manifest_rows(manifest_path, updated_rows)

    if dragonrod_manifest_path.exists():
        dragonrod_manifest = json.loads(dragonrod_manifest_path.read_text(encoding="utf-8"))
        for entry in dragonrod_manifest.get("imports", []):
            mapped = rename_map.get(entry["prepared_image"])
            if mapped is None:
                raise SystemExit(f"DragonRod import entry not found in rename map: {entry['prepared_image']}")
            entry["index"] = int(mapped["index"])
            entry["prepared_image"] = str(mapped["new_image_abs"])
            entry["prepared_caption"] = str(mapped["new_caption_abs"])
            entry["prepared_image_relative"] = str(mapped["new_image_rel"])
        dragonrod_manifest["imports"] = sorted(dragonrod_manifest.get("imports", []), key=lambda entry: entry["index"])
        dragonrod_manifest_path.write_text(json.dumps(dragonrod_manifest, indent=2) + "\n", encoding="utf-8")

    clean_dir(train_images_dir)
    for image_path in sorted(input_dir.glob("*.png")):
        target = train_images_dir / image_path.name
        if args.symlink_train_images:
            target.symlink_to(image_path.resolve())
        else:
            shutil.copy2(image_path, target)

    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "pair_count": len(sorted_pairs),
                "manifest": str(manifest_path),
                "dragonrod_manifest": str(dragonrod_manifest_path),
                "train_images_dir": str(train_images_dir),
                "symlink_train_images": args.symlink_train_images,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
