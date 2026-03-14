#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import ssl
import urllib.request
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image, ImageSequence


PAGES = [
    "https://www.dragonrod-art.com/baby-i",
    "https://www.dragonrod-art.com/baby-ii",
    "https://www.dragonrod-art.com/rookie",
    "https://www.dragonrod-art.com/champion",
    "https://www.dragonrod-art.com/armor",
    "https://www.dragonrod-art.com/ultimate",
    "https://www.dragonrod-art.com/mega",
    "https://www.dragonrod-art.com/no-level",
    "https://www.dragonrod-art.com/appmon-standard",
    "https://www.dragonrod-art.com/appmon-super",
    "https://www.dragonrod-art.com/appmon-ultimate",
    "https://www.dragonrod-art.com/appmon-god",
    "https://www.dragonrod-art.com/digital-lifeforms",
    "https://www.dragonrod-art.com/dim-digimon-database",
    "https://www.dragonrod-art.com/appmon-database",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redownload missing DragonRod sprites using Wix warmup-data inventories."
    )
    parser.add_argument("--output-dir", default="training/dragonrod")
    parser.add_argument("--archive-dir", default="training/dragonrod_archived_duplicates")
    parser.add_argument("--manifest", default="training/dragonrod_wix_redownload_manifest.json")
    parser.add_argument(
        "--skip-label",
        action="append",
        default=[],
        help="Label to skip entirely. Can be passed multiple times.",
    )
    return parser.parse_args()


def sanitize(label: str) -> str:
    label = html.unescape(label).strip()
    label = re.sub(r"[\\/:*?\"<>|]", "_", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def canon(label: str) -> str:
    s = sanitize(label)
    s = re.sub(r"[^\w]+", "", s.lower())
    return s


def fetch_text(url: str, ctx: ssl.SSLContext, headers: dict[str, str]) -> str:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def parse_page_items(url: str, ctx: ssl.SSLContext, headers: dict[str, str]) -> list[dict]:
    text = fetch_text(url, ctx, headers)
    soup = BeautifulSoup(text, "html.parser")
    warmup = soup.find("script", id="wix-warmup-data")
    if warmup is None or not warmup.string:
        raise RuntimeError(f"wix-warmup-data missing: {url}")
    data = json.loads(warmup.string)
    items: list[dict] = []
    for app_data in data.get("appsWarmupData", {}).values():
        if not isinstance(app_data, dict):
            continue
        for key, value in app_data.items():
            if key.endswith("_galleryData") and isinstance(value, dict) and isinstance(value.get("items"), list):
                items.extend(value["items"])
    return items


def save_image_bytes(image_bytes: bytes, out_path: Path) -> None:
    image = Image.open(BytesIO(image_bytes))
    frame = next(ImageSequence.Iterator(image)).convert("RGBA")
    frame.save(out_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()
    manifest_path = Path(args.manifest).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    base_headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }

    existing = {canon(p.stem): str(p) for p in output_dir.glob("*.png")}
    existing.update({canon(p.stem): str(p) for p in archive_dir.glob("*.png")})
    skip_labels = {canon(label) for label in args.skip_label}

    added = []
    skipped_present = []
    skipped_secure = []
    failed = []
    page_summaries = []

    for page_url in PAGES:
        page_headers = dict(base_headers)
        page_headers["Referer"] = page_url
        items = parse_page_items(page_url, ctx, page_headers)
        seen_labels: set[str] = set()
        added_count = 0
        present_count = 0
        secure_count = 0
        failed_count = 0

        for item in items:
            meta = item.get("metaData") or {}
            title = meta.get("title")
            media_url = item.get("mediaUrl") or meta.get("name")
            if not title or not media_url:
                continue

            label = sanitize(title)
            key = canon(label)
            if key in seen_labels:
                continue
            seen_labels.add(key)

            if key in skip_labels:
                secure_count += 1
                skipped_secure.append({"page": page_url, "label": label, "reason": "explicit_skip"})
                continue

            if key in existing:
                present_count += 1
                skipped_present.append({"page": page_url, "label": label, "existing": existing[key]})
                continue

            candidates = [
                f"https://static.wixstatic.com/media/{media_url}",
                f"https://static.wixstatic.com/media/{media_url}/v1/fill/w_{meta.get('width')},h_{meta.get('height')},al_c,q_95,enc_auto/{meta.get('fileName') or label + '.png'}",
            ]

            image_bytes = None
            last_error = None
            for candidate in candidates:
                try:
                    req = urllib.request.Request(candidate, headers=page_headers)
                    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                        image_bytes = resp.read()
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = repr(exc)

            if image_bytes is None:
                failed_count += 1
                failed.append(
                    {
                        "page": page_url,
                        "label": label,
                        "media_url": media_url,
                        "last_error": last_error,
                    }
                )
                continue

            out_path = output_dir / f"{label}.png"
            suffix = 2
            while out_path.exists():
                out_path = output_dir / f"{label} ({suffix}).png"
                suffix += 1

            save_image_bytes(image_bytes, out_path)
            existing[key] = str(out_path)
            added_count += 1
            added.append({"page": page_url, "label": label, "saved": str(out_path), "media_url": media_url})

        page_summaries.append(
            {
                "page": page_url,
                "items_seen": len(seen_labels),
                "added": added_count,
                "already_present": present_count,
                "skipped_or_explicit": secure_count,
                "failed": failed_count,
            }
        )

    summary = {
        "pages": page_summaries,
        "added_count": len(added),
        "already_present_count": len(skipped_present),
        "skipped_secure_count": len(skipped_secure),
        "failed_count": len(failed),
        "added": added,
        "failed": failed,
        "skipped_secure": skipped_secure,
    }
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
