#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import ssl
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image, ImageSequence

try:
    from pyppeteer import launch
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "pyppeteer is required for browser-based DragonRod scraping. "
        "Install it with: python3 -m pip install --user pyppeteer"
    ) from exc


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

MEDIA_RE = re.compile(r"https://static\.wixstatic\.com/media/([^/?#]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redownload missing DragonRod sprites by driving the live Wix gallery in a browser."
    )
    parser.add_argument("--output-dir", default="training/dragonrod")
    parser.add_argument("--archive-dir", default="training/dragonrod_archived_duplicates")
    parser.add_argument("--manifest", default="training/dragonrod_browser_redownload_manifest.json")
    parser.add_argument("--page", action="append", default=[], help="Specific page URL to scrape. Can be repeated.")
    parser.add_argument(
        "--skip-label",
        action="append",
        default=[],
        help="Label to skip entirely. Can be passed multiple times.",
    )
    parser.add_argument(
        "--chrome-path",
        default="",
        help="Path to Chrome/Chromium executable. Auto-detected if omitted.",
    )
    parser.add_argument("--viewport-width", type=int, default=1440)
    parser.add_argument("--viewport-height", type=int, default=2200)
    parser.add_argument("--scroll-delay", type=float, default=1.2, help="Seconds to wait after each bottom scroll.")
    parser.add_argument("--stable-rounds", type=int, default=3, help="Stop after N unchanged item counts.")
    parser.add_argument("--max-scroll-rounds", type=int, default=40, help="Hard cap on scroll iterations per page.")
    parser.add_argument("--timeout-ms", type=int, default=120000)
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


def save_image_bytes(image_bytes: bytes, out_path: Path) -> None:
    image = Image.open(BytesIO(image_bytes))
    frame = next(ImageSequence.Iterator(image)).convert("RGBA")
    frame.save(out_path)


def fetch_bytes(url: str, headers: dict[str, str], ctx: ssl.SSLContext) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        return resp.read()


def direct_media_candidates(image_url: str) -> list[str]:
    image_url = image_url.strip()
    if not image_url:
        return []
    match = MEDIA_RE.search(image_url)
    if not match:
        return [image_url]
    media_id = match.group(1)
    candidates = [
        f"https://static.wixstatic.com/media/{media_id}",
        image_url,
    ]
    return list(dict.fromkeys(candidates))


def find_chrome_path(explicit: str) -> str:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return str(path)
        raise SystemExit(f"Chrome path not found: {path}")

    candidates = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise SystemExit(
        "Could not auto-detect Chrome/Chromium. Pass --chrome-path /path/to/browser"
    )


def parse_expected_total(page_html: str) -> int | None:
    soup = BeautifulSoup(page_html, "html.parser")
    warmup = soup.find("script", id="wix-warmup-data")
    if warmup is None or not warmup.string:
        return None
    data = json.loads(warmup.string)
    totals: list[int] = []
    for app_data in data.get("appsWarmupData", {}).values():
        if not isinstance(app_data, dict):
            continue
        for key, value in app_data.items():
            if key.endswith("_galleryData") and isinstance(value, dict):
                total = value.get("totalItemsCount")
                if isinstance(total, int):
                    totals.append(total)
    return max(totals) if totals else None


async def collect_page_items(page, page_url: str, args: argparse.Namespace) -> dict:
    await page.goto(page_url, {"waitUntil": "networkidle2", "timeout": args.timeout_ms})
    expected_total = parse_expected_total(await page.content())

    previous_count = 0
    stable_rounds = 0
    snapshots: list[int] = []

    for _ in range(args.max_scroll_rounds):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(args.scroll_delay)
        count = await page.evaluate(
            """() => document.querySelectorAll('.item-link-wrapper[data-hook="item-link-wrapper"]').length"""
        )
        snapshots.append(count)
        if count == previous_count:
            stable_rounds += 1
        else:
            stable_rounds = 0
        previous_count = count
        if expected_total is not None and count >= expected_total:
            break
        if stable_rounds >= args.stable_rounds:
            break

    items = await page.evaluate(
        """() => Array.from(document.querySelectorAll('.item-link-wrapper[data-hook="item-link-wrapper"]')).map(node => {
            const img = node.querySelector('img');
            const text = (node.textContent || '').replace(/\\s+/g, ' ').trim();
            return {
                idx: node.getAttribute('data-idx'),
                label: text,
                image_url: img ? (img.currentSrc || img.src || '') : '',
            };
        })"""
    )
    return {
        "page": page_url,
        "expected_total": expected_total,
        "snapshots": snapshots,
        "items": items,
    }


async def scrape_pages(args: argparse.Namespace, chrome_path: str) -> list[dict]:
    browser = await launch(
        headless=True,
        executablePath=chrome_path,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    )
    try:
        page = await browser.newPage()
        await page.setViewport({"width": args.viewport_width, "height": args.viewport_height})
        results = []
        for page_url in (args.page or PAGES):
            print(f"[dragonrod-browser] scraping {page_url}", file=sys.stderr)
            result = await collect_page_items(page, page_url, args)
            print(
                f"[dragonrod-browser] {page_url} -> {len(result['items'])} items "
                f"(expected {result['expected_total']})",
                file=sys.stderr,
            )
            results.append(result)
        return results
    finally:
        await browser.close()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    chrome_path = find_chrome_path(args.chrome_path)
    skip_labels = {canon(label) for label in args.skip_label}

    existing = {canon(path.stem): str(path) for path in output_dir.glob("*.png")}
    existing.update({canon(path.stem): str(path) for path in archive_dir.glob("*.png")})

    browser_results = asyncio.run(scrape_pages(args, chrome_path))

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    added = []
    failed = []
    skipped_existing = []
    skipped_explicit = []
    page_summaries = []

    for page_result in browser_results:
        page_url = page_result["page"]
        added_count = 0
        failed_count = 0
        skipped_existing_count = 0
        skipped_explicit_count = 0
        seen_labels: set[str] = set()

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": page_url,
        }

        for item in page_result["items"]:
            label = sanitize(item.get("label", ""))
            image_url = item.get("image_url", "")
            if not label or not image_url:
                continue

            key = canon(label)
            if key in seen_labels:
                continue
            seen_labels.add(key)

            if key in skip_labels:
                skipped_explicit_count += 1
                skipped_explicit.append({"page": page_url, "label": label, "reason": "explicit_skip"})
                continue

            if key in existing:
                skipped_existing_count += 1
                skipped_existing.append({"page": page_url, "label": label, "existing": existing[key]})
                continue

            image_bytes = None
            last_error = None
            for candidate in direct_media_candidates(image_url):
                try:
                    image_bytes = fetch_bytes(candidate, headers=headers, ctx=ctx)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = repr(exc)

            if image_bytes is None:
                failed_count += 1
                failed.append(
                    {
                        "page": page_url,
                        "label": label,
                        "image_url": image_url,
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
            added.append({"page": page_url, "label": label, "saved": str(out_path), "image_url": image_url})

        page_summaries.append(
            {
                "page": page_url,
                "expected_total": page_result["expected_total"],
                "scraped_total": len(seen_labels),
                "scroll_snapshots": page_result["snapshots"],
                "added": added_count,
                "already_present": skipped_existing_count,
                "skipped_explicit": skipped_explicit_count,
                "failed": failed_count,
            }
        )

    summary = {
        "chrome_path": chrome_path,
        "pages": page_summaries,
        "added_count": len(added),
        "already_present_count": len(skipped_existing),
        "skipped_explicit_count": len(skipped_explicit),
        "failed_count": len(failed),
        "added": added,
        "failed": failed,
        "skipped_explicit": skipped_explicit,
    }
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
