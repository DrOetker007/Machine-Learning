"""Download and cache listing images referenced by the Inside Airbnb data.

The downloader is resumable, deduplicates identical URLs and asks the Airbnb CDN
for a small image width suitable for CLIP preprocessing.  The generated manifest
supports multiple ranked images per listing, although the supplied listings file
currently contains only the main ``picture_url``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from PIL import Image


THREAD_LOCAL = threading.local()
USER_AGENT = "Mozilla/5.0 (compatible; ML-course-image-cache/1.0)"


def thumbnail_url(url: str, width: int) -> str:
    """Add a CDN resize parameter while preserving existing query parameters."""
    parts = urlsplit(url)
    # Raw Inside-Airbnb URLs use /pictures/, which bypasses Airbnb's resizing
    # service.  The /im/pictures/ endpoint honors im_w and avoids multi-MB files.
    path = f"/im{parts.path}" if parts.path.startswith("/pictures/") else parts.path
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["im_w"] = str(width)
    query["im_format"] = "webp"
    return urlunsplit((parts.scheme, parts.netloc, path, urlencode(query), parts.fragment))


def cache_path(image_root: Path, url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return image_root / digest[:2] / f"{digest}.webp"


def session() -> requests.Session:
    if not hasattr(THREAD_LOCAL, "session"):
        current = requests.Session()
        current.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "image/jpeg,image/png,image/webp,*/*;q=0.8",
            }
        )
        THREAD_LOCAL.session = current
    return THREAD_LOCAL.session


def valid_image_bytes(content: bytes) -> None:
    if len(content) < 1_000:
        raise ValueError(f"response too small ({len(content)} bytes)")
    with Image.open(io.BytesIO(content)) as image:
        image.verify()


def download_one(url: str, target: Path, timeout: float, retries: int) -> dict:
    if target.exists() and target.stat().st_size >= 1_000:
        return {"status": "cached", "bytes": target.stat().st_size, "error": ""}

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    last_error = ""
    for attempt in range(retries):
        try:
            response = session().get(url, timeout=timeout)
            response.raise_for_status()
            valid_image_bytes(response.content)
            temporary.write_bytes(response.content)
            os.replace(temporary, target)
            return {"status": "downloaded", "bytes": len(response.content), "error": ""}
        except Exception as exc:  # individual URLs must not stop the full cache
            last_error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    return {"status": "failed", "bytes": 0, "error": last_error[:300]}


def build_manifest(listings_path: Path, image_root: Path, width: int) -> pd.DataFrame:
    listings = pd.read_csv(listings_path, usecols=["id", "picture_url"], low_memory=False)
    manifest = (
        listings.dropna(subset=["picture_url"])
        .rename(columns={"id": "listing_id", "picture_url": "image_url"})
        .copy()
    )
    manifest["image_rank"] = 0
    manifest["image_role"] = "main"
    manifest["download_url"] = manifest["image_url"].astype(str).map(
        lambda value: thumbnail_url(value, width)
    )
    manifest["local_path"] = manifest["download_url"].map(
        lambda value: cache_path(image_root, value).as_posix()
    )
    return manifest.sort_values(["listing_id", "image_rank"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listings", type=Path, default=Path("data/listings.csv.gz"))
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--manifest", type=Path, default=Path("data/image_manifest.csv"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Limit unique URLs for a test run")
    args = parser.parse_args()

    manifest = build_manifest(args.listings, args.image_root, args.width)
    unique = manifest[["download_url", "local_path"]].drop_duplicates("download_url")
    if args.limit is not None:
        unique = unique.head(args.limit)

    print(f"Manifest rows: {len(manifest):,}")
    print(f"Unique images in this run: {len(unique):,}")
    started = time.perf_counter()
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                row.download_url,
                Path(row.local_path),
                args.timeout,
                args.retries,
            ): row.download_url
            for row in unique.itertuples(index=False)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            url = futures[future]
            results[url] = future.result()
            if completed % 250 == 0 or completed == len(futures):
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed else 0.0
                print(f"{completed:,}/{len(futures):,} images ({rate:.1f}/s)")

    # Existing cached files that were outside a limited test run are represented too.
    manifest["download_status"] = manifest["download_url"].map(
        lambda url: results.get(url, {}).get(
            "status", "cached" if Path(cache_path(args.image_root, url)).exists() else "not_attempted"
        )
    )
    manifest["bytes"] = manifest["download_url"].map(
        lambda url: results.get(url, {}).get(
            "bytes", Path(cache_path(args.image_root, url)).stat().st_size
            if Path(cache_path(args.image_root, url)).exists() else 0
        )
    )
    manifest["error"] = manifest["download_url"].map(
        lambda url: results.get(url, {}).get("error", "")
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_suffix(args.manifest.suffix + ".part")
    manifest.to_csv(temporary_manifest, index=False)
    os.replace(temporary_manifest, args.manifest)

    cached = manifest["local_path"].map(lambda value: Path(value).exists()).sum()
    failed = sum(result["status"] == "failed" for result in results.values())
    total_bytes = sum(
        path.stat().st_size
        for path in args.image_root.rglob("*")
        if path.is_file() and path.name != ".gitignore"
    )
    print(f"Locally available manifest rows: {cached:,}/{len(manifest):,}")
    print(f"Failed unique URLs in this run: {failed:,}")
    print(f"Cache size: {total_bytes / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
