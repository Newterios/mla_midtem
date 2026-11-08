"""Perceptual hashes for cached listing photos.

Reads data/photos/manifest.csv (written by scripts/cache_photos.py) and computes
a 64-bit pHash and dHash (imagehash, hash_size=8) for every cached image.
Writes cache/photo_hashes.csv with columns
    ad_id, k, phash, dhash      (hashes as 16-char hex strings)

Use: near-duplicate photos across ads (same flat relisted, agency stock
photos) -> Hamming distance on these hashes; group-aware CV can then keep
duplicates in the same fold.

Usage:
    python scripts/hash_photos.py [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "data" / "photos"
MANIFEST = PHOTOS / "manifest.csv"
OUT = ROOT / "cache" / "photo_hashes.csv"


def hash_one(key: tuple[str, str]) -> tuple[str, str, str, str]:
    ad_id, k = key
    with Image.open(PHOTOS / f"{ad_id}_{k}.jpg") as im:
        im = im.convert("RGB")
        return ad_id, k, str(imagehash.phash(im)), str(imagehash.dhash(im))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with MANIFEST.open(encoding="utf-8") as f:
        keys = [(r["ad_id"], r["k"]) for r in csv.DictReader(f) if r["status"] in ("ok", "cached")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(hash_one, keys, chunksize=256))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ad_id", "k", "phash", "dhash"])
        w.writerows(rows)
    print(f"hashed {len(rows)} images from {len({r[0] for r in rows})} ads -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
