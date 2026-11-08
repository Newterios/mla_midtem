"""Download and cache listing photos from krisha.kz raw ad pages.

Reads data/raw/details_raw.jsonl, fetches up to 5 photo URLs per ad
(host krisha-photos.kcdn.online, public, no login), resizes each image so
its shorter side is 256 px, and stores it as data/photos/<ad_id>_<k>.jpg
(JPEG quality 90, k = 1..5 in page order).

Polite by design: at most 8 concurrent requests, 15 s timeout, 2 retries with
backoff, an honest browser-like User-Agent. Resumable: images already on disk
are not downloaded again (their manifest row is rebuilt from the file).

Writes data/photos/manifest.csv with columns
    ad_id, k, url, status, w, h, sha1
status is "ok", "cached", or a failure reason (http_<code>, timeout,
conn_error, decode_error, ...). w/h are the saved (resized) size, sha1 is of
the saved JPEG bytes.

Photos are never committed or redistributed (see DATA_CARD.md); data/photos/
is gitignored.

Usage:
    python scripts/cache_photos.py [--workers 8] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "details_raw.jsonl"
OUT = ROOT / "data" / "photos"
MANIFEST = OUT / "manifest.csv"

SHORT_SIDE = 256
JPEG_QUALITY = 90
TIMEOUT = 15
RETRIES = 2  # extra attempts after the first one
MAX_PER_AD = 5
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 "
    "(non-commercial university ML course project; low-rate photo cache)"
)
FIELDS = ["ad_id", "k", "url", "status", "w", "h", "sha1"]

log = logging.getLogger("cache_photos")
_tls = threading.local()


def session() -> requests.Session:
    """One keep-alive session per thread (requests.Session is not thread-safe)."""
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept": "image/avif,image/webp,image/jpeg,*/*"})
        s.mount("https://", HTTPAdapter(pool_connections=1, pool_maxsize=1))
        _tls.s = s
    return s


def load_jobs(limit: int | None) -> list[tuple[str, int, str]]:
    jobs = []
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            urls = rec.get("photo_urls") or []
            for k, url in enumerate(urls[:MAX_PER_AD], start=1):
                jobs.append((str(rec["id"]), k, url))
    return jobs[:limit] if limit else jobs


def resize_save(data: bytes, path: Path) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = SHORT_SIDE / min(w, h)
        if scale < 1:
            im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    blob = buf.getvalue()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.replace(path)  # atomic: a crash never leaves a half-written .jpg
    return im.size[0], im.size[1], hashlib.sha1(blob).hexdigest()


def describe_cached(path: Path) -> tuple[int, int, str]:
    blob = path.read_bytes()
    with Image.open(io.BytesIO(blob)) as im:
        w, h = im.size
    return w, h, hashlib.sha1(blob).hexdigest()


def fetch(job: tuple[str, int, str]) -> dict:
    ad_id, k, url = job
    row = {"ad_id": ad_id, "k": k, "url": url, "status": "", "w": "", "h": "", "sha1": ""}
    path = OUT / f"{ad_id}_{k}.jpg"
    if path.exists():
        try:
            row["w"], row["h"], row["sha1"] = describe_cached(path)
            row["status"] = "cached"
            return row
        except (OSError, UnidentifiedImageError):
            path.unlink(missing_ok=True)  # corrupt leftover: download again

    status = "unknown"
    for attempt in range(RETRIES + 1):
        if attempt:
            time.sleep(1.5 * 2 ** (attempt - 1))
        try:
            r = session().get(url, timeout=TIMEOUT)
        except requests.Timeout:
            status = "timeout"
            continue
        except requests.RequestException as e:
            status = f"conn_error:{type(e).__name__}"
            continue
        if r.status_code != 200:
            status = f"http_{r.status_code}"
            if r.status_code in (403, 404, 410):
                break  # permanent, do not hammer
            continue
        try:
            row["w"], row["h"], row["sha1"] = resize_save(r.content, path)
            row["status"] = "ok"
            return row
        except (OSError, UnidentifiedImageError, ValueError):
            status = "decode_error"
            break
    row["status"] = status
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8, help="max concurrent requests (<= 8)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N URLs (smoke test)")
    args = ap.parse_args()
    workers = max(1, min(args.workers, 8))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(args.limit)
    log.info("jobs: %d URLs from %d ads, workers=%d", len(jobs), len({j[0] for j in jobs}), workers)

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if i % 500 == 0 or i == len(jobs):
                bad = sum(r["status"] not in ("ok", "cached") for r in rows)
                rate = i / max(time.time() - t0, 1e-9)
                log.info("progress %d/%d  failed=%d  %.1f url/s", i, len(jobs), bad, rate)

    rows.sort(key=lambda r: (r["ad_id"], r["k"]))
    tmp = MANIFEST.with_suffix(".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(MANIFEST)

    good = [r for r in rows if r["status"] in ("ok", "cached")]
    fails: dict[str, int] = {}
    for r in rows:
        if r["status"] not in ("ok", "cached"):
            fails[r["status"]] = fails.get(r["status"], 0) + 1
    log.info("done in %.0f s: %d images (%d new, %d cached), ads with >=1 photo: %d, failures: %s",
             time.time() - t0, len(good), sum(r["status"] == "ok" for r in rows),
             sum(r["status"] == "cached" for r in rows), len({r["ad_id"] for r in good}), fails or "none")


if __name__ == "__main__":
    main()
