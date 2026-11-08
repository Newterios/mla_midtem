"""Per-ad table from the raw ad pages (data/raw/details_raw.jsonl).

The clean CSV keeps only the fields used by the Midterm/Endterm. The raw pages also carry the
description, up to five photo URLs and several fields that no model has used yet (flooring, door,
security, furniture, internet, phone, glazed balcony, exchange, mortgage text, complex name...).
krisha.kz shows a field either in the "info" block or the "param" block, so both are coalesced.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .core import ROOT

DETAILS_JSONL = ROOT / "data" / "raw" / "details_raw.jsonl"

# output column -> raw keys in order of preference
RAW_FIELDS = {
    "flooring": ["param.flat.flooring", "info.flat.flooring"],
    "door": ["param.flat.door", "info.flat.door"],
    "security": ["param.flat.security", "info.flat.security"],
    "furniture": ["param.live.furniture", "info.live.furniture"],
    "internet": ["param.inet.type", "info.inet.type"],
    "phone": ["param.flat.phone", "info.flat.phone"],
    "balcony_glazed": ["param.flat.balcony_g", "info.flat.balcony_g"],
    "exchange": ["param.has_change", "info.has_change"],
    "renovation_raw": ["info.flat.renovation"],
    "mortgaged_text": ["mortgaged_text"],
    "complex_name": ["info.map.complex"],
    "complex_id": ["adv.complexId"],
    "estate_verified": ["adv.isEstateVerified"],
    "title": ["h1"],
    "address_title": ["adv.addressTitle"],
    "scraped_at": ["scraped_at"],
}

# --------------------------------------------------------------------------- text helpers
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_HANDLE = re.compile(r"(?<!\w)@\w{2,}")
_PHONE = re.compile(r"(?<![\d.,])(?:\+?7|8)?[\s\-(]*7\d{2}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?![\d.,])")
_CONTACT_MASK = re.compile(r"\[contact removed\]", re.I)
_SENTENCE_END = re.compile(r"[.!?\n]")
_MONEY = re.compile(r"цен[ауые]|стоимост|миллион|млн|тысяч|тыс\b|тенге|kzt|₸|\$|доллар|тг\b", re.I)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_NON_LETTER = re.compile(r"[^a-zа-яёәғқңөұүһі\s]")
_SPACES = re.compile(r"\s+")


def mask_contacts(text: str | None) -> str:
    """Replace phones, e-mails, @handles and URLs with '[contact removed]' (same mask as the clean CSV)."""
    text = str(text or "")
    for rx in (_URL, _EMAIL, _HANDLE, _PHONE):
        text = rx.sub("[contact removed]", text)
    return text


def sanitize_text(text: str | None) -> str:
    """Model-ready description: lower case, no URLs/contacts, no sentence that mentions a price or
    money (it would leak the target), no digits, letters only (Russian/Kazakh/Latin), single spaces."""
    text = str(text or "").lower()
    for rx in (_URL, _EMAIL, _HANDLE, _PHONE, _CONTACT_MASK):
        text = rx.sub(" ", text)
    text = " ".join(part for part in _SENTENCE_END.split(text) if not _MONEY.search(part))
    text = _NUMBER.sub(" ", text)
    text = _NON_LETTER.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


# --------------------------------------------------------------------------- loader
def _iter_records(path: Path):
    """Yield JSON records line by line (binary read: no line-length limit); count broken lines."""
    bad = 0
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                bad += 1
    if bad:
        print(f"[krisha_ml.raw] skipped {bad} unreadable lines in {path.name}")


def _first(rec: dict, keys: list[str]):
    for k in keys:
        v = rec.get(k)
        if v is not None and v != "":
            return v
    return None


@lru_cache(maxsize=1)
def _load_cached(path: str) -> pd.DataFrame:
    rows = {}
    for rec in _iter_records(Path(path)):
        try:
            ad_id = int(rec["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if ad_id in rows and rec.get("fetch_error"):
            continue  # keep a successful fetch over a failed re-fetch
        urls = rec.get("photo_urls") or []
        row = {"id": ad_id,
               "fetch_ok": not rec.get("fetch_error"),
               "description": mask_contacts(rec.get("description")),
               "photo_urls": [u for u in urls if isinstance(u, str)],
               "n_photos_listed": rec.get("n_photos")}
        row.update({col: _first(rec, keys) for col, keys in RAW_FIELDS.items()})
        rows[ad_id] = row
    df = pd.DataFrame(list(rows.values()))
    df["n_photo_urls"] = df["photo_urls"].map(len).astype("int16")
    df["complex_id"] = pd.to_numeric(df["complex_id"], errors="coerce").astype("Int64")
    df["estate_verified"] = df["estate_verified"].map({True: 1, False: 0}).astype("float64")
    df["description_sanitized"] = df["description"].map(sanitize_text)
    return df.sort_values("id").reset_index(drop=True)


def load_raw_details(path: Path | str = DETAILS_JSONL) -> pd.DataFrame:
    """One row per ad id found in the raw pages (4,822 rows; 13 failed fetches have ``fetch_ok=False``
    and empty fields). Columns: id, fetch_ok, description (contacts masked), description_sanitized,
    photo_urls (list, <= 5), n_photo_urls, n_photos_listed, and the raw fields in ``RAW_FIELDS``
    (strings as shown on krisha.kz, None when the seller left the field empty).
    Join to model rows with ``krisha_ml.core.join_features(train.ids, raw)``."""
    return _load_cached(str(path)).copy()


def multi_hot(series: pd.Series, prefix: str, sep: str = ",", min_count: int = 1) -> pd.DataFrame:
    """Split comma-separated raw values (e.g. ``security``: 'решетки на окнах, охрана') into 0/1 columns.
    Values seen fewer than ``min_count`` times are dropped. Missing -> all zeros."""
    items = series.fillna("").astype(str).str.lower().str.split(sep).map(lambda xs: {x.strip() for x in xs if x.strip()})
    counts = pd.Series([x for s in items for x in s]).value_counts()
    keep = sorted(counts[counts >= min_count].index)
    data = np.array([[v in s for v in keep] for s in items], dtype="int8").reshape(len(items), len(keep))
    return pd.DataFrame(data, index=series.index, columns=[f"{prefix}{re.sub(r'\W+', '_', v).strip('_')}" for v in keep])
