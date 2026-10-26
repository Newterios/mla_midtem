"""Check the free-text fields for personal contact data and mask it (Endterm fix after the Midterm).

The Midterm data card promised that no seller name or phone number is stored.
That is true for the structured fields, but the seller-written `description`
had never been checked.  This script scans the raw snapshot and the cleaned
dataset for phone numbers, e-mail addresses and messenger / social-media
handles, masks them in place with the token below, and prints the counts.
The structured fields, the target and the fixed split are untouched.

Result on our snapshot (31 Oct 2026): krisha.kz already strips phone numbers
from the description text, so no phone number was found; 7 ads contained a
realtor's social-media handle, which is masked.

    .venv/bin/python data_collection/scrub_personal_data.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DETAILS = ROOT / "data/raw/details_raw.jsonl"
RAW_LISTINGS = ROOT / "data/raw/listings_raw.jsonl"
CLEAN = ROOT / "data/clean/krisha_apartments_clean.csv"
TOKEN = "[contact removed]"

PATTERNS = {
    # Kazakh phone numbers as sellers type them: +7 777 123 45 67, 87771234567, 8 (7172) 12-34-56, 7771234567
    "phone": re.compile(r"(?<![\d.,])(?:\+?7|8)?[\s\-\(]*7\d{2}[\s\-\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?![\d.,])"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "handle": re.compile(r"(?<!\w)@\w{3,}"),
    "messenger number": re.compile(r"((?:whats?app|ватсап|вацап|вотсап|телеграм|telegram)\s*[:\-]?\s*)\+?\d[\d\s\-\(\)]{6,}", re.I),
}


def scrub(text):
    """returns (masked text, {pattern: count})"""
    counts = {}
    if not isinstance(text, str):
        return text, counts
    for name, pat in PATTERNS.items():
        if name == "messenger number":
            text, k = pat.subn(lambda m: m.group(1) + TOKEN, text)
        else:
            text, k = pat.subn(TOKEN, text)
        if k:
            counts[name] = k
    return text, counts


def scrub_jsonl(path: Path, fields):
    rows, hits = [], {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for fld in fields:
                if fld in rec:
                    rec[fld], c = scrub(rec[fld])
                    for k, v in c.items():
                        hits[k] = hits.get(k, 0) + v
            rows.append(rec)
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return hits


def main():
    h1 = scrub_jsonl(RAW_DETAILS, ["description", "h1"])
    h2 = scrub_jsonl(RAW_LISTINGS, ["preview_text", "title"])
    df = pd.read_csv(CLEAN)
    hits, changed = {}, 0
    out = []
    for s in df["description"]:
        t, c = scrub(s)
        out.append(t)
        changed += int(bool(c))
        for k, v in c.items():
            hits[k] = hits.get(k, 0) + v
    df["description"] = out
    df.to_csv(CLEAN, index=False)
    print(f"raw ad pages      : masked {h1 or 'nothing'}")
    print(f"raw search cards  : masked {h2 or 'nothing'}")
    print(f"clean csv         : {changed} of {len(df)} descriptions changed; masked {hits or 'nothing'}")


if __name__ == "__main__":
    main()
