"""Build the November temporal hold-out: data/raw_2026-11/listings_raw.jsonl -> data/clean/krisha_nov_holdout.csv.

LISTING-LEVEL ONLY. On 8 Nov 2026 krisha.kz answered every ad-page request with a SafeLine bot challenge
(HTTP 403), so the November scrape stopped after the search pages: 1,748 search cards (Astana + Almaty,
8 Nov), no ad pages. The scraper exits on a challenge and is not restarted. This script therefore uses only
the search cards, parsed with the Midterm's card fallback (notebook cell 7, has_detail=False):

  available : id, city, district (card's Russian name -> October transliteration map), price, rooms, area,
              floor / total_floors (title), year_built, building type, condition, bathroom, ceiling height,
              furnished (preview text, when the seller filled them), has_complex, mortgaged, seller_type, is_urgent
  missing   : lat/lon, street/house number, kitchen_area, n_photos, balcony, parking, former_dorm, exchange,
              full description (only the ~250-character preview) and photos (one thumbnail URL)

So the hold-out has fewer fields than the October ad pages (the October CSV has 4,802 of 4,815 ads with an
opened page). It can test only a REDUCED-FEATURE model (one trained on the fields above), or show how much
the full Endterm model degrades when coordinates, kitchen area and photo count are missing (imputed). It
cannot test text/photo/geo blocks of the Final.

Steps: the Midterm cleaning rules (krisha_ml/midterm_cleaning.py, verified to reproduce the October CSV
exactly with --check-october); drop ad ids already in October (raw cards, raw pages, clean CSV); drop
re-listings of an October ad under a new id (same city, card address, rooms, rounded area, floor, price
within 5 %; --keep-reposts keeps them); contact-data scrub (data_collection/scrub_personal_data.py) of the
preview text. Adds one column, posted_after_october (id above the largest October id).
The Endterm coordinate fix is not applied (no coordinates here anyway).

    python scripts/build_nov_holdout.py
    python scripts/build_nov_holdout.py --check-october
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_collection"))

from krisha_ml.midterm_cleaning import (REPOST_KEYS, build_clean, detail_rows, learn_district_map,  # noqa: E402
                                        read_raw, repost_keys)
from scrub_personal_data import scrub, scrub_jsonl  # noqa: E402

OCT_RAW = ROOT / "data" / "raw"
OCT_CLEAN = ROOT / "data" / "clean" / "krisha_apartments_clean.csv"
NOV_RAW = ROOT / "data" / "raw_2026-11"
OUT_CSV = ROOT / "data" / "clean" / "krisha_nov_holdout.csv"
OUT_META = ROOT / "outputs" / "nov_holdout" / "build_meta.json"


def jsonl_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {int(json.loads(line)["id"]) for line in f if line.strip()}


def october_ids() -> set[int]:
    ids = jsonl_ids(OCT_RAW / "listings_raw.jsonl") | jsonl_ids(OCT_RAW / "details_raw.jsonl")
    return ids | set(pd.read_csv(OCT_CLEAN, usecols=["id"])["id"].astype(int))


def october_district_map() -> dict[str, str]:
    listings, details = read_raw(OCT_RAW)
    return learn_district_map(listings, detail_rows(listings, details))


def october_reposts(nov: pd.DataFrame, oct_clean: pd.DataFrame) -> pd.Series:
    """True for a November ad that repeats an October clean ad (Midterm step-3 rule across periods)."""
    o = repost_keys(oct_clean)
    o = o[(o["street_n"] != "") & (o["house_n"] != "")]
    first_oct = o.sort_values("scraped_at").groupby(REPOST_KEYS, dropna=False)["price"].first().rename("oct_price")
    n = repost_keys(nov).join(first_oct, on=REPOST_KEYS)
    hit = n["oct_price"].notna() & ((n["price"] - n["oct_price"]).abs() / n["oct_price"] <= 0.05)
    return (hit & (n["street_n"] != "") & (n["house_n"] != "")).reindex(nov.index).fillna(False).astype(bool)


CARD_KEYS = ["city_slug", "addr_n", "rooms", "area_r", "floor"]


def card_keys(listings: pd.DataFrame) -> pd.DataFrame:
    """Search-card version of the repost key: the card shows the address in Russian ("district, street house
    — landmarks"), so cards of both periods can be compared even when the ad page could not be opened."""
    title = listings["title"].astype("string")
    return pd.DataFrame({
        "id": listings["id"].astype("int64"),
        "city_slug": listings["city_slug"],
        "addr_n": listings["subtitle"].astype("string").str.split(" — ").str[0].str.lower()
                  .str.replace(r"[^\w]", "", regex=True).fillna(""),
        "rooms": title.str.extract(r"(\d+)-комнатн")[0].astype(float),
        "area_r": title.str.extract(r"([\d.,]+)\s*м²")[0].str.replace(",", ".").astype(float).round(0),
        "floor": title.str.extract(r"(\d+)/\d+\s*этаж")[0].astype(float),
        "price": pd.to_numeric(listings["price_text"].astype("string").str.replace(r"[^\d]", "", regex=True), errors="coerce"),
    })


def october_card_reposts(nov_ids: pd.Series, nov_listings: pd.DataFrame, oct_listings: pd.DataFrame) -> pd.Series:
    """True for a November ad whose search card repeats an October card (same city, address, rooms, rounded
    area, floor; price within 5 %): the same flat re-listed under a new id."""
    o = card_keys(oct_listings).dropna(subset=["price"])
    o = o[o["addr_n"] != ""].groupby(CARD_KEYS, dropna=False)["price"].first().rename("oct_price")
    n = card_keys(nov_listings).join(o, on=CARD_KEYS).set_index("id")
    hit = n["oct_price"].notna() & ((n["price"] - n["oct_price"]).abs() / n["oct_price"] <= 0.05) & (n["addr_n"] != "")
    hit = hit[~hit.index.duplicated()]
    return nov_ids.map(hit).fillna(False).astype(bool)


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> list[tuple[str, int]]:
    """Column-wise comparison after a CSV round trip (NaN == NaN, floats with isclose)."""
    b = pd.read_csv(io.StringIO(b.to_csv(index=False)))
    diff = []
    for col in a.columns:
        x, y = a[col], b[col]
        if x.dtype.kind in "fi" and y.dtype.kind in "fi":
            eq = np.isclose(x, y, equal_nan=True)
        else:
            eq = (x.astype(str) == y.astype(str)) | (x.isna() & y.isna())
        if not np.all(eq):
            diff.append((col, int((~np.asarray(eq)).sum())))
    return diff


def check_october() -> dict:
    clean, _, _ = build_clean(OCT_RAW)
    ref = pd.read_csv(OCT_CLEAN)
    same_rows = len(clean) == len(ref) and bool((clean["id"].to_numpy() == ref["id"].to_numpy()).all())
    diff = frames_equal(ref, clean) if same_rows else [("rows", abs(len(clean) - len(ref)))]
    print(f"October reproduction: {len(clean)} rows vs {len(ref)} in the CSV, same ids/order {same_rows}, "
          f"differing columns {diff or 'none'}")
    return {"rows": len(clean), "same_ids_and_order": same_rows, "differing_columns": diff}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", default=str(NOV_RAW))
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--keep-reposts", action="store_true", help="keep re-listings of October ads under a new id")
    ap.add_argument("--scrub-raw", action="store_true", help="also mask contacts in the November raw files in place")
    ap.add_argument("--check-october", action="store_true", help="verify the cleaning reproduces the October CSV")
    a = ap.parse_args()
    raw_dir = ROOT / a.raw_dir          # absolute paths stay absolute
    out_csv = ROOT / a.out

    meta: dict = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "raw_dir": str(raw_dir)}
    if a.check_october:
        meta["october_check"] = check_october()
    if a.scrub_raw:
        meta["raw_scrub"] = {"details": scrub_jsonl(raw_dir / "details_raw.jsonl", ["description", "h1"]),
                             "listings": scrub_jsonl(raw_dir / "listings_raw.jsonl", ["preview_text", "title"])}

    clean, steps, info = build_clean(raw_dir, extra_district_map=october_district_map(), use_details=False)
    meta["level"] = "listing (search cards only; ad pages blocked by SafeLine on 2026-11-08, scraper stopped)"
    meta["fields_missing_vs_october"] = ["lat", "lon", "street", "house_num", "kitchen_area", "n_photos", "balcony", "parking", "former_dorm", "exchange", "full description", "photos"]
    meta.update(info)
    meta["cleaning_steps"] = steps.to_dict("records")
    print(steps.to_string(index=False))

    n0 = len(clean)
    in_oct = clean["id"].isin(october_ids())
    clean = clean[~in_oct].copy()
    oct_clean = pd.read_csv(OCT_CLEAN, parse_dates=["scraped_at"])
    rep_page = october_reposts(clean, oct_clean)
    rep_card = october_card_reposts(clean["id"], read_raw(raw_dir)[0], read_raw(OCT_RAW)[0])
    repost = rep_page | rep_card
    meta["dropped_october_ids"] = int(in_oct.sum())
    meta["october_reposts_new_id"] = int(repost.sum())
    meta["october_reposts_by_rule"] = {"ad_page_address": int(rep_page.sum()), "search_card_address": int(rep_card.sum())}
    if not a.keep_reposts:
        clean = clean[~repost]
    meta["reposts_dropped"] = not a.keep_reposts

    hits, changed, out = {}, 0, []
    for text in clean["description"]:
        t, c = scrub(text)
        out.append(t)
        changed += int(bool(c))
        for k, v in c.items():
            hits[k] = hits.get(k, 0) + v
    clean["description"] = out
    meta["scrub"] = {"descriptions_changed": changed, "masked": hits}

    clean = clean.reset_index(drop=True)
    # the only column the October CSV does not have: True when the ad id is above the largest October id,
    # i.e. the ad was created after our October collection (krisha ids grow with time); False = an older ad
    # that our October crawl (first 160 search pages per city) did not reach
    clean["posted_after_october"] = clean["id"] > max(october_ids())
    meta["posted_after_october"] = int(clean["posted_after_october"].sum())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_csv, index=False)
    meta.update({"rows_after_cleaning": n0, "rows_out": len(clean), "out_csv": str(out_csv),
                 "by_city": clean["city"].value_counts().to_dict(),
                 "has_detail": clean["has_detail"].value_counts().to_dict(),
                 "scraped_at_range": [str(clean["scraped_at"].min()), str(clean["scraped_at"].max())] if len(clean) else None,
                 "price_median_M": round(float(clean["price"].median()) / 1e6, 2) if len(clean) else None})
    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"rows after Midterm cleaning {n0}; dropped {meta['dropped_october_ids']} October ids, "
          f"{meta['october_reposts_new_id']} re-listings of October ads ({'dropped' if not a.keep_reposts else 'kept'}); "
          f"contacts masked in {changed} descriptions {hits or ''}")
    print(f"wrote {len(clean)} rows {meta['by_city']} -> {out_csv}; meta -> {OUT_META.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
