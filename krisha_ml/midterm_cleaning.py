"""The Midterm cleaning (raw krisha.kz pages -> clean ad table), as a reusable module.

Line-for-line port of Midterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb, cells 5 (ad pages), 7 (search-card
fallback for ads whose page was not opened), 9 (category mapping) and 11 (the five cleaning steps).
``build_clean(data/raw)`` reproduces data/clean/krisha_apartments_clean.csv (checked by
scripts/build_nov_holdout.py --check-october and tests/test_midterm_cleaning.py), so the November
hold-out built with the same functions is cleaned exactly like the October data.

The coordinate fix of the Endterm (out-of-city lat/lon -> NaN) is NOT applied here, as in the October
CSV; ``core.load_clean`` / ``core.fix_coordinates`` apply it when the table is loaded for modelling.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CITY_NAMES = {"astana": "Astana", "almaty": "Almaty"}
CLEAN_COLUMNS = [
    "id", "city", "district", "street", "house_num", "lat", "lon", "price", "has_price", "rooms", "area",
    "kitchen_area", "floor", "total_floors", "year_built", "building_type_raw", "condition_raw", "bathroom_raw",
    "ceiling_height", "furnished_raw", "balcony_raw", "parking_raw", "former_dorm_raw", "exchange_raw",
    "has_complex", "mortgaged", "seller_type", "n_photos", "is_urgent", "description", "photo_urls", "scraped_at",
    "has_detail", "building_type", "condition", "bathroom", "furnished", "balcony", "parking", "former_dorm",
    "price_m2", "building_age", "log_area",
]
REPOST_KEYS = ["city", "street_n", "house_n", "rooms", "floor", "area_r"]


# --------------------------------------------------------------------------- raw input
def read_raw(raw_dir: Path | str, use_details: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """listings_raw.jsonl and details_raw.jsonl of one snapshot, read as the Midterm read them.
    use_details=False ignores the ad pages: every ad is parsed from its search card only."""
    raw_dir = Path(raw_dir)
    listings = pd.read_json(raw_dir / "listings_raw.jsonl", lines=True)
    path = raw_dir / "details_raw.jsonl"
    if use_details and path.exists() and path.stat().st_size > 0:
        details = pd.read_json(path, lines=True)
    else:  # no ad page opened yet (stage 2 not started): every ad falls back to its search card
        details = pd.DataFrame({"id": pd.Series(dtype="int64"), "fetch_error": pd.Series(dtype=bool),
                                "scraped_at": pd.Series(dtype="datetime64[ns, UTC]")})
    return listings, details


def _num(s, pat=r"([\d]+(?:[.,]\d+)?)"):
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)  # read_json already typed it
    if not isinstance(s, str):
        return np.nan
    m = re.search(pat, s.replace("\xa0", " "))  # non-breaking space
    return float(m.group(1).replace(",", ".")) if m else np.nan


def _card_num(s, pat):
    m = re.search(pat, s) if isinstance(s, str) else None
    return float(m.group(1).replace(",", ".")) if m else np.nan


def _card_pick(s, pat, group=1):
    m = re.search(pat, s, flags=re.I) if isinstance(s, str) else None
    return m.group(group) if m else None


def _ok_details(details: pd.DataFrame) -> pd.DataFrame:
    err = details["fetch_error"] if "fetch_error" in details.columns else pd.Series(False, index=details.index)
    return details[err.fillna(False) == False].copy()  # noqa: E712  (object column with NaN/True)


def detail_rows(listings: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    """Midterm cell 5: one row per successfully opened ad page."""
    d = _ok_details(details)
    d = d.merge(listings[["id", "city_slug", "owner_label", "is_urgent", "is_hot", "preview_text", "scraped_at"]]
                .rename(columns={"scraped_at": "scraped_at_card"}), on="id", how="left")

    def col(*names):
        # krisha shows the same attribute either in the short-info block ("info.") or the parameters block ("param.")
        out = pd.Series([None] * len(d), index=d.index, dtype=object)
        for n in names:
            if n in d.columns:
                out = out.where(out.notna(), d[n])
        return out

    def raw(name):
        return d[name] if name in d.columns else pd.Series([None] * len(d), index=d.index, dtype=object)

    df = pd.DataFrame({
        "id": d["id"].astype(int),
        "city": d["city_slug"].map(CITY_NAMES),
        "district": raw("adv.address.district").astype("string").str.replace(r"(^r-n_|_r-n$)", "", regex=True).str.replace("_", " "),
        "street": raw("adv.address.street"), "house_num": raw("adv.address.house_num"),
        "lat": pd.to_numeric(raw("adv.lat"), errors="coerce"), "lon": pd.to_numeric(raw("adv.lon"), errors="coerce"),
        "price": pd.to_numeric(raw("adv.price"), errors="coerce"),
        "has_price": raw("adv.hasPrice"),
        "rooms": pd.to_numeric(raw("adv.rooms"), errors="coerce"),
        "area": pd.to_numeric(raw("adv.square"), errors="coerce"),
        "kitchen_area": col("info.live.square").map(lambda s: _num(s, r"кухни\s*[—-]\s*([\d.,]+)")),
        "floor": col("info.flat.floor").map(lambda s: _num(s, r"^\s*(\d+)")),
        "total_floors": col("info.flat.floor").map(lambda s: _num(s, r"из\s*(\d+)")),
        "year_built": col("info.house.year").map(lambda s: _num(s, r"(\d{4})")),
        "building_type_raw": col("info.flat.building", "param.flat.building"),
        "condition_raw": col("info.flat.renovation", "param.flat.renovation"),
        "bathroom_raw": col("info.flat.toilet", "param.flat.toilet"),
        "ceiling_height": col("param.ceiling", "info.ceiling").map(lambda s: _num(s)),
        "furnished_raw": col("param.live.furniture", "info.live.furniture"),
        "balcony_raw": col("param.flat.balcony", "info.flat.balcony"),
        "parking_raw": col("param.flat.parking", "info.flat.parking"),
        "former_dorm_raw": col("param.flat.priv_dorm", "info.flat.priv_dorm"),
        "exchange_raw": col("param.has_change", "info.has_change"),
        "has_complex": col("info.map.complex").notna(),
        "mortgaged": raw("mortgaged_text").notna(),
        "seller_type": raw("adv.userType"),
        "n_photos": pd.to_numeric(raw("n_photos"), errors="coerce"),
        "is_urgent": d["is_urgent"].fillna(False).astype(bool),
        "description": raw("description"),
        "photo_urls": raw("photo_urls"),
        "scraped_at": pd.to_datetime(d["scraped_at"], utc=True),
    })
    df["has_detail"] = True
    return df


def learn_district_map(listings: pd.DataFrame, detail_df: pd.DataFrame) -> dict[str, str]:
    """Russian district name on the search card -> transliterated name on the ad page (mode per name)."""
    both = listings.merge(detail_df[["id", "district"]], on="id")
    both["ru"] = both["subtitle"].astype("string").str.extract(r"^([^,]+?)\s+р-н")[0]
    return {ru: g["district"].mode().iloc[0] for ru, g in both.dropna(subset=["ru", "district"]).groupby("ru")}


def card_rows(listings: pd.DataFrame, detail_df: pd.DataFrame, dist_map: dict[str, str]) -> pd.DataFrame:
    """Midterm cell 7: ads whose page was not opened, parsed from the search card (has_detail=False)."""
    cards = listings[~listings["id"].isin(detail_df["id"])].copy()
    cards = cards[~cards["price_text"].astype("string").str.strip().str.lower().str.startswith("от", na=False)]
    cd = pd.DataFrame({
        "id": cards["id"].astype(int),
        "city": cards["city_slug"].map(CITY_NAMES),
        "district": cards["subtitle"].astype("string").str.extract(r"^([^,]+?)\s+р-н")[0].map(dist_map),
        "street": None, "house_num": None, "lat": np.nan, "lon": np.nan,
        "price": cards["price_text"].astype("string").str.replace(r"[^\d]", "", regex=True).pipe(pd.to_numeric, errors="coerce"),
        "has_price": True,
        "rooms": cards["title"].map(lambda s: _card_num(s, r"(\d+)-комнатн")),
        "area": cards["title"].map(lambda s: _card_num(s, r"([\d.,]+)\s*м²")),
        "kitchen_area": np.nan,
        "floor": cards["title"].map(lambda s: _card_num(s, r"(\d+)/\d+\s*этаж")),
        "total_floors": cards["title"].map(lambda s: _card_num(s, r"\d+/(\d+)\s*этаж")),
        "year_built": cards["preview_text"].map(lambda s: _card_num(s, r"(\d{4})\s*г\.п")),
        "building_type_raw": cards["preview_text"].map(lambda s: _card_pick(s, r"(монолитн\w*|кирпичн\w*|панельн\w*|каркасно[-\w]*|деревянн\w*|иное)\s+дом")),
        "condition_raw": cards["preview_text"].map(lambda s: _card_pick(s, r"состояние:\s*([^,]+)")),
        "bathroom_raw": cards["preview_text"].map(lambda s: _card_pick(s, r"санузел\s+([^,]+)")),
        "ceiling_height": cards["preview_text"].map(lambda s: _card_num(s, r"потолки\s*([\d.,]+)\s*м")),
        "furnished_raw": cards["preview_text"].map(lambda s: _card_pick(s, r"(меблирована\s+[^,]+|без мебели)")),
        "balcony_raw": None, "parking_raw": None, "former_dorm_raw": None, "exchange_raw": None,
        "has_complex": cards["preview_text"].astype("string").str.contains("жил. комплекс", regex=False, na=False),
        "mortgaged": cards["preview_text"].astype("string").str.contains("В залоге", regex=False, na=False),
        "seller_type": cards["owner_label"].map({"Хозяин недвижимости": "owner", "Крыша Агент": "specialist",
                                                 "Специалист": "specialist", "Компания": "company"}),
        "n_photos": np.nan,
        "is_urgent": cards["is_urgent"].fillna(False).astype(bool),
        "description": cards["preview_text"],
        "photo_urls": cards["image_url"].map(lambda u: [u] if isinstance(u, str) else []),
        "scraped_at": pd.to_datetime(cards["scraped_at"], utc=True),
    })
    cd["has_detail"] = False
    return cd


# --------------------------------------------------------------------------- categories (cell 9)
def _map_building(s):
    if not isinstance(s, str):
        return "unknown"
    s = s.lower()
    for k, v in [("монолит", "monolithic"), ("кирпич", "brick"), ("панел", "panel"), ("каркасно", "frame"),
                 ("дерев", "other"), ("иное", "other")]:
        if k in s:
            return v
    return "other"


def _map_condition(s):
    if not isinstance(s, str):
        return "unknown"
    s = s.lower()
    for k, v in [("свежий", "fresh_renovation"), ("евро", "fresh_renovation"), ("хорош", "good"), ("аккуратн", "good"),
                 ("средн", "average"), ("требует", "needs_repair"), ("черн", "rough"), ("свободная", "free_layout"),
                 ("предчистов", "pre_finish")]:
        if k in s:
            return v
    return "other"


def _map_bathroom(s):
    if not isinstance(s, str):
        return "unknown"
    s = s.lower()
    if "2" in s or "два" in s:
        return "two_or_more"
    if "раздель" in s:
        return "separate"
    if "совмещ" in s:
        return "combined"
    return "other"


def _map_furnished(s):
    if not isinstance(s, str):
        return "unknown"
    s = s.lower()
    if "полность" in s:
        return "fully"
    if "частич" in s:
        return "partly"
    if "без" in s or "нет" in s:
        return "no"
    return "other"


def _map_parking(s):
    if not isinstance(s, str):
        return "unknown"
    s = s.lower()
    if "гараж" in s:
        return "garage"
    if "паркинг" in s:
        return "parking"
    if "рядом" in s or "откр" in s:
        return "street"
    if "нет" in s:
        return "none"
    return "other"


def map_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["building_type"] = df["building_type_raw"].map(_map_building)
    df["condition"] = df["condition_raw"].map(_map_condition)
    df["bathroom"] = df["bathroom_raw"].map(_map_bathroom)
    df["furnished"] = df["furnished_raw"].map(_map_furnished)
    df["balcony"] = df["balcony_raw"].map(lambda s: "unknown" if not isinstance(s, str) else ("none" if "нет" in s.lower() else "has"))
    df["parking"] = df["parking_raw"].map(_map_parking)
    df["former_dorm"] = df["former_dorm_raw"].map(lambda s: "unknown" if not isinstance(s, str) else ("yes" if "да" in s.lower() else "no"))
    df["seller_type"] = df["seller_type"].fillna("unknown")
    return df


# --------------------------------------------------------------------------- cleaning steps (cell 11)
def repost_keys(df: pd.DataFrame) -> pd.DataFrame:
    """The near-duplicate key columns of step 3 (normalised street / house number, rounded area)."""
    return df.assign(area_r=df["area"].round(0),
                     street_n=df["street"].astype("string").str.lower().str.replace(r"[^\w]", "", regex=True).fillna(""),
                     house_n=df["house_num"].astype("string").str.lower().str.replace(r"[^\w]", "", regex=True).fillna(""))


def clean_steps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The five Midterm cleaning steps; returns (clean table, step log)."""
    steps = []

    def record(name, before, after):
        steps.append({"step": name, "rows_before": before, "rows_after": after, "removed": before - after})

    n0 = len(df)
    m = df["price"].notna() & (df["price"] > 0) & (df["has_price"] != False)  # noqa: E712
    df1 = df[m].copy(); record("1 numeric fixed price", n0, len(df1))
    df2 = df1.drop_duplicates("id").copy(); record("2 unique listing id", len(df1), len(df2))
    k = repost_keys(df2).sort_values("scraped_at")
    first_price = k.groupby(REPOST_KEYS, dropna=False)["price"].transform("first")
    pos = k.groupby(REPOST_KEYS, dropna=False).cumcount()
    repost = (pos > 0) & ((k["price"] - first_price).abs() / first_price <= 0.05) & (k["street_n"] != "") & (k["house_n"] != "")
    repost = repost.reindex(df2.index).fillna(False).astype(bool)
    df3 = df2[~repost].copy(); record("3 remove near-duplicate reposts", len(df2), len(df3))

    def ok(col, lo, hi):  # a missing value is allowed (imputed later), a wrong one is not
        return df3[col].isna() | df3[col].between(lo, hi)
    m = (df3["area"].between(10, 400) & df3["rooms"].between(1, 8) & ok("floor", 1, 60) & ok("total_floors", 1, 60)
         & ~(df3["floor"] > df3["total_floors"]) & ok("year_built", 1930, 2027) & df3["price"].between(3e6, 1e9))
    df4 = df3[m].copy(); record("4 plausible ranges", len(df3), len(df4))
    bad_k = df4["kitchen_area"] >= df4["area"]; df4.loc[bad_k, "kitchen_area"] = np.nan
    bad_c = ~df4["ceiling_height"].between(2.2, 5); df4.loc[bad_c, "ceiling_height"] = np.nan
    record(f"5 kitchen/ceiling set to NaN ({int(bad_k.sum())}/{int(bad_c.sum())} values)", len(df4), len(df4))

    clean = df4.reset_index(drop=True)
    clean["price_m2"] = clean["price"] / clean["area"]
    clean["building_age"] = 2026 - clean["year_built"]  # the Midterm's reference year, kept for comparability
    clean["log_area"] = np.log(clean["area"])
    return clean[CLEAN_COLUMNS], pd.DataFrame(steps)


def build_clean(raw_dir: Path | str, extra_district_map: dict[str, str] | None = None, use_details: bool = True
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Raw snapshot folder -> (clean table, step log, info). ``extra_district_map`` fills card districts
    that this snapshot's own pages cannot map (used for November with the October map)."""
    listings, details = read_raw(raw_dir, use_details)
    det = detail_rows(listings, details)
    dist_map = learn_district_map(listings, det)
    if extra_district_map:
        dist_map = {**extra_district_map, **dist_map}
    cards = card_rows(listings, det, dist_map)
    df = map_categories(pd.concat([det, cards], ignore_index=True))
    clean, steps = clean_steps(df)
    info = {"listings": len(listings), "detail_lines": len(details), "detail_ok": len(det), "card_only": len(cards),
            "unmapped_card_districts": int(cards["district"].isna().sum()), "district_map_size": len(dist_map)}
    return clean, steps, info
