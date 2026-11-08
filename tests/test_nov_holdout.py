"""Checks of the November hold-out pipeline (about 10 s, no network).

Run: python -m pytest -q tests/test_nov_holdout.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_collection"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_nov_holdout as bnh  # noqa: E402
import scrape_krisha_nov as scraper  # noqa: E402
from krisha_ml.midterm_cleaning import build_clean, clean_steps, map_categories  # noqa: E402

SAMPLES = ROOT / "data" / "raw" / "html_sample"


def test_midterm_cleaning_reproduces_october_csv():
    check = bnh.check_october()
    assert check["rows"] == 4815 and check["same_ids_and_order"] and check["differing_columns"] == []


def test_parse_detail_on_saved_october_page():
    page = sorted(SAMPLES.glob("detail_*.html"))[0]
    rec = scraper.parse_detail(page.read_text(encoding="utf-8"), page.stem.split("_")[1])
    assert "adv.error" not in rec, rec.get("adv.error")
    assert rec["adv.price"] > 0 and rec["adv.square"] > 0 and rec["adv.rooms"] >= 1
    assert rec["adv.lat"] and rec["adv.lon"] and len(rec["photo_urls"]) <= 5
    assert any(k.startswith("info.") for k in rec)


def test_parse_card_on_saved_listing_page():
    page = next(SAMPLES.glob("listing_*_p1.html"))
    cards = BeautifulSoup(page.read_text(encoding="utf-8"), "lxml").select("div.a-card[data-id]")
    assert len(cards) >= 15
    recs = [scraper.parse_card(c, "astana") for c in cards]
    assert all(r["id"] and r["price_text"] and "м²" in r["title"] for r in recs)


def test_robots_allow_list_blocks_disallowed_paths():
    with pytest.raises(AssertionError):
        scraper.get("https://krisha.kz/ajax/anything")
    with pytest.raises(AssertionError):
        scraper.get("https://krisha.kz/a/show-map/123")


def _row(**kw):
    base = dict(id=1, city="Astana", district="Esil", street="Kabanbay", house_num="1", lat=51.1, lon=71.4,
                price=30e6, has_price=True, rooms=2.0, area=55.0, kitchen_area=10.0, floor=3.0, total_floors=9.0,
                year_built=2015.0, building_type_raw="панельный", condition_raw="хорошее", bathroom_raw="раздельный",
                ceiling_height=2.7, furnished_raw=None, balcony_raw="балкон", parking_raw=None, former_dorm_raw="нет",
                exchange_raw=None, has_complex=False, mortgaged=False, seller_type="owner", n_photos=10.0,
                is_urgent=False, description="", photo_urls=[], scraped_at=pd.Timestamp("2026-11-08", tz="UTC"),
                has_detail=True)
    base.update(kw)
    return base


def test_clean_steps_rules():
    df = map_categories(pd.DataFrame([
        _row(id=1),
        _row(id=1),                                     # duplicate id
        _row(id=2, price=30.5e6, scraped_at=pd.Timestamp("2026-11-09", tz="UTC")),  # repost of id 1 (+1.7 %)
        _row(id=3, area=500.0),                         # implausible area
        _row(id=4, price=None),                         # no price
        _row(id=5, kitchen_area=80.0, ceiling_height=9.0, street="Other"),
    ]))
    clean, steps = clean_steps(df)
    assert clean["id"].tolist() == [1, 5]
    assert steps["removed"].tolist() == [1, 1, 1, 1, 0]
    assert np.isnan(clean.loc[1, "kitchen_area"]) and np.isnan(clean.loc[1, "ceiling_height"])


def test_cross_period_repost_flag():
    oct_ = pd.DataFrame([_row(id=10, scraped_at=pd.Timestamp("2026-10-05", tz="UTC"))])
    nov = pd.DataFrame([_row(id=20, price=31e6), _row(id=21, price=40e6), _row(id=22, street=None, house_num=None)])
    assert bnh.october_reposts(nov, oct_).tolist() == [True, False, False]


def test_build_on_card_only_snapshot(tmp_path):
    # a snapshot whose ad pages were all blocked still cleans: every ad falls back to its search card
    src = ROOT / "data" / "raw" / "listings_raw.jsonl"
    lines = src.read_text(encoding="utf-8").splitlines()[:60]
    (tmp_path / "listings_raw.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "details_raw.jsonl").write_text('{"id": "1", "fetch_error": true, "scraped_at": "2026-11-08T10:00:00+00:00"}\n')
    clean, _, info = build_clean(tmp_path, extra_district_map=bnh.october_district_map())
    assert info["detail_ok"] == 0 and len(clean) > 0 and (~clean["has_detail"]).all()
    assert clean["district"].notna().mean() > 0.9 and clean["price"].gt(0).all()  # cards without "р-н" stay NaN, as in the Midterm


def test_cross_period_card_repost_flag():
    card = dict(city_slug="astana", title="2-комнатная квартира · 55.2 м² · 3/9 этаж",
                subtitle="Есильский р-н, Кабанбай батыра 1 — Хан Шатыр", price_text="30 000 000 ₸")
    oct_ = pd.DataFrame([dict(card, id=10)])
    nov = pd.DataFrame([dict(card, id=20, price_text="31 000 000 ₸"),                  # same flat, +3.3 %
                        dict(card, id=21, price_text="40 000 000 ₸"),                  # price moved too much
                        dict(card, id=22, title="2-комнатная квартира · 55.2 м² · 4/9 этаж")])  # other floor
    flags = bnh.october_card_reposts(pd.Series([20, 21, 22]), nov, oct_)
    assert flags.tolist() == [True, False, False]
