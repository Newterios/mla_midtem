"""
Krisha.kz apartment-sale scraper (Astana + Almaty).
Course project, Machine Learning Algorithms, group SE-2424.
Team: Rakhat Balgabekov, Aitbek Kaiyrbekuly.

Stage 1  (--stage listings): walks the search result pages
          https://krisha.kz/prodazha/kvartiry/<city>/?page=N
          and stores every ad card as one JSON line in data/raw/listings_raw.jsonl
Stage 2  (--stage details):  opens the mobile ad page m.krisha.kz/a/show/<id>
          for every collected id and stores the structured fields, the
          description text and the photo URLs in data/raw/details_raw.jsonl

Raw values are stored exactly as they appear on the page (strings).
All cleaning happens later in the notebook.
Politeness: one request at a time, random pause 0.6-1.2 s, retries with back-off.
robots.txt of krisha.kz (checked 2026-10-04) does not disallow /prodazha/ or /a/show/.
We do not store the seller's name or phone number.
"""
import argparse, json, random, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
(RAW / "html_sample").mkdir(exist_ok=True)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CITIES = {"astana": "Астана", "almaty": "Алматы"}
session = requests.Session()
session.headers.update({"User-Agent": UA, "Accept-Language": "ru-RU,ru;q=0.9"})


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get(url, tries=4):
    for t in range(tries):
        try:
            r = session.get(url, timeout=40)
            if r.status_code == 200:
                return r.text
            log(f"HTTP {r.status_code} for {url} (try {t+1})")
        except requests.RequestException as e:
            log(f"error {e} for {url} (try {t+1})")
        time.sleep(3 * (t + 1) + random.random() * 3)
    return None


def pause():
    time.sleep(0.6 + random.random() * 0.6)


# ---------------------------------------------------------------- stage 1
def parse_card(card, city):
    def txt(sel):
        e = card.select_one(sel)
        return e.get_text(" ", strip=True) if e else None
    stats = [e.get_text(strip=True) for e in card.select(".a-card__stats-item")]
    img = card.select_one("img.a-image__img")
    return {
        "id": card.get("data-id"),
        "city_slug": city,
        "title": txt(".a-card__title"),
        "price_text": txt(".a-card__price"),
        "subtitle": txt(".a-card__subtitle"),
        "preview_text": txt(".a-card__text-preview"),
        "stats": stats,
        "owner_label": txt(".a-card__owner-label"),
        "is_urgent": "is-urgent" in card.get("class", []),
        "is_hot": "a-card--hot" in " ".join(card.get("class", [])),
        "image_url": img.get("src") if img else None,
        "url": "https://krisha.kz/a/show/" + str(card.get("data-id")),
    }


def stage_listings(pages_per_city, target_per_city):
    out = RAW / "listings_raw.jsonl"
    seen = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            seen.add(json.loads(line)["id"])
        log(f"resuming, {len(seen)} ids already stored")
    with out.open("a", encoding="utf-8") as f:
        for city in CITIES:
            n_city = 0
            for page in range(1, pages_per_city + 1):
                url = f"https://krisha.kz/prodazha/kvartiry/{city}/?page={page}"
                html = get(url)
                if html is None:
                    log(f"giving up page {page} of {city}")
                    continue
                if page <= 2:
                    (RAW / "html_sample" / f"listing_{city}_p{page}.html").write_text(html, encoding="utf-8")
                soup = BeautifulSoup(html, "lxml")
                cards = soup.select("div.a-card[data-id]")
                new = 0
                for c in cards:
                    rec = parse_card(c, city)
                    if not rec["id"] or rec["id"] in seen:
                        continue
                    seen.add(rec["id"])
                    rec["scraped_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    rec["page"] = page
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    new += 1
                    n_city += 1
                f.flush()
                log(f"{city} page {page}: {len(cards)} cards, {new} new, city total {n_city}")
                if n_city >= target_per_city:
                    break
                pause()
    log(f"stage 1 done, {len(seen)} unique ids in {out}")


# ---------------------------------------------------------------- stage 2
def parse_detail(html, ad_id):
    soup = BeautifulSoup(html, "lxml")
    rec = {"id": ad_id}
    for item in soup.select("div.offer__info-item"):
        name = item.get("data-name") or "location"
        val = item.select_one(".offer__advert-short-info")
        rec["info." + name] = val.get_text(" ", strip=True) if val else item.get_text(" ", strip=True)
    for dt in soup.select(".offer__parameters dt"):
        dd = dt.find_next_sibling("dd")
        rec["param." + (dt.get("data-name") or dt.get_text(strip=True))] = dd.get_text(" ", strip=True) if dd else None
    m = soup.select_one(".offer__parameters-mortgaged")
    rec["mortgaged_text"] = m.get_text(strip=True) if m else None
    d = soup.select_one(".offer__description .js-description, .offer__description .a-text-white-spaces")
    rec["description"] = d.get_text("\n", strip=True) if d else None
    h1 = soup.select_one("h1")
    rec["h1"] = h1.get_text(" ", strip=True) if h1 else None
    p = soup.select_one(".offer__price")
    rec["price_text"] = p.get_text(" ", strip=True) if p else None
    m = re.search(r"window\.data\s*=\s*(\{.*?\});\s*(?:\n|</script>)", html, re.S)
    if m:
        try:
            adv = json.loads(m.group(1)).get("advert", {}) or {}
            keep = ["price", "hasPrice", "square", "rooms", "userType", "complexId", "storage",
                    "status", "isEstateVerified", "sectionAlias", "categoryAlias", "addressTitle", "title"]
            for k in keep:
                rec["adv." + k] = adv.get(k)
            mp = adv.get("map") or {}
            rec["adv.lat"], rec["adv.lon"] = mp.get("lat"), mp.get("lon")
            addr = adv.get("address") or {}
            for k in ["district", "city", "street", "house_num"]:
                rec["adv.address." + k] = addr.get(k)
            photos = adv.get("photos") or []
            rec["n_photos"] = len(photos)
            rec["photo_urls"] = [ph.get("src") for ph in photos[:5]]
        except Exception as e:  # noqa
            rec["adv.error"] = str(e)
    return rec


def stage_details(limit, shard="0/1"):
    # shards let two polite workers run side by side; parts are merged afterwards
    i, n = (int(x) for x in shard.split("/"))
    src = RAW / "listings_raw.jsonl"
    out = RAW / ("details_raw.jsonl" if n == 1 else f"details_raw.part{i}.jsonl")
    # ads whose price is "от X" (developer "from" prices) have no fixed asking
    # price and are dropped in cleaning anyway, so we do not open them
    ids = []
    for l in src.open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("price_text", "") and r["price_text"].strip().lower().startswith("от"):
            continue
        ids.append(r["id"])
    ids = ids[i::n]
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")}
    todo = [i for i in ids if i not in done][:limit]
    log(f"stage 2: {len(ids)} ids, {len(done)} done, {len(todo)} to fetch")
    n_fail = 0
    with out.open("a", encoding="utf-8") as f:
        for k, ad_id in enumerate(todo, 1):
            html = get(f"https://m.krisha.kz/a/show/{ad_id}")
            if html is None or "offer__info-item" not in html:
                n_fail += 1
                rec = {"id": ad_id, "fetch_error": True}
                if html is not None and "safeline" in html.lower():
                    log("captcha challenge met, sleeping 60 s")
                    time.sleep(60)
            else:
                rec = parse_detail(html, ad_id)
                if k <= 10:
                    (RAW / "html_sample" / f"detail_{ad_id}.html").write_text(html, encoding="utf-8")
            rec["scraped_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0:
                log(f"details {k}/{len(todo)} (fails {n_fail})")
            pause()
    log(f"stage 2 done: {len(todo)} fetched, {n_fail} failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["listings", "details"], required=True)
    ap.add_argument("--pages", type=int, default=140, help="max search pages per city")
    ap.add_argument("--target", type=int, default=2600, help="unique ids per city to stop at")
    ap.add_argument("--limit", type=int, default=10**9, help="max detail pages this run")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th id starting at i")
    a = ap.parse_args()
    meta = RAW / "collection_log.json"
    hist = json.loads(meta.read_text()) if meta.exists() else []
    hist.append({"stage": a.stage, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "args": vars(a), "source": "https://krisha.kz/prodazha/kvartiry/ (astana, almaty)"})
    meta.write_text(json.dumps(hist, indent=2, ensure_ascii=False))
    if a.stage == "listings":
        stage_listings(a.pages, a.target)
    else:
        stage_details(a.limit, a.shard)
