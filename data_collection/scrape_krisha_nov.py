"""
Krisha.kz apartment-sale scraper, November temporal hold-out (Astana + Almaty).
Course project, Machine Learning Algorithms, group SE-2424.
Team: Rakhat Balgabekov, Aitbek Kaiyrbekuly.

A copy of data_collection/scrape_krisha.py (the October scraper, left unchanged)
with four additions for the Final's temporal hold-out:

* --raw-dir (default data/raw_2026-11/): every file goes there, so the October
  snapshot in data/raw/ is never touched.
* --exclude-ids (default data/raw/listings_raw.jsonl): ad ids already collected
  in October are treated as seen; only NEW ads are stored. --target counts only
  new ads with a fixed price (the "от X" developer prices are never opened and
  are dropped in cleaning anyway), so the target is the number of ad pages.
* --stage all: search pages first, then the ad pages, in one background run.
* contact data (phones, e-mails, handles) is masked with the scrub of
  data_collection/scrub_personal_data.py before a record is written;
* on the FIRST bot challenge (SafeLine, HTTP 403/468) the run logs it and exits: no session reset,
  no retry, no attempt to get past the challenge;
* no further politeness change: an allow-list of URL paths that
  robots.txt permits, and failed ad pages are retried when the run is resumed.

Stage 1  (listings): https://krisha.kz/prodazha/kvartiry/<city>/?page=N  -> <raw-dir>/listings_raw.jsonl
Stage 2  (details):  https://m.krisha.kz/a/show/<id>                   -> <raw-dir>/details_raw.jsonl

Raw values are stored exactly as they appear on the page (strings); cleaning is
done by scripts/build_nov_holdout.py with the Midterm rules.
Politeness: one request at a time, random pause 0.6-1.2 s, retries with back-off,
clean exit on any bot challenge (HTTP 403/468 SafeLine).
robots.txt of krisha.kz and m.krisha.kz (re-checked 2026-11-08) does not disallow
/prodazha/ or /a/show/; it disallows /ajax/, /a/show-map/, /a/comments, /captcha ...,
none of which is requested. We do not store the seller's name or phone number.

    python data_collection/scrape_krisha_nov.py --stage all --target 750
"""
import argparse, json, random, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrub_personal_data import scrub   # noqa: E402  contact data is masked before anything is written

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_2026-11"          # replaced by --raw-dir in main()

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CITIES = {"astana": "Астана", "almaty": "Алматы"}
ALLOWED_PATHS = ("/prodazha/kvartiry/", "/a/show/")      # robots.txt-permitted paths we use, nothing else
session = requests.Session()
session.headers.update({"User-Agent": UA, "Accept-Language": "ru-RU,ru;q=0.9"})


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def is_challenge(html):
    return html == "CAPTCHA" or (html is not None and "safeline" in html[:20000].lower())


class BotChallenge(Exception):
    """krisha.kz answered with a bot challenge (SafeLine, HTTP 403/468)."""


def stop_on_challenge(url):
    # A bot challenge means the site does not want automated access right now. We do NOT reset the
    # session, change identity or retry to get past it: the run logs the event and exits cleanly.
    log(f"bot challenge (SafeLine) on {url}: the site refuses automated access, exiting; "
        f"collected files are kept and nothing is retried")
    raise BotChallenge(url)


def get(url, tries=4):
    path = urlparse(url).path
    assert path.startswith(ALLOWED_PATHS), f"path not in the robots.txt allow-list: {url}"
    for t in range(tries):
        try:
            r = session.get(url, timeout=40)
            if r.status_code == 200:
                return r.text
            if r.status_code == 468 or (r.status_code == 403 and "safeline" in r.text[:20000].lower()):
                # bot challenge (in November SafeLine answers 403): retrying makes it worse, let the caller back off
                return "CAPTCHA"
            if r.status_code == 404:          # the ad was removed after the search pages were collected
                log(f"HTTP 404 (removed) for {url}")
                return None
            log(f"HTTP {r.status_code} for {url} (try {t+1})")
        except requests.RequestException as e:
            log(f"error {e} for {url} (try {t+1})")
        time.sleep(3 * (t + 1) + random.random() * 3)
    return None


def pause():
    time.sleep(0.6 + random.random() * 0.6)


def is_from_price(price_text):
    return bool(price_text) and price_text.strip().lower().startswith("от")


def read_ids(path):
    if not path or not Path(path).exists():
        return set()
    with Path(path).open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


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


def stage_listings(pages_per_city, target_per_city, exclude_ids):
    out = RAW / "listings_raw.jsonl"
    stored = []
    if out.exists():
        stored = [json.loads(l) for l in out.open(encoding="utf-8") if l.strip()]
        log(f"resuming, {len(stored)} ids already stored")
    seen = {r["id"] for r in stored} | exclude_ids
    log(f"{len(exclude_ids)} October ids excluded")
    with out.open("a", encoding="utf-8") as f:
        for city in CITIES:
            # target = new ads with a fixed price, i.e. ad pages that stage 2 will open
            n_city = sum(1 for r in stored if r["city_slug"] == city and not is_from_price(r.get("price_text")))
            if n_city >= target_per_city:
                log(f"{city}: target already reached ({n_city})")
                continue
            empty_run = 0
            for page in range(1, pages_per_city + 1):
                url = f"https://krisha.kz/prodazha/kvartiry/{city}/?page={page}"
                html = get(url)
                if is_challenge(html):
                    stop_on_challenge(url)
                if html is None:
                    log(f"giving up page {page} of {city}")
                    continue
                if page <= 2:
                    (RAW / "html_sample" / f"listing_{city}_p{page}.html").write_text(html, encoding="utf-8")
                soup = BeautifulSoup(html, "lxml")
                cards = soup.select("div.a-card[data-id]")
                if not cards:
                    empty_run += 1
                    log(f"{city} page {page}: no cards ({empty_run} in a row)")
                    if empty_run >= 3:         # past the last result page or the markup changed
                        break
                    pause()
                    continue
                empty_run = 0
                new = new_fixed = 0
                for c in cards:
                    rec = parse_card(c, city)
                    if not rec["id"] or rec["id"] in seen:
                        continue
                    seen.add(rec["id"])
                    rec["scraped_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    rec["page"] = page
                    for fld in ("title", "preview_text"):
                        rec[fld] = scrub(rec[fld])[0]
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    new += 1
                    if not is_from_price(rec["price_text"]):
                        new_fixed += 1
                        n_city += 1
                f.flush()
                log(f"{city} page {page}: {len(cards)} cards, {new} new ({new_fixed} fixed price), city total {n_city}")
                if n_city >= target_per_city:
                    break
                pause()
    log(f"stage 1 done, {len(seen) - len(exclude_ids)} new ids in {out}")


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
    else:
        rec["adv.error"] = "window.data not found"
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
        if is_from_price(r.get("price_text")):
            continue
        ids.append(r["id"])
    ids = ids[i::n]
    done = set()
    if out.exists():   # a successful page counts as done; failed ones are retried on resume
        done = {r["id"] for r in map(json.loads, out.open(encoding="utf-8")) if not r.get("fetch_error")}
    todo = [a for a in ids if a not in done][:limit]
    log(f"stage 2: {len(ids)} ids, {len(done)} done, {len(todo)} to fetch")
    n_fail = 0
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for k, ad_id in enumerate(todo, 1):
            html = get(f"https://m.krisha.kz/a/show/{ad_id}")
            if is_challenge(html):
                stop_on_challenge(f"https://m.krisha.kz/a/show/{ad_id}")
            if html is None or is_challenge(html) or "offer__info-item" not in html:
                n_fail += 1
                rec = {"id": ad_id, "fetch_error": True}
            else:
                rec = parse_detail(html, ad_id)
                for fld in ("description", "h1"):
                    rec[fld] = scrub(rec[fld])[0]
                if k <= 10:
                    (RAW / "html_sample" / f"detail_{ad_id}.html").write_text(html, encoding="utf-8")
            rec["scraped_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0:
                rate = k / (time.time() - t0) * 60
                log(f"details {k}/{len(todo)} (fails {n_fail}), "
                    f"{rate:.0f} pages/min, ETA {(len(todo) - k) / rate:.0f} min")
            pause()
    log(f"stage 2 done: {len(todo)} fetched, {n_fail} failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["listings", "details", "all"], required=True)
    ap.add_argument("--raw-dir", default="data/raw_2026-11", help="output folder (relative to the repo root)")
    ap.add_argument("--exclude-ids", default="data/raw/listings_raw.jsonl",
                    help="JSONL whose ids are already collected (October); '' to disable")
    ap.add_argument("--pages", type=int, default=200, help="max search pages per city")
    ap.add_argument("--target", type=int, default=750, help="new fixed-price ids per city to stop at")
    ap.add_argument("--limit", type=int, default=10**9, help="max detail pages this run")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th id starting at i")
    a = ap.parse_args()
    RAW = (ROOT / a.raw_dir).resolve()
    if RAW == (ROOT / "data" / "raw").resolve():
        sys.exit("refusing to write into data/raw/ (the October snapshot)")
    (RAW / "html_sample").mkdir(parents=True, exist_ok=True)
    meta = RAW / "collection_log.json"
    hist = json.loads(meta.read_text()) if meta.exists() else []
    hist.append({"stage": a.stage, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "args": vars(a), "source": "https://krisha.kz/prodazha/kvartiry/ (astana, almaty)"})
    meta.write_text(json.dumps(hist, indent=2, ensure_ascii=False))
    try:
        if a.stage in ("listings", "all"):
            excl = read_ids(ROOT / a.exclude_ids) if a.exclude_ids else set()
            stage_listings(a.pages, a.target, excl)
        if a.stage in ("details", "all"):
            stage_details(a.limit, a.shard)
    except BotChallenge:
        sys.exit(0)
