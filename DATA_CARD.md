# Data card — krisha.kz apartment sale listings, Astana and Almaty

**Team:** Rakhat Balgabekov, Aitbek Kaiyrbekuly (SE-2424) · **Collected:** search pages and the first half of the ad pages on 4 October 2026, the second half of the ad pages on 8 October 2026 (first request 2026-10-04T17:02:20+00:00 UTC; every raw row carries its own `scraped_at`)

| Item | Value |
|---|---|
| Source | krisha.kz → *Продажа квартир* → Astana (`/prodazha/kvartiry/astana/`) and Almaty (`/prodazha/kvartiry/almaty/`) |
| One row | one apartment-for-sale advertisement (one listing id) |
| Additional sources named at registration (olx.kz, kn.kz) | not used: krisha.kz alone exceeded the promised 3,000 rows; the cross-site matching rule is applied within krisha.kz to remove reposts |
| Collection code | `data_collection/scrape_krisha.py` — stage 1: search result pages (20 ads/page) → ad cards; stage 2: mobile ad page `m.krisha.kz/a/show/<id>` → structured fields, description, photo URLs. Single-threaded, 0.6–1.2 s pause, retries with back-off. |
| Raw snapshot | `data/raw/listings_raw.jsonl` (5,810 cards), `data/raw/details_raw.jsonl` (4,802 ad pages), raw HTML samples in `data/raw/html_sample/`, run log in `data/raw/collection_log.json` |
| Rows | 4,822 ads (4,802 with the full ad page, 13 from the search card only) → **4,815 rows after cleaning** (2,503 Astana, 2,312 Almaty) |
| Card-only rows | Stage 2 ran in two halves: on 4 October the site switched on a bot challenge (HTTP 468) for new sessions, so the second half was collected on 8 October after it had lifted. The 13 ads removed from the site in between fall back to their search card (rooms, area, floor, district, price, preview line with building type, year, condition, ceiling, bathroom, furnishing); coordinates, kitchen area, parking, balcony and photo count are missing for them. `has_detail` marks the source of each row. |
| Text / images | `description` (seller's text) and `photo_urls` — reserved for the Final |
| Source check | `robots.txt` (4 Oct 2026) disallows only service paths (`/ajax/`, `/comments`, `/claim`, map/search-parameter URLs); listing pages and ad pages are allowed; no login needed. The data describe objects (apartments), not people: seller name, phone and account id are **not** stored. Developer ads with a "от …" price were not opened (no fixed asking price). |

## Fields
| Field | Type | Meaning | Origin on the page |
|---|---|---|---|
| `id` | int | krisha.kz listing id | card data-id |
| `city` | category | Astana / Almaty | search URL |
| `district` | category | city district | ad JSON address.district |
| `street, house_num` | text | address typed by the seller; used only for duplicate detection | ad JSON |
| `lat, lon` | float | map coordinates of the ad | ad JSON map |
| `price` | int | **target**: asking price, KZT | ad JSON price |
| `rooms` | int | number of rooms | ad JSON |
| `area` | float | total area, m² | ad JSON square |
| `kitchen_area` | float | kitchen area, m² | 'Площадь' line |
| `floor, total_floors` | int | floor / floors in the building | 'Этаж' line |
| `year_built` | int | year of construction | 'Год постройки' |
| `building_type` | category | monolithic / brick / panel / other / unknown | 'Тип дома' |
| `condition` | category | renovation state | 'Состояние' |
| `bathroom` | category | combined / separate / two_or_more / unknown | 'Санузел' |
| `ceiling_height` | float | ceiling height, m | 'Высота потолков' |
| `furnished` | category | fully / partly / no / unknown | 'Квартира меблирована' |
| `balcony, parking, former_dorm` | category | other attributes | parameters block |
| `has_complex` | bool | ad names a residential complex | 'Жилой комплекс' |
| `mortgaged` | bool | 'В залоге' flag | ad page |
| `seller_type` | category | owner / specialist / company | ad JSON userType |
| `n_photos` | int | number of photos | ad JSON |
| `is_urgent` | bool | 'срочно' label on the card | card class |
| `description` | text | seller's free text — kept for the Final, not a feature now | ad page |
| `photo_urls` | list | first 5 photo URLs — for the Final | ad JSON |
| `scraped_at` | datetime | collection time, UTC | scraper |
| `has_detail` | bool | True = fields from the opened ad page, False = from the search card | scraper |
| `price_m2, building_age` | derived | helper columns for EDA only; price_m2 is never a feature | notebook |

## Cleaning (rows before → after)
| Step | before | after | removed |
|---|---|---|---|
| 1 numeric fixed price | 4,822 | 4,822 | 0 |
| 2 unique listing id | 4,822 | 4,822 | 0 |
| 3 remove near-duplicate reposts | 4,822 | 4,817 | 5 |
| 4 plausible ranges | 4,817 | 4,815 | 2 |
| 5 kitchen/ceiling set to NaN (7/282 values) | 4,815 | 4,815 | 0 |

Missing numeric values are imputed inside the modelling pipeline (training fold only); missing categories get an explicit `unknown` level.

## Fixed split
80/20, `random_state=42`; test ids in `data/clean/split_ids.csv`. The same split is reused at the Endterm and the Final.
