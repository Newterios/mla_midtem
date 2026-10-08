# Apartment sale price prediction (Astana, Almaty) — Midterm, SE-2424

Team: Rakhat Balgabekov, Aitbek Kaiyrbekuly. Direction 1 (Apartments for sale), regression, target = asking price (KZT).

## Files
| Path | What |
|---|---|
| `Midterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb` | the notebook: data card, cleaning, EDA, models, results, error analysis, conclusions, plan, contributions |
| `data_collection/scrape_krisha.py` | our scraper (stage 1 search pages, stage 2 ad pages) |
| `data/raw/listings_raw.jsonl`, `data/raw/details_raw.jsonl` | raw snapshot, untouched field values, `scraped_at` timestamps |
| `data/raw/collection_log.json`, `data/raw/stage*.log` | when and how the data were collected (search pages and first half of ad pages on 4 Oct 2026, second half on 8 Oct 2026) |
| `data/raw/html_sample/` | a few raw HTML pages as evidence of the source |
| `data/clean/krisha_apartments_clean.csv` | cleaned dataset |
| `data/clean/split_ids.csv` | ids of the fixed test split (seed 42) — reused at the Endterm and the Final |
| `figures/` | every plot of the notebook + `results_table.csv` |
| `models/best_model.joblib` | the fitted best model (pipeline + linear model on log price), saved by the notebook |
| `slides/Midterm_SE-2424_slides.pptx` (+ `.pdf`) | the presentation |
| `DATA_CARD.md` | the data card as a separate file |

## How to reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python data_collection/scrape_krisha.py --stage listings   # search pages (~5 min)
.venv/bin/python data_collection/scrape_krisha.py --stage details    # ad pages (~1 h, polite delays)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Midterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb
```
The notebook runs top to bottom from the raw files in `data/raw/`.
