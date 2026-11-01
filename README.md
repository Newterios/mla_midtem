# Apartment asking-price prediction (Astana, Almaty) — Endterm, SE-2424

Team: Rakhat Balgabekov, Aitbek Kaiyrbekuly. Direction 1 (Apartments for sale), regression, target = asking price (KZT).
Data: 4,815 krisha.kz ads collected with our own scraper on 4 and 8 October 2026. Fixed test split (963 ads, seed 42) saved at the Midterm and reused unchanged.

## Endterm files
| Path | What |
|---|---|
| `Endterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb` | **the Endterm notebook**, executed top to bottom: Midterm fixes, ensembles (forest, boosting, XGBoost, LightGBM) + randomised search, K-Means / DBSCAN / PCA, PyTorch network, one comparison on the saved split, error analysis, conclusions, plan for the Final, contributions |
| `notebooks_src/endterm.py` | the notebook's source in percent format; `tools/build_notebook.py` turns it into the `.ipynb` |
| `endterm/results_table.csv` | every model: 5-fold CV mean ± SD and the one test evaluation |
| `endterm/search_random.csv`, `search_gradient.csv`, `search_xgboost.csv`, `search_lightgbm.csv` | all candidates of the four randomised searches (forest, histogram boosting, XGBoost, LightGBM) |
| `endterm/segment_profiles.csv`, `endterm/error_by_segment.csv`, `endterm/summary.json` | segment profiles, errors by band / city / rooms / segment, key numbers |
| `figures/e01…e11_*.png` | every Endterm figure (the Midterm figures are `01…08`) |
| `models/endterm_best_model.joblib` | the CV-selected model, fitted on all training rows |
| `slides/Endterm_SE-2424_slides.pptx` | the Endterm presentation (built by `tools/build_slides.py` from the figures and tables) |
| `ENDTERM_REPORT.md` | the Endterm tables in one file (built by `tools/build_report.py`) |
| `DEFENSE_QA.md` | questions we expect at the defense, with our answers |
| `data_collection/scrub_personal_data.py` | the contact-data check of the free text (Midterm fix) |
| `Midterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb`, `DATA_CARD.md`, `data/`, `data_collection/scrape_krisha.py`, `figures/0*.png`, `slides/Midterm_*` | the Midterm submission, retained with the data-card note, masked handles and corrected final-review statement |

## Result in one line
The CV-selected model is tuned histogram gradient boosting on log(price). See `endterm/summary.json` and Section 6 of the notebook for the test MAE, MdAPE and the bootstrap interval; the Midterm notebook reported MAE 8.65 M KZT and MdAPE 11.3 % for its linear model on the same 963 test ads; re-run in Section 2.3 with the Endterm feature list (no `has_detail`, 10 wrong coordinates set to missing) the same model gives 8.66 M and 11.27 %, which is the Midterm reference used in the Endterm tables.

## How to reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python data_collection/scrub_personal_data.py            # idempotent contact-data check
.venv/bin/python tools/build_notebook.py notebooks_src/endterm.py Endterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 Endterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb
.venv/bin/python tools/fill_text.py          # interpretation texts from tools/texts_endterm.md
.venv/bin/python tools/build_slides.py
.venv/bin/python tools/build_report.py        # ENDTERM_REPORT.md from the endterm/ tables
```
The notebook runs from `data/clean/krisha_apartments_clean.csv` and `data/clean/split_ids.csv` in about 8 minutes on a laptop (no GPU; the PyTorch network trains on the CPU in seconds). The Midterm notebook still runs from the raw files in `data/raw/`.
