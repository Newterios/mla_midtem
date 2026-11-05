# Apartment asking-price prediction (Astana, Almaty) — Endterm, SE-2424

Team: Rakhat Balgabekov, Aitbek Kaiyrbekuly. Direction 1 (Apartments for sale), regression, target = asking price (KZT).
Data: 4,815 krisha.kz ads collected with our own scraper on 4 and 8 October 2026. Fixed test split (963 ads, seed 42) saved at the Midterm and reused unchanged.

## Files
| Path | What |
|---|---|
| `Endterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb` | **the Endterm notebook**, executed top to bottom: change log since the Midterm, ensembles (forest, boosting, XGBoost, LightGBM) with randomised search, K-Means / DBSCAN / PCA, PyTorch network, one comparison on the saved split, error analysis, conclusions, plan for the Final, contributions and AI tools |
| `slides/Endterm_SE-2424_slides.pptx`, `.pdf` | the Endterm presentation (10 slides) |
| `endterm/results_table.csv` | every model: 5-fold CV mean ± SD and the one test evaluation |
| `endterm/search_random.csv`, `search_gradient.csv`, `search_xgboost.csv`, `search_lightgbm.csv` | all candidates of the four randomised searches |
| `endterm/segment_profiles.csv`, `error_by_segment.csv`, `nn_gradient_flow.csv`, `summary.json` | segment profiles, errors by band / city / rooms / segment, gradient flow of the network, key numbers |
| `ENDTERM_REPORT.md` | the Endterm tables in one file |
| `figures/e01…e11_*.png` | every Endterm figure (the Midterm figures are `01…08`) |
| `models/endterm_best_model.joblib` | the CV-selected model, fitted on all training rows |
| `data/clean/krisha_apartments_clean.csv`, `data/clean/split_ids.csv` | the cleaned data and the saved test split (ids) |
| `data/raw/` | the raw scrape: listings and ad pages as JSON lines, the collection log, 32 sample pages |
| `data_collection/scrape_krisha.py`, `scrub_personal_data.py` | the scraper (Midterm) and the contact-data check of the free text (Endterm fix) |
| `Midterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb`, `DATA_CARD.md`, `slides/Midterm_*` | the Midterm submission, unchanged except the data-card note and the masked handles |

## Result in one line
The CV-selected model is tuned histogram gradient boosting on log(price): test MAE 6.29 M KZT (bootstrap 5.56–7.14), MdAPE 9.13 % (8.31–9.88), R² 0.855 on the 963 saved test ads. The Midterm notebook reported MAE 8.65 M and MdAPE 11.3 % for its linear model on the same ads; re-run in Section 2.3 with the Endterm feature list (no `has_detail`, 10 wrong coordinates set to missing) the same model gives 8.66 M and 11.27 %, which is the Midterm reference used in the Endterm tables.

## How to run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python data_collection/scrub_personal_data.py      # idempotent contact-data check
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 Endterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb
```
or open the notebook in Jupyter and choose Restart and run all. It reads `data/clean/krisha_apartments_clean.csv` and `data/clean/split_ids.csv` and runs in about 8 minutes on a laptop (no GPU; the PyTorch network trains on the CPU in seconds). The Midterm notebook still runs from the raw files in `data/raw/`.
