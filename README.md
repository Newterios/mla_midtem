# Apartment asking-price prediction (Astana, Almaty) — Final, SE-2424

Team: Rakhat Balgabekov, Aitbek Kaiyrbekuly. Direction 1 (Apartments for sale), regression, target = asking price (KZT).
Data: 4,815 krisha.kz ads collected with our own scraper on 4 and 8 October 2026. Fixed test split (963 ads, seed 42) saved at the Midterm and reused unchanged.

## Final (Week 10)

### Result in one line
Final model = a small multimodal network (location, raw ad fields, photo and text features/embeddings) 50/50 log-blended with the Endterm HGB, chosen by CV on duplicate-aware folds (6.100 ± 0.388 vs base 6.723 M KZT; rule: below base − 0.354 = 6.370 M): on the 963 saved test ads, scored once, MAE **5.80 M KZT** (bootstrap 5.09–6.67; Endterm 6.29, paired difference −0.80 to −0.16 M), MdAPE **7.82 %** (Endterm 9.13 %), R² 0.853. No single block passes the rule step by step; the model passes as a whole.

### Files
| Path | What |
|---|---|
| `Final_SE-2424_Balgabekov_Kaiyrbekuly.ipynb` | **the Final notebook**, executed top to bottom; reads cached `outputs/`, retrains nothing heavy |
| `FINAL_REPORT.md` | the Final tables in one file (`scripts/build_final_report.py`) |
| `slides/Final_SE-2424_slides.pptx` | the Final presentation, 15 slides (`scripts/build_final_slides.py`) |
| `CONTRACT.md` | the shared code contract of the tracks: data, folds, rule, API of `krisha_ml` |
| `krisha_ml/` | shared package: `core` (Endterm reproduction, CV harness), `dedup` + `cv_schemes` (leakage), `geo`, `photos`, `text`, `trust`, `fusion` (integration) |
| `scripts/run_final.py` | integration: compare / select / ablate / decide / fit / test (once) / nov / figures / export |
| `scripts/leakage_audit.py`, `run_geo.py`, `photo_0*.py`, `run_text.py`, `run_trust.py` | the five tracks |
| `outputs/<track>/` | every CV table, figure and decision; `outputs/final/` = model choice, test, November hold-out |
| `models/final_model.joblib`, `final_model_meta.json` | the final model fitted on the 3,852 training ads + its metadata |
| `app/gradio_app.py`, `app/predict.py` | the demo app (UI) and its inference API |
| `data/clean/krisha_nov_holdout.csv`, `data_collection/scrape_krisha_nov.py` | 1,507 November ads (listing cards only) and their scraper |
| `tests/` | pytest suite (`python -m pytest -q`) |

### How to run
```bash
PY=~/Downloads/mla/mla_midtem/.venv/bin/python           # or a fresh venv with requirements.txt
OMP_NUM_THREADS=3 $PY scripts/run_final.py compare decide fit nov figures --jobs 3   # CV + choice (~5 min)
OMP_NUM_THREADS=3 $PY scripts/run_final.py test figures export                      # the ONE test evaluation (refuses to rerun)
$PY scripts/build_final_notebook.py && $PY -m jupyter nbconvert --to notebook --execute --inplace Final_SE-2424_Balgabekov_Kaiyrbekuly.ipynb
$PY scripts/build_final_report.py && $PY scripts/build_final_slides.py
$PY -m pytest -q
```

### App
```bash
$PY app/gradio_app.py            # http://127.0.0.1:7860 ; --no-photos for a faster start
```
For a typed-in listing the app builds the final model's inputs from the form (OSM geo from the coordinates, text flags and the USER-base embedding from the description, CLIP embedding of up to 3 photos; raw ad fields and photo-track table features stay missing and the app says so) and shows the estimate with the Endterm base next to it, a conformal 80/90 % interval and verdict, SHAP reasons, 5 comparable training ads, red flags and the photo condition with a mismatch check. Example: `outputs/final/app_example.png` (`scripts/make_app_example.py`).

## Endterm

### Files
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

### Result in one line
The CV-selected model is tuned histogram gradient boosting on log(price): test MAE 6.29 M KZT (bootstrap 5.56–7.14), MdAPE 9.13 % (8.31–9.88), R² 0.855 on the 963 saved test ads. The Midterm notebook reported MAE 8.65 M and MdAPE 11.3 % for its linear model on the same ads; re-run in Section 2.3 with the Endterm feature list (no `has_detail`, 10 wrong coordinates set to missing) the same model gives 8.66 M and 11.27 %, which is the Midterm reference used in the Endterm tables.

### How to run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python data_collection/scrub_personal_data.py      # idempotent contact-data check
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 Endterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb
```
or open the notebook in Jupyter and choose Restart and run all. It reads `data/clean/krisha_apartments_clean.csv` and `data/clean/split_ids.csv` and runs in about 8 minutes on a laptop (no GPU; the PyTorch network trains on the CPU in seconds). The Midterm notebook still runs from the raw files in `data/raw/`.
