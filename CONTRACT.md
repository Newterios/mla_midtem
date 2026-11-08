# krisha_ml contract (Final, shared foundation)

`krisha_ml/` reproduces the **submitted Endterm** exactly and gives every Final track one CV harness, so all tracks report numbers on the same rows, folds and model.

Python: `~/Downloads/mla/mla_midtem/.venv/bin/python`, run from the repo root (`pytest.ini` puts the root on `sys.path`; scripts in `scripts/` add it themselves).

## 1. Base numbers (verified 2026-11-08, `scripts/verify_endterm.py`)

| | value |
|---|---|
| Base CV MAE (5 folds, 3,852 training ads) | **6.616 ± 0.354 M KZT** (mean ± population SD, as in the Endterm) |
| Folds MAE, M KZT | 6.277, 6.860, 7.142, 6.200, 6.601 |
| Base CV MdAPE / R² | 8.93 % / 0.857 |
| **1-SD rule threshold** | a variant is kept only if `base.mean - variant.mean > 0.3535` M, i.e. CV MAE **< 6.262 M** |
| Test sanity (submitted model, one-time check) | MAE 6.2909 M, RMSE 13.90 M, MdAPE 9.129 %, R² 0.8549, ±10 % 53.7 %, ±20 % 84.3 % |

The CV and test numbers match `endterm/search_gradient.csv` and `endterm/summary.json` to 4 decimals (differences < 1e-6). Saved in `outputs/core/`: `base_cv.json`, `base_cv_folds.csv`, `base_oof.csv` (id, fold, price, base OOF prediction, log residual for every training ad), `endterm_repro_test.json`.

## 2. Rules for every track

- Work on **training rows only**: `load_train()`. Never call `load_train_test()` (integration stage only). Do not evaluate on the 963 test ads.
- Report `CVResult.row(base)` (mean ± SD, folds, delta, `folds_better`, `passes_1sd_rule`). A block is **kept only if `passes_1sd_rule` is True**. Report honestly when it is not.
- Anything **fitted on data** (TF-IDF, PCA/SVD of embeddings, KMeans, target or neighbour encodings, any model of the price) must be fitted inside the fold: use `feature_fn` or a residual `corrector`, never a table computed on all ads. Static per-ad values (parsed raw fields, counts, raw embedding columns) may go in `extra`.
- Seed 42 everywhere. Outputs to `outputs/<track>/`, caches to `cache/`, logs to `logs/`, photos to `data/photos/` (all gitignored; photos are never committed).
- CPU: HGB already uses all cores. With 5 tracks in parallel use `n_jobs<=2` in the harness, or set `OMP_NUM_THREADS=2`.

## 3. API (`from krisha_ml import core, raw`)

### Data
```python
core.load_clean(path=CLEAN_CSV) -> pd.DataFrame        # 4,815 ads, 10 out-of-city coordinates set to NaN
core.fix_coordinates(df) -> (df_copy, fixed_mask)      # the Endterm coordinate step (CITY_BOX)
core.encode_features(df) -> pd.DataFrame               # the 24 FEATURES, Endterm encoding
core.load_test_ids() -> set[int]                       # 963 ids from data/clean/split_ids.csv
core.load_train() -> Dataset                           # 3,852 training ads, Endterm row order
core.load_train_test() -> (Dataset, Dataset)           # integration stage only
# Dataset(df: all clean columns incl. description/photo_urls, X: 24 features, y: price KZT, ids: int64)
#         all share row order and a RangeIndex
core.cv_folds(n_rows=3852) -> list[(train_idx, val_idx)]   # KFold(5, shuffle=True, random_state=42), positional
```
Constants: `SEED=42`, `NUM` (11), `CAT` (10), `BOOL` (3), `FEATURES = NUM + CAT + BOOL` (24, no `has_detail`, no `price_m2`), `HGB_PARAMS`, `CITY_BOX`, `ENDTERM_REFERENCE`, `ROOT`, `CACHE_DIR`, `OUTPUT_DIR`.

### Model and metrics
```python
core.make_preprocessor(extra_num=(), extra_cat=()) -> ColumnTransformer
#   numeric+bool (+extra_num): median imputation with missing indicators; CAT (+extra_cat): one-hot, min_frequency=10
core.make_base_model(extra_num=(), extra_cat=(), **hgb_overrides) -> TransformedTargetRegressor
#   HGB(min_samples_leaf=5, max_leaf_nodes=7, max_iter=600, max_depth=None, learning_rate=0.12,
#       l2_regularization=0.0, random_state=42, early_stopping=False) on log(price); predicts KZT
core.metrics(y_true, y_pred) -> {"mae_M", "rmse_M", "mdape", "r2", "within10", "within20"}
core.mdape(y_true, y_pred) -> float  # percent
```

### Joining per-ad features
```python
core.join_features(ids, extra, id_col="id", prefix=None) -> pd.DataFrame
#   ids: id array or a frame with an "id" column; extra keyed by "id" column or index.
#   One row per requested id, same order, RangeIndex; missing ads -> NaN; duplicated ids -> ValueError.
```

### CV of "base + extra features"
```python
core.cv_evaluate(name="base", extra=None, feature_fn=None, model_factory=None, train=None, n_jobs=1) -> CVResult
```
- `extra`: per-ad DataFrame keyed by id (static, unfitted features). Numeric/bool columns -> imputed numeric branch; other dtypes -> one-hot (NaN -> "unknown"). Column names must not clash with `FEATURES`.
- `feature_fn(df_tr, y_tr, df_va) -> (E_tr, E_va)`: called once per fold with the fold's clean rows (RangeIndex) and training prices; fit on `df_tr` only, return two frames with identical columns, one row per ad in the given order. Can be combined with `extra`.
- `model_factory(extra_num, extra_cat) -> estimator` with `fit(X, y_kzt)` / `predict(X) -> kzt`; default `make_base_model`. X = 24 base columns followed by the extra columns.

```python
# example: unused raw fields as a feature block
raw_tab = raw.load_raw_details()
block = raw_tab[["id", "flooring", "door", "furniture", "internet"]]
res = core.cv_evaluate("base + raw fields", extra=block, n_jobs=2)
print(res.row(core.base_cv()))

# example: fold-fitted text features
def tfidf_svd(df_tr, y_tr, df_va):
    vec = TfidfVectorizer(min_df=5).fit(df_tr["description"].map(raw.sanitize_text))
    svd = TruncatedSVD(20, random_state=42).fit(vec.transform(df_tr["description"].map(raw.sanitize_text)))
    f = lambda d: pd.DataFrame(svd.transform(vec.transform(d["description"].map(raw.sanitize_text))), columns=[f"txt{i}" for i in range(20)])
    return f(df_tr), f(df_va)
res = core.cv_evaluate("base + text SVD", feature_fn=tfidf_svd)
```

### CV of "base + residual correction" (stacking on log scale)
```python
core.cv_evaluate_residual(corrector, name="base + residual", train=None, n_jobs=1) -> CVResult
# corrector(df_tr, resid_tr, df_va) -> delta_va   (np.ndarray, log scale, one finite value per val ad)
```
For outer fold k: `resid_tr = log(y_tr) - inner_oof_k`, where `inner_oof_k` are inner 5-fold OOF base predictions on the training fold (KFold(5, shuffle, 42) over those rows), so the corrector never sees residuals of a model that was trained on the same ad. Both frames carry `base_log_pred` (inner OOF on `df_tr`; base model fitted on the whole training fold on `df_va`). Prediction = `exp(base_log_pred_va + delta_va)`. A corrector returning zeros reproduces the base exactly (tested).

```python
def text_ridge(df_tr, resid_tr, df_va):
    vec = TfidfVectorizer(preprocessor=raw.sanitize_text, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    model = Ridge(alpha=10).fit(vec.fit_transform(df_tr["description"].fillna("")), resid_tr)
    return model.predict(vec.transform(df_va["description"].fillna("")))
res = core.cv_evaluate_residual(text_ridge, "base + text residual")
```
Helpers: `core.base_residual_inputs(train=None) -> {"inner": [5 arrays], "outer": array}` (cached in `cache/base_residual_inputs_<hash>.npz`, atomic write), `core.base_cv() -> CVResult` (base, from the cache), `core.oof_log_pred(X, y, n_folds=5, seed=42)` (honest OOF log predictions for any training set: use it in the integration stage to fit a corrector on all training ads).

### CVResult
Fields `name, fold_mae, fold_mdape, fold_r2, oof_pred` (KZT, `load_train()` order); properties `mean`, `sd` (ddof=0); `delta(base)`, `passes_rule(base)`, `row(base)`.

### Raw ad pages (`krisha_ml/raw.py`)
```python
raw.load_raw_details(path=DETAILS_JSONL) -> pd.DataFrame   # 4,822 ads (13 with fetch_ok=False), ~1.3 s
raw.sanitize_text(text) -> str      # lower, no URLs/e-mails/@handles/phones/"[contact removed]",
                                    # no sentence mentioning price/money (цена, млн, тенге, ₸, $...), no digits, letters only
raw.mask_contacts(text) -> str      # phones/e-mails/@handles/URLs -> "[contact removed]"
raw.multi_hot(series, prefix, sep=",", min_count=1) -> pd.DataFrame   # e.g. security lists -> 0/1 columns
```
Columns: `id, fetch_ok, description` (contacts masked), `description_sanitized, photo_urls` (list, ≤ 5), `n_photo_urls, n_photos_listed`, and raw fields (info/param blocks coalesced, None when empty): `flooring` (57 % filled), `door` (62 %), `security` (61 %, comma list), `furniture` (64 %), `internet` (44 %), `phone` (28 %), `balcony_glazed` (17 %), `exchange`, `renovation_raw`, `mortgaged_text` (10 %; the same information as the base feature `mortgaged`), `complex_name, complex_id` (75 %), `estate_verified, title, address_title, scraped_at`. Every clean ad id is present.

Other reusable inputs: `final/photo_embeddings.npz` (MobileNetV3-small, keys `ids` (4,815), `features` (4,815 × 1152), `valid_photos`).

Smoke run of the three examples above, untuned, only to show the harness works (2026-11-08): raw fields 6.705 ± 0.294 (+0.09, fails); text SVD-20 as features 6.864 ± 0.341 (+0.25, fails); text Ridge residual 6.431 ± 0.379 (−0.19, better on 5/5 folds, still fails the 0.354 rule). Tracks own the tuned versions.

## 4. Tests
`python -m pytest -q` (9 tests, ~5 s): split/feature list, coordinate fix (10 ads), folds, base CV = Endterm, a noise column fails the rule, zero corrector = base, join order/missing/duplicates, metrics, sanitizer, raw table.

## 5. Track geo (`krisha_ml/geo.py`, `scripts/run_geo.py`, 2026-11-08)
```python
from krisha_ml import geo
geo.fetch_all()                       # Overpass, once; raw JSON in cache/osm/<city>_<layer>.json (19 files, 5.8 MB)
geo.geo_features(df) -> DataFrame     # id + 18 geo_* columns (km distances, 1 km counts, bus stops in 500 m); NaN without coordinates
geo.rawfield_features() -> DataFrame  # id + 16 raw_* columns (flooring, door, furniture, internet, phone, balcony_glazed: str; exchange 0/1; security multi-hot + count)
```
Static per-ad tables (no prices used), so they go into `cv_evaluate(extra=...)`. Saved: `outputs/geo/features_geo.csv`, `features_rawfields.csv` (all 4,815 clean ads, id-keyed).
CV (rule threshold 6.262 M): base+geo 6.531 ± 0.251 (−0.085), base+raw 6.766 ± 0.291 (+0.150), base+geo+raw 6.481 ± 0.280 (−0.135, 4/5 folds better), centre distance only 6.543 ± 0.243 (−0.073). **None passes the 1-SD rule; nothing from this track is kept.** Details: `outputs/geo/cv.csv`, `permutation_importance.csv`, `residual_map.png`.

## 5. Leakage audit and the Final folds (`krisha_ml/dedup.py`, `krisha_ml/cv_schemes.py`, 2026-11-08)

Run: `python scripts/leakage_audit.py` (resumable, every CV run cached in `cache/leakage_cv/`; ~10 min on an idle machine). Per-image MobileNetV3-small embeddings: `python scripts/embed_photos.py` -> `cache/mnv3_image_emb.npz` (ad_id, k, emb 576-d, L2-normalised, static).

**Duplicates** (all 4,815 clean ads, descriptive count only; the folds come from a second search over the 3,852 training ads alone, which gives the identical groups; signals: pHash<=8 & dHash<=10, embedding cosine>=0.97, card match, description char-TF-IDF>=0.90; stock images in >5 ads ignored; photo links need plausible cards): 48 pairs, 43 groups, 90 ads (39 pairs + 4 triples). Tier `same_flat` (card agrees): 18 pairs; tier `near` (mostly different flats of one new building sharing developer renders/plans): 30 pairs. 13 groups cross the train/test split -> **13 test ads (1.35 %) have a twin in train** (7 same-flat), ids in `outputs/leakage/test_ads_with_train_twin.csv`.

**CV schemes, base model, training ads** (`cv_schemes_summary.csv`, mean over fold seeds):

| scheme | CV MAE, M | seed SD | seeds |
|---|---|---|---|
| random KFold (Endterm scheme) | 6.720 | 0.058 | 20 |
| same KFold, duplicate groups kept together | 6.734 | 0.064 | 20 |
| placebo: same KFold, 26 random ads moved | 6.717 | 0.069 | 20 |
| GroupKFold by duplicate group | 6.724 | 0.081 | 10 |
| GroupKFold by building (+dups) | 7.490 | 0.114 | 10 |
| spatial 1 km blocks (+dups) | 7.692 | 0.093 | 10 |
| spatial 3 km blocks (+dups) | 8.395 | 0.187 | 10 |

Paired by seed: duplicates-together minus random = +0.015 ± 0.058 M (20 seeds); placebo minus random = -0.003 ± 0.051. **Duplicate leakage inflates the base CV by only ~0.015 M.** The Endterm seed-42 folds (6.616) are the best of 20 fold seeds: the hyper-parameters were tuned on them, so 6.616 is ~0.10 M optimistic. Building/spatial CV is 0.8-1.7 M worse: that is extrapolation to unseen buildings/areas, not leakage w.r.t. the random test split.

**Recommended Final folds:** `cv_schemes.load_group_folds()` (`outputs/leakage/folds_group.csv`: id, fold, dup_group, dup_group_size) = the Endterm folds with each duplicate group moved whole into one fold (26 ads moved). Base on them: 6.723 ± 0.264 M (folds 6.371 / 6.901 / 7.109 / 6.513 / 6.721). Evaluate a variant on these folds with `cv_schemes.cv_with_folds(name, folds, extra=..., feature_fn=...)` and compare to the base **on the same folds**; residual correctors: `cv_schemes.cv_residual_with_folds(corrector, folds, name)` (same corrector contract as `core.cv_evaluate_residual`; inner residual folds are duplicate-aware too, cached in `cache/`); keep the rule margin fixed at the Endterm 0.3535 M, i.e. keep only if CV MAE < 6.370 M on these folds (fold SD itself moves 0.26-0.50 with the fold seed). Photo/text tracks: report both fold sets; a gain that is clearly larger on the Endterm folds than on the repaired folds means the block memorises duplicate photos/text.
