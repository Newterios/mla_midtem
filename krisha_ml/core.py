"""Reproduction of the submitted Endterm model and the shared CV harness for the Final.

Everything here mirrors the Endterm notebook (Endterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb,
sections 2.2, 2.2-bis and 3.3) line for line: same cleaning step, same 24 features, same
encoding, same fixed test split, same 5 folds, same tuned HistGradientBoosting on log(price).

The team rule for the Final: a new feature block or correction is kept only if it lowers the
5-fold CV MAE (train rows only, the folds of ``cv_folds()``) by more than one fold SD of the
base model.  Use ``cv_evaluate`` / ``cv_evaluate_residual`` and ``CVResult.passes_rule``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------- constants
ROOT = Path(os.environ.get("KRISHA_ROOT", Path(__file__).resolve().parent.parent))
CLEAN_CSV = ROOT / "data" / "clean" / "krisha_apartments_clean.csv"
SPLIT_CSV = ROOT / "data" / "clean" / "split_ids.csv"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42
N_FOLDS = 5
TARGET = "price"
ID = "id"

NUM = ["area", "log_area", "rooms", "kitchen_area", "floor", "total_floors", "year_built",
       "ceiling_height", "n_photos", "lat", "lon"]
CAT = ["city", "district", "building_type", "condition", "bathroom", "furnished", "balcony",
       "parking", "former_dorm", "seller_type"]
BOOL = ["has_complex", "mortgaged", "is_urgent"]
FEATURES = NUM + CAT + BOOL  # 24 columns, order matters for exact reproduction

# Generous lat/lon box per city; coordinates outside it are set to NaN (Endterm section 2.2).
CITY_BOX = {"Astana": (50.8, 51.4, 71.0, 71.9), "Almaty": (43.0, 43.5, 76.6, 77.3)}

# Best setting of the Endterm RandomizedSearchCV (40 candidates, CV MAE 6.62 +- 0.35 M KZT).
HGB_PARAMS = {"min_samples_leaf": 5, "max_leaf_nodes": 7, "max_iter": 600, "max_depth": None,
              "learning_rate": 0.12, "l2_regularization": 0.0}

# Reference numbers of the submitted Endterm (endterm/summary.json, endterm/search_gradient.csv).
ENDTERM_REFERENCE = {"cv_mae_M": 6.615971, "cv_mae_sd_M": 0.353523, "cv_mdape": 8.933971, "cv_r2": 0.856864,
                     "test_mae_M": 6.290902, "test_mdape": 9.129338, "test_r2": 0.854899}


# --------------------------------------------------------------------------- data
def fix_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Set lat/lon to NaN where they fall outside the city's box. Returns (copy, mask of fixed rows)."""
    df = df.copy()
    outside = pd.Series(False, index=df.index)
    for city, (la0, la1, lo0, lo1) in CITY_BOX.items():
        in_city = df["city"] == city
        inside_box = df["lat"].between(la0, la1) & df["lon"].between(lo0, lo1)
        outside |= in_city & df["lat"].notna() & ~inside_box
    df.loc[outside, ["lat", "lon"]] = np.nan
    return df, outside


@lru_cache(maxsize=1)
def _load_clean_cached(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df, _ = fix_coordinates(df)
    return df


def load_clean(path: Path | str = CLEAN_CSV) -> pd.DataFrame:
    """All 4,815 clean ads with the Endterm coordinate fix applied (a fresh copy each call)."""
    return _load_clean_cached(str(path)).copy()


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """The 24 Endterm features with the Endterm encoding (bool -> 0/1, cat -> str with 'unknown')."""
    X = df[FEATURES].copy()
    for c in BOOL:
        X[c] = X[c].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    for c in CAT:
        X[c] = X[c].astype(object).where(X[c].notna(), "unknown").astype(str)
    for c in NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def load_test_ids(path: Path | str = SPLIT_CSV) -> set[int]:
    split = pd.read_csv(path)
    return set(split.loc[split["split"] == "test", ID].astype("int64"))


@dataclass(frozen=True)
class Dataset:
    """Rows of one side of the split. ``df`` keeps every clean column (description, photo_urls, ...),
    ``X`` the 24 encoded features, ``y`` the asking price in KZT, ``ids`` the krisha ad id.
    All four share the same row order and a RangeIndex."""
    df: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    ids: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def _make_dataset(df: pd.DataFrame) -> Dataset:
    df = df.reset_index(drop=True)
    return Dataset(df=df, X=encode_features(df), y=df[TARGET].to_numpy(dtype="float64"),
                   ids=df[ID].to_numpy(dtype="int64"))


def load_train() -> Dataset:
    """The 3,852 training ads, in the Endterm row order (the folds of ``cv_folds`` index this)."""
    df = load_clean()
    return _make_dataset(df[~df[ID].isin(load_test_ids())])


def load_train_test() -> tuple[Dataset, Dataset]:
    """Train and the 963 test ads. INTEGRATION STAGE ONLY: never use the test side for a choice."""
    df = load_clean()
    mask = df[ID].isin(load_test_ids())
    train, test = _make_dataset(df[~mask]), _make_dataset(df[mask])
    assert len(test) == 963 and len(train) == 3852, (len(train), len(test))
    return train, test


def cv_splitter() -> KFold:
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def cv_folds(n_rows: int = 3852) -> list[tuple[np.ndarray, np.ndarray]]:
    """The Endterm folds as positional (train_idx, val_idx) pairs over ``load_train()`` rows."""
    return list(cv_splitter().split(np.arange(n_rows)))


# --------------------------------------------------------------------------- model
def make_preprocessor(extra_num: Sequence[str] = (), extra_cat: Sequence[str] = ()) -> ColumnTransformer:
    """Endterm tree preprocessing: median imputation + missing indicators for numeric/bool columns,
    one-hot (min_frequency=10) for categoricals. Extra columns are appended after the base ones."""
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))])
    return ColumnTransformer([
        ("num", num, NUM + BOOL + list(extra_num)),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), CAT + list(extra_cat)),
    ])


def make_base_model(extra_num: Sequence[str] = (), extra_cat: Sequence[str] = (), **hgb_overrides) -> TransformedTargetRegressor:
    """The submitted Endterm model (tuned HGB on log(price)); predictions come back in KZT."""
    params = {**HGB_PARAMS, **hgb_overrides}
    model = HistGradientBoostingRegressor(**params, random_state=SEED, early_stopping=False)
    return TransformedTargetRegressor(regressor=Pipeline([("pre", make_preprocessor(extra_num, extra_cat)), ("model", model)]),
                                      func=np.log, inverse_func=np.exp, check_inverse=False)


# --------------------------------------------------------------------------- metrics
def mdape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.median(np.abs(y_pred - y_true) / y_true) * 100)


def metrics(y_true, y_pred) -> dict[str, float]:
    """MAE/RMSE in M KZT, MdAPE %, R2 and share of ads within +-10 % / +-20 % of the price."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    return {"mae_M": float(mean_absolute_error(y_true, y_pred) / 1e6),
            "rmse_M": float(root_mean_squared_error(y_true, y_pred) / 1e6),
            "mdape": float(np.median(ape) * 100),
            "r2": float(r2_score(y_true, y_pred)),
            "within10": float((ape <= 0.10).mean()),
            "within20": float((ape <= 0.20).mean())}


# --------------------------------------------------------------------------- extra features
def join_features(ids: Iterable[int] | pd.DataFrame, extra: pd.DataFrame, id_col: str = ID,
                  prefix: str | None = None) -> pd.DataFrame:
    """Align a per-ad feature table to ``ids`` (an id array or a frame with an ``id`` column).

    ``extra`` is keyed by ad id either through its index or an ``id`` column. The result has one row
    per requested id, in that order, with a RangeIndex; ads missing from ``extra`` get NaN.
    """
    ids = ids[id_col].to_numpy() if isinstance(ids, pd.DataFrame) else np.asarray(list(ids))
    table = extra.set_index(id_col) if id_col in extra.columns else extra.copy()
    table.index = table.index.astype("int64")
    if not table.index.is_unique:
        raise ValueError(f"extra features have {int(table.index.duplicated().sum())} duplicated ad ids")
    out = table.reindex(np.asarray(ids, dtype="int64")).reset_index(drop=True)
    if prefix:
        out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


def split_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric/bool columns -> imputed numeric branch; everything else -> one-hot branch."""
    num = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])]
    return num, [c for c in frame.columns if c not in num]


def _prepare_extra(E: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    E = E.reset_index(drop=True).copy()
    for c in num:
        E[c] = pd.to_numeric(E[c], errors="coerce").astype("float64")
    for c in cat:
        E[c] = E[c].astype(object).where(E[c].notna(), "unknown").astype(str)
    return E


# --------------------------------------------------------------------------- CV harness
@dataclass
class CVResult:
    name: str
    fold_mae: np.ndarray                  # per-fold MAE, M KZT, folds of cv_folds()
    fold_mdape: np.ndarray
    fold_r2: np.ndarray
    oof_pred: np.ndarray = field(repr=False)   # out-of-fold predictions in KZT, load_train() order

    @property
    def mean(self) -> float:
        return float(self.fold_mae.mean())

    @property
    def sd(self) -> float:
        # population SD (ddof=0), the same as sklearn's std_test_* used in the Endterm
        return float(self.fold_mae.std())

    def delta(self, base: "CVResult") -> float:
        return self.mean - base.mean

    def passes_rule(self, base: "CVResult") -> bool:
        """Team rule: kept only if CV MAE drops by more than one fold SD of the base model."""
        return (base.mean - self.mean) > base.sd

    def row(self, base: "CVResult | None" = None) -> dict:
        r = {"variant": self.name, "cv_mae_M": round(self.mean, 4), "cv_mae_sd_M": round(self.sd, 4),
             "cv_mdape": round(float(self.fold_mdape.mean()), 3), "cv_r2": round(float(self.fold_r2.mean()), 4),
             "folds_mae_M": [round(float(v), 3) for v in self.fold_mae]}
        if base is not None:
            r["delta_vs_base_M"] = round(self.delta(base), 4)
            r["folds_better"] = int((self.fold_mae < base.fold_mae).sum())
            r["passes_1sd_rule"] = self.passes_rule(base)
        return r

    def __str__(self) -> str:
        return f"{self.name}: CV MAE {self.mean:.3f} +- {self.sd:.3f} M, MdAPE {self.fold_mdape.mean():.2f} %, R2 {self.fold_r2.mean():.3f}"


FeatureFn = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]
ModelFactory = Callable[[list[str], list[str]], object]


def _fold_fit_predict(train: Dataset, tr, va, extra, feature_fn, model_factory):
    X_tr, X_va = train.X.iloc[tr].reset_index(drop=True), train.X.iloc[va].reset_index(drop=True)
    blocks_tr, blocks_va = [], []
    if extra is not None:
        blocks_tr.append(extra.iloc[tr]); blocks_va.append(extra.iloc[va])
    if feature_fn is not None:
        e_tr, e_va = feature_fn(train.df.iloc[tr].reset_index(drop=True), train.y[tr], train.df.iloc[va].reset_index(drop=True))
        if list(e_tr.columns) != list(e_va.columns) or len(e_tr) != len(tr) or len(e_va) != len(va):
            raise ValueError("feature_fn must return (train_frame, val_frame) with equal columns and one row per ad")
        blocks_tr.append(e_tr); blocks_va.append(e_va)
    num, cat = [], []
    if blocks_tr:
        E_tr = pd.concat([b.reset_index(drop=True) for b in blocks_tr], axis=1)
        E_va = pd.concat([b.reset_index(drop=True) for b in blocks_va], axis=1)
        clash = set(E_tr.columns) & set(FEATURES)
        if clash or not E_tr.columns.is_unique:
            raise ValueError(f"extra feature names must be unique and differ from the base features: {sorted(clash)}")
        num, cat = split_columns(E_tr)
        X_tr = pd.concat([X_tr, _prepare_extra(E_tr, num, cat)], axis=1)
        X_va = pd.concat([X_va, _prepare_extra(E_va, num, cat)], axis=1)
    model = (model_factory or make_base_model)(num, cat)
    model.fit(X_tr, train.y[tr])
    return model.predict(X_va)


def _result_from_oof(name: str, y: np.ndarray, oof: np.ndarray, folds) -> CVResult:
    m = [metrics(y[va], oof[va]) for _, va in folds]
    return CVResult(name, np.array([d["mae_M"] for d in m]), np.array([d["mdape"] for d in m]),
                    np.array([d["r2"] for d in m]), oof)


def cv_evaluate(name: str = "base", extra: pd.DataFrame | None = None, feature_fn: FeatureFn | None = None,
                model_factory: ModelFactory | None = None, train: Dataset | None = None, n_jobs: int = 1) -> CVResult:
    """5-fold CV of "base + extra features" on the training ads, Endterm folds.

    extra         per-ad features that need no fitting (keyed by ad id: index or ``id`` column); they are
                  joined with ``join_features``. Anything fitted on data (TF-IDF, PCA, target encodings)
                  must go through ``feature_fn`` instead, so it is refitted inside every fold.
    feature_fn    ``fn(df_tr, y_tr, df_va) -> (E_tr, E_va)``: fit on the training fold only and return
                  feature frames for both sides (same columns, one row per ad, in the given order).
    model_factory ``fn(extra_num, extra_cat) -> estimator`` (default ``make_base_model``).
    Numeric/bool extra columns go to the imputed numeric branch, other dtypes are one-hot encoded.
    """
    train = train or load_train()
    folds = cv_folds(len(train))
    E = join_features(train.ids, extra) if extra is not None else None
    preds = Parallel(n_jobs=n_jobs)(delayed(_fold_fit_predict)(train, tr, va, E, feature_fn, model_factory) for tr, va in folds)
    oof = np.empty(len(train))
    for (_, va), p in zip(folds, preds):
        oof[va] = p
    return _result_from_oof(name, train.y, oof, folds)


# --------------------------------------------------------------------------- residual correction
def _cache_key(tag: str, train: Dataset) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"tag": tag, "params": HGB_PARAMS, "seed": SEED, "sklearn": sklearn.__version__}, sort_keys=True, default=str).encode())
    h.update(train.ids.tobytes()); h.update(train.y.tobytes())
    return h.hexdigest()[:16]


def oof_log_pred(X: pd.DataFrame, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED, model_factory=None) -> np.ndarray:
    """Honest out-of-fold log-price predictions of the base model on (X, y): every row is predicted by
    a model that did not see it. Use these to build residuals ``log(y) - oof`` for a corrector."""
    oof = np.empty(len(y))
    for tr, va in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        m = (model_factory or make_base_model)([], []).fit(X.iloc[tr], y[tr])
        oof[va] = np.log(m.predict(X.iloc[va]))
    return oof


def base_residual_inputs(train: Dataset | None = None, use_cache: bool = True) -> dict[str, list[np.ndarray] | np.ndarray]:
    """Cached base predictions for residual correction, per outer fold k of ``cv_folds()``:

    inner[k]  inner 5-fold OOF log prediction for the training rows ``folds[k][0]`` (seed 42 KFold
              over those rows) -> honest training residuals ``log(y_tr) - inner[k]``
    outer     outer OOF log prediction for every training ad (= base CV predictions, log scale)
    """
    train = train or load_train()
    folds = cv_folds(len(train))
    path = CACHE_DIR / f"base_residual_inputs_{_cache_key('resid', train)}.npz"
    if use_cache and path.exists():
        z = np.load(path)
        return {"inner": [z[f"inner{k}"] for k in range(len(folds))], "outer": z["outer"]}
    inner = Parallel(n_jobs=len(folds))(delayed(oof_log_pred)(train.X.iloc[tr].reset_index(drop=True), train.y[tr]) for tr, _ in folds)
    outer = np.log(cv_evaluate("base", train=train, n_jobs=len(folds)).oof_pred)
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")   # atomic: parallel tracks share this cache
    np.savez(tmp, outer=outer, **{f"inner{k}": a for k, a in enumerate(inner)})
    os.replace(tmp, path)
    return {"inner": inner, "outer": outer}


Corrector = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


def cv_evaluate_residual(corrector: Corrector, name: str = "base + residual", train: Dataset | None = None,
                         n_jobs: int = 1) -> CVResult:
    """5-fold CV of "base + residual correction" (stacking on log scale, no leakage).

    For each outer fold k: the corrector is called as ``corrector(df_tr, resid_tr, df_va)`` where
    ``resid_tr = log(y_tr) - inner OOF base log prediction`` on the training fold (cross-fitted, so the
    residuals are honest), and both frames carry a column ``base_log_pred`` (inner OOF for df_tr, the
    base model fitted on the whole training fold for df_va). It must return ``delta_va`` (log-scale,
    one value per validation ad); the corrected prediction is ``exp(base_log_pred_va + delta_va)``.
    """
    train = train or load_train()
    folds = cv_folds(len(train))
    base = base_residual_inputs(train)
    log_y = np.log(train.y)

    def one(k, tr, va):
        df_tr = train.df.iloc[tr].reset_index(drop=True).assign(base_log_pred=base["inner"][k])
        df_va = train.df.iloc[va].reset_index(drop=True).assign(base_log_pred=base["outer"][va])
        delta = np.asarray(corrector(df_tr, log_y[tr] - base["inner"][k], df_va), dtype=float).ravel()
        if delta.shape != (len(va),) or not np.isfinite(delta).all():
            raise ValueError("corrector must return one finite log-scale delta per validation ad")
        return np.exp(base["outer"][va] + delta)

    preds = Parallel(n_jobs=n_jobs)(delayed(one)(k, tr, va) for k, (tr, va) in enumerate(folds))
    oof = np.empty(len(train))
    for (_, va), p in zip(folds, preds):
        oof[va] = p
    return _result_from_oof(name, train.y, oof, folds)


@lru_cache(maxsize=1)
def base_cv() -> CVResult:
    """The base Endterm model's CV result (computed once per process; reuses the cached OOF)."""
    train = load_train()
    outer = base_residual_inputs(train)["outer"]
    return _result_from_oof("Endterm base (tuned HGB, 24 features)", train.y, np.exp(outer), cv_folds(len(train)))
