"""Alternative CV schemes for the leakage audit and the group-aware folds of the Final.

All schemes split the 3,852 training ads of ``core.load_train()`` (positional indices, same row order)
and run the unchanged Endterm base model, so their CV MAE differs only through the folds.

    random    KFold(5, shuffle, seed)                              - the Endterm scheme
    random+dupfix  the same KFold, then every duplicate group moved whole into one fold
              (``repair_folds``): a paired comparison that isolates the duplicate effect
    dup       GroupKFold(5, shuffle, seed), groups = duplicate groups (krisha_ml.dedup)
    building  GroupKFold by building: same residential complex id, or the same coordinates
              (~10 m), or the same street + house number; merged with the duplicate groups
    spatial   GroupKFold by ~1 km grid cell per city (ads without coordinates: city + district);
              merged with the duplicate groups so one flat never spans two blocks

Use ``load_group_folds()`` to get the recommended Final folds (outputs/leakage/folds_group.csv):
the Endterm folds (KFold 5, shuffle, 42) repaired so that no duplicate group spans two folds.
"""
from __future__ import annotations

from pathlib import Path

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold, KFold

from . import core
from .core import ID, N_FOLDS, OUTPUT_DIR, SEED, CVResult, Dataset

FOLDS_GROUP_CSV = OUTPUT_DIR / "leakage" / "folds_group.csv"
KM_PER_DEG_LAT = 111.32


# --------------------------------------------------------------------------- group labels
def merge_labels(*labels: np.ndarray) -> np.ndarray:
    """Union of several groupings over the same rows: rows sharing a label in ANY grouping end up in
    one merged group (connected components). NaN labels never link rows. Returns int labels."""
    n = len(labels[0])
    rows, cols, offset = [], [], n          # bipartite graph: row nodes 0..n-1, one node per label value
    for lab in labels:
        codes, uniq = pd.factorize(pd.Series(lab, dtype=object), use_na_sentinel=True)
        have = codes >= 0
        rows.append(np.flatnonzero(have)); cols.append(codes[have] + offset)
        offset += len(uniq)
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(offset, offset))
    _, comp = connected_components(g, directed=False)
    return pd.factorize(comp[:n])[0]


def spatial_blocks(df: pd.DataFrame, cell_km: float = 1.0) -> np.ndarray:
    """~cell_km x cell_km grid cell per city (string label); ads without coordinates -> city|district."""
    lat0 = df.groupby("city")["lat"].transform("min")
    lon0 = df.groupby("city")["lon"].transform("min")
    km_lon = KM_PER_DEG_LAT * np.cos(np.radians(df.groupby("city")["lat"].transform("mean")))
    gy = np.floor((df["lat"] - lat0) * KM_PER_DEG_LAT / cell_km)
    gx = np.floor((df["lon"] - lon0) * km_lon / cell_km)
    cell = df["city"] + "|" + gy.astype("Int64").astype(str) + "_" + gx.astype("Int64").astype(str)
    return np.where(df["lat"].notna(), cell, df["city"] + "|" + df["district"].astype(str)).astype(str)


def building_labels(df: pd.DataFrame, complex_id: pd.Series | None = None) -> np.ndarray:
    """Building groups: complex id (if given), rounded coordinates (4 decimals ~ 10 m) and
    street + house number, merged."""
    coords = np.where(df["lat"].notna(), df["city"] + "|" + df["lat"].round(4).astype(str) + "," + df["lon"].round(4).astype(str), None)
    addr = np.where(df["street"].notna() & df["house_num"].notna(),
                    df["city"] + "|" + df["street"].astype(str).str.lower() + "|" + df["house_num"].astype(str).str.lower(), None)
    labs = [pd.Series(coords, dtype=object), pd.Series(addr, dtype=object)]
    if complex_id is not None:
        labs.append(pd.Series(np.asarray(complex_id, dtype=object)))
    return merge_labels(*labs)


# --------------------------------------------------------------------------- folds
def random_folds(n: int, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n)))


def group_folds(groups: np.ndarray, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold(5) with shuffled group order (sklearn >= 1.6): every group lies in exactly one fold."""
    groups = np.asarray(groups)
    return list(GroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(len(groups)), groups=groups))


def repair_folds(fold_vec: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Minimal group-aware fix of given folds: every group with members in several folds is moved
    whole into the fold that holds most of its members (ties: the fold of its first row). Rows
    outside multi-fold groups keep their fold, so results stay paired with the original folds."""
    fv = np.asarray(fold_vec).copy()
    g = pd.Series(np.asarray(groups))
    for _, idx in g.groupby(g).groups.items():
        idx = np.sort(np.asarray(idx))
        if len(idx) > 1 and len(set(fv[idx])) > 1:
            counts = np.bincount(fv[idx])
            best = np.flatnonzero(counts == counts.max())
            fv[idx] = fv[idx[0]] if fv[idx[0]] in best else best[0]
    return fv


def fold_vector(folds, n: int) -> np.ndarray:
    v = np.full(n, -1, dtype=int)
    for k, (_, va) in enumerate(folds):
        v[va] = k
    assert (v >= 0).all()
    return v


def folds_from_vector(v: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    v = np.asarray(v)
    return [(np.flatnonzero(v != k), np.flatnonzero(v == k)) for k in range(int(v.max()) + 1)]


def load_group_folds(path: Path | str = FOLDS_GROUP_CSV, train: Dataset | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """The recommended Final folds (duplicate-group aware) as positional (train_idx, val_idx) pairs
    over ``core.load_train()`` rows. Raises if the file does not cover exactly the training ads."""
    train = train or core.load_train()
    f = pd.read_csv(path)
    fold = f.set_index(ID)["fold"].reindex(train.ids)
    if fold.isna().any() or len(f) != len(train):
        raise ValueError(f"{path} does not match the training ads")
    return folds_from_vector(fold.to_numpy(int))


# --------------------------------------------------------------------------- CV
def cv_with_folds(name: str, folds, train: Dataset | None = None, extra: pd.DataFrame | None = None,
                  feature_fn=None, model_factory=None, n_jobs: int = 1) -> CVResult:
    """``core.cv_evaluate`` with any folds (positional over ``load_train()`` rows)."""
    train = train or core.load_train()
    E = core.join_features(train.ids, extra) if extra is not None else None
    preds = Parallel(n_jobs=n_jobs)(delayed(core._fold_fit_predict)(train, tr, va, E, feature_fn, model_factory)
                                    for tr, va in folds)
    oof = np.empty(len(train))
    for (_, va), p in zip(folds, preds):
        oof[va] = p
    return core._result_from_oof(name, train.y, oof, folds)


# --------------------------------------------------------------------------- residual correction on any folds
def _dup_groups_for(train: Dataset) -> np.ndarray:
    """Duplicate-group labels of the training ads, as saved in folds_group.csv (training ads only)."""
    f = pd.read_csv(FOLDS_GROUP_CSV).set_index(ID)["dup_group"].reindex(train.ids)
    if f.isna().any():
        raise ValueError(f"{FOLDS_GROUP_CSV} does not cover the training ads")
    return f.to_numpy("int64")


def residual_inputs_with_folds(folds, train: Dataset | None = None, groups: np.ndarray | None = None,
                               n_jobs: int = 5) -> dict:
    """``core.base_residual_inputs`` for any outer folds. Inner OOF per outer training fold uses
    KFold(5, shuffle, 42) over those rows, repaired so a duplicate group never spans two inner folds
    (``groups``: duplicate labels over ``train`` rows; default = the saved duplicate groups). Cached."""
    train = train or core.load_train()
    groups = _dup_groups_for(train) if groups is None else np.asarray(groups)
    fv = fold_vector(folds, len(train))
    key = int(pd.util.hash_array(np.concatenate([fv, pd.factorize(groups)[0]])).sum() % 2**31)
    path = core.CACHE_DIR / f"residual_inputs_folds_{key}_{core._cache_key('resid', train)}.npz"
    if path.exists():
        z = np.load(path)
        return {"inner": [z[f"inner{k}"] for k in range(len(folds))], "outer": z["outer"]}

    def inner_oof(tr):
        X, y, g = train.X.iloc[tr].reset_index(drop=True), train.y[tr], groups[tr]
        inner = folds_from_vector(repair_folds(fold_vector(random_folds(len(tr)), len(tr)), g))
        oof = np.empty(len(tr))
        for itr, iva in inner:
            oof[iva] = np.log(core.make_base_model().fit(X.iloc[itr], y[itr]).predict(X.iloc[iva]))
        return oof

    inner = Parallel(n_jobs=n_jobs)(delayed(inner_oof)(tr) for tr, _ in folds)
    outer = np.log(cv_with_folds("base", folds, train, n_jobs=n_jobs).oof_pred)
    core.CACHE_DIR.mkdir(exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, outer=outer, **{f"inner{k}": a for k, a in enumerate(inner)})
    os.replace(tmp, path)
    return {"inner": inner, "outer": outer}


def cv_residual_with_folds(corrector, folds, name: str = "base + residual", train: Dataset | None = None,
                           n_jobs: int = 1) -> CVResult:
    """``core.cv_evaluate_residual`` on any folds (e.g. ``load_group_folds()``); same corrector contract.
    Inner residuals are cross-fitted with duplicate-aware inner folds (``residual_inputs_with_folds``)."""
    train = train or core.load_train()
    base = residual_inputs_with_folds(folds, train)
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
    return core._result_from_oof(name, train.y, oof, folds)
