"""Integration stage of the Final: every feature track fused into one model, chosen by CV.

Pipeline (``scripts/run_final.py`` drives it, outputs go to ``outputs/final/``):

1. ``load_blocks``: every available per-ad block, keyed by ad id.
       geo         outputs/geo/features_geo.csv            static (OSM distances / counts)
       rawfields   outputs/geo/features_rawfields.csv      static (unused raw ad-page fields)
       photo       outputs/photos/features_photo.csv       static (falls back to clip_zero_shot_ad.csv)
       text_flags  outputs/text/features_text.csv          static (regex flags; ``oof_*`` columns dropped)
       text        the description itself                  TF-IDF, fitted inside every fold
       text_emb    cache/text_emb_<key>.npy                sentence embedding (best key of outputs/text/cv.csv)
       photo_emb   cache/clip_emb.npz (mean per ad)        image embedding (falls back to final/photo_embeddings.npz)
   A missing file only removes its block (with a message). Columns that exist for training ads but
   not for the test ads (out-of-fold predictions of a track) are dropped: a final model cannot use them.
   A track can veto its block with ``outputs/<track>/decision.json`` ({"keep": false} or
   {"<block>": {"keep": false}}); vetoed blocks are still measured but never selected.

2. Three fusion strategies, each a fit/predict model on a "frame" (clean columns + block columns):
       A  ``ResidualStackModel``  base Endterm HGB + cross-fitted log-residual corrections, one per block
          (HGB for static blocks, Ridge for TF-IDF and embeddings), blended with non-negative weights
          learned on inner out-of-fold corrections (like legacy_final, but with honest residuals).
       B  ``FeatureFusionModel``  blocks as extra columns of the same HGB: static columns as they are,
          embeddings PCA-reduced inside the fold, text as an inner-OOF TF-IDF price prior (1 column).
       C  ``NeuralFusionModel``   a small PyTorch network: tabular branch + text-embedding branch +
          photo-embedding branch, L1 loss on standardised log price, early stopping (MPS/CPU).
   ``StackingCV`` / ``FeatureCV`` cache the per-fold pieces so forward selection over many block
   sets costs only the final blend / HGB fit.

3. ``forward_select``: greedy forward selection with the team 1-SD rule (a block is kept only if it
   lowers the current CV MAE by more than one fold SD of the current model, never less than the fixed
   Endterm margin RULE_MARGIN_M = 0.3535 M).

4. ``bootstrap_metrics`` / ``segment_table``: the ONE test evaluation (2,000 paired resamples).

5. ``reduced_features`` / ``make_reduced_model``: November hold-out (temporal drift) on the fields
   the November listing cards carry.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from . import core
from .core import BOOL, CACHE_DIR, CAT, FEATURES, ID, NUM, OUTPUT_DIR, ROOT, SEED, CVResult

FINAL_DIR = OUTPUT_DIR / "final"
MODELS_DIR = ROOT / "models"
FOLDS_GROUP_CSV = OUTPUT_DIR / "leakage" / "folds_group.csv"
DUP_GROUPS_CSV = OUTPUT_DIR / "leakage" / "duplicate_groups.csv"
NOV_CSV = ROOT / "data" / "clean" / "krisha_nov_holdout.csv"

BLOCK_ORDER = ["geo", "rawfields", "photo", "text_flags", "text", "text_emb", "photo_emb"]
BLOCK_KIND = {"geo": "tab", "rawfields": "tab", "photo": "tab", "text_flags": "tab",
              "text": "text", "text_emb": "emb", "photo_emb": "emb"}
TRACK_OF = {"geo": "geo", "rawfields": "geo", "photo": "photos", "photo_emb": "photos",
            "text": "text", "text_flags": "text", "text_emb": "text"}
TEXT_EMB_KEYS = ["user-base", "e5-base", "rubert-tiny2"]
# columns a track may write next to its features that are not features
NON_FEATURE_COLS = {"fold", "split", "group", "group_size", "dup_group", "dup_group_size", "price", "y",
                    "target", "log_price", "base_log_pred", "pred", "prediction", "residual"}
SEP = "__"   # block column names are "<block>__<column>"
# The team 1-SD rule uses a FIXED margin: the Endterm base fold SD (0.3535 M). On the duplicate-aware
# folds the base fold SD is 0.264, but the fold SD itself moves 0.26-0.50 with the fold seed
# (outputs/leakage/cv_schemes_runs.csv), so base.sd on one fold set would make the rule looser by chance
# (the earlier threshold 6.459 = 6.723 - 0.264 was this mistake). Threshold = base.mean - RULE_MARGIN_M.
RULE_MARGIN_M = float(core.ENDTERM_REFERENCE["cv_mae_sd_M"])


def rule_margin(base: CVResult) -> float:
    """Margin of the base rule on any fold set: never smaller than the Endterm 1-SD margin."""
    return max(RULE_MARGIN_M, float(base.sd))


# --------------------------------------------------------------------------- settings
@dataclass
class Settings:
    """Every knob of the integration run (saved in the model metadata)."""
    hgb_overrides: dict = field(default_factory=dict)        # on top of core.HGB_PARAMS (quick: fewer trees)
    inner_folds: int = 5                                     # cross-fitting inside a training fold
    corr_hgb: dict = field(default_factory=lambda: dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                        min_samples_leaf=40, l2_regularization=1.0))
    tfidf_min_df: int = 3
    tfidf_max_features: int = 50000
    tfidf_char: bool = False      # word 1-2 only: char n-grams add ~0.01 M (text track) at 5x the cost
    text_alpha: float = 5.0       # the text track's per-fold GCV choice for word TF-IDF (4.6)
    emb_alphas: tuple = tuple(float(a) for a in np.logspace(0, 5, 11))
    pca_components: int = 16
    nn_epochs: int = 250
    nn_patience: int = 20
    nn_batch: int = 256
    nn_lr: float = 2e-3
    nn_weight_decay: float = 1e-4
    nn_seeds: int = 2                 # ensemble of seeds (mean of log predictions)
    nn_device: str = "cpu"            # measured: MPS is no faster for this small MLP; "auto" picks MPS
    nn_threads: int = 1               # torch threads per process (CV folds run in parallel on CPU)
    n_jobs: int = 3
    n_boot: int = 2000

    @classmethod
    def quick(cls) -> "Settings":
        return cls(hgb_overrides={"max_iter": 40}, inner_folds=3, corr_hgb=dict(max_iter=30, learning_rate=0.1,
                   max_leaf_nodes=7, min_samples_leaf=10), tfidf_min_df=1, tfidf_max_features=3000,
                   pca_components=4, nn_epochs=8, nn_patience=3, nn_seeds=1, nn_device="cpu", n_jobs=1, n_boot=200)


# --------------------------------------------------------------------------- blocks
@dataclass
class Block:
    name: str
    kind: str                                   # "tab" | "text" | "emb"
    source: str
    table: pd.DataFrame | None = field(default=None, repr=False)   # id-indexed, columns "<name>__<col>"
    keep: bool | None = None                    # a track's own decision (None = no decision file)
    notes: list[str] = field(default_factory=list)
    cols: list[str] | None = field(default=None, repr=False)     # column names of a table-less (light) copy
    _digest: str | None = field(default=None, repr=False)

    @property
    def columns(self) -> list[str]:
        if self.table is None:
            return list(self.cols or [])
        return list(self.table.columns)

    def light(self) -> "Block":
        """The same block without its table (models keep only column names; the frame carries the values)."""
        return Block(self.name, self.kind, self.source, None, self.keep, list(self.notes), self.columns, self.digest())

    def digest(self) -> str:
        if self._digest is not None:
            return self._digest
        if self.table is None:
            return "text:description"
        h = hashlib.sha256(pd.util.hash_pandas_object(self.table, index=True).to_numpy().tobytes())
        h.update(",".join(self.table.columns).encode())
        return h.hexdigest()[:16]

    def summary(self) -> dict:
        return {"block": self.name, "kind": self.kind, "source": self.source, "n_cols": len(self.columns),
                "n_ads": None if self.table is None else int(len(self.table)), "keep_decision": self.keep,
                "notes": "; ".join(self.notes), "digest": self.digest()}


def read_id_table(path: Path | str, name: str, test_ids: Iterable[int] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """An id-keyed feature CSV as an id-indexed frame with "<name>__" column names.

    Drops: duplicated ids (first kept), bookkeeping columns, ``oof_*`` columns, columns that are filled
    for training ads but (almost) empty for the test ads (out-of-fold values: unusable in a final
    model), all-empty and constant columns."""
    notes = []
    df = pd.read_csv(path)
    if ID not in df.columns:
        raise ValueError(f"{path}: no '{ID}' column")
    df[ID] = pd.to_numeric(df[ID], errors="coerce")
    df = df.dropna(subset=[ID])
    df[ID] = df[ID].astype("int64")
    if df[ID].duplicated().any():
        notes.append(f"{int(df[ID].duplicated().sum())} duplicated ids dropped")
        df = df.drop_duplicates(ID, keep="first")
    t = df.set_index(ID)
    drop = [c for c in t.columns if c.lower() in NON_FEATURE_COLS or c.lower().startswith("oof")]
    if test_ids is not None:
        te = t.index.isin(np.asarray(list(test_ids), dtype="int64"))
        if te.any() and (~te).any():
            for c in t.columns:
                fill_tr, fill_te = t.loc[~te, c].notna().mean(), t.loc[te, c].notna().mean()
                if c not in drop and fill_tr > 0.2 and fill_te < 0.5 * fill_tr:
                    drop.append(c)
    drop += [c for c in t.columns if c not in drop and t[c].nunique(dropna=True) <= 1]
    if drop:
        notes.append(f"dropped {len(drop)} non-feature/train-only/constant columns: {', '.join(map(str, drop[:8]))}"
                     + (" ..." if len(drop) > 8 else ""))
    t = t.drop(columns=drop)
    for c in t.columns:   # bool-like strings -> numbers so they go to the numeric branch
        if t[c].dtype == object and set(t[c].dropna().astype(str).str.lower().unique()) <= {"true", "false"}:
            t[c] = t[c].astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    t.columns = [f"{name}{SEP}{c}" for c in t.columns]
    return t, notes


def _best_text_key(out_dir: Path = OUTPUT_DIR) -> list[str]:
    """Embedding keys ordered by the text track's CV (best residual Ridge first), then the defaults."""
    order = []
    p = out_dir / "text" / "cv.csv"
    if p.exists():
        cv = pd.read_csv(p)
        if {"variant", "cv_mae_M"} <= set(cv.columns):
            emb = cv[cv["variant"].str.contains("embedding", case=False, na=False)].sort_values("cv_mae_M")
            for v in emb["variant"]:
                for k in TEXT_EMB_KEYS:
                    if str(v).lower().startswith(k) and k not in order:
                        order.append(k)
    return order + [k for k in TEXT_EMB_KEYS if k not in order]


def load_text_embedding(cache_dir: Path = CACHE_DIR, out_dir: Path = OUTPUT_DIR) -> tuple[pd.DataFrame, str] | None:
    ids_path = cache_dir / "text_emb_ids.npy"
    if not ids_path.exists():
        return None
    ids = np.load(ids_path).astype("int64")
    for key in _best_text_key(out_dir):
        p = cache_dir / f"text_emb_{key}.npy"
        if p.exists():
            emb = np.load(p).astype(np.float32)
            if len(emb) == len(ids):
                cols = [f"text_emb{SEP}e{i}" for i in range(emb.shape[1])]
                return pd.DataFrame(emb, index=pd.Index(ids, name=ID), columns=cols), key
    return None


def load_photo_embedding(cache_dir: Path = CACHE_DIR, root: Path = ROOT) -> tuple[pd.DataFrame, str] | None:
    """Ad-level photo embedding: mean of the per-photo CLIP embeddings, L2-normalised (ads without a
    usable photo are absent -> NaN after joining). Falls back to the legacy MobileNetV3 file."""
    p = cache_dir / "clip_emb.npz"
    if p.exists():
        z = np.load(p)
        emb, ad = z["emb"].astype(np.float32), z["ad_id"].astype("int64")
        ids, inv = np.unique(ad, return_inverse=True)
        m = np.zeros((len(ids), emb.shape[1]), np.float64)
        np.add.at(m, inv, emb)
        m /= np.bincount(inv)[:, None]
        m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        cols = [f"photo_emb{SEP}e{i}" for i in range(m.shape[1])]
        return pd.DataFrame(m.astype(np.float32), index=pd.Index(ids, name=ID), columns=cols), "clip_emb.npz (CLIP ViT-B-32, mean per ad)"
    p = root / "final" / "photo_embeddings.npz"
    if p.exists():
        z = np.load(p)
        ok = z["valid_photos"] > 0
        cols = [f"photo_emb{SEP}e{i}" for i in range(z["features"].shape[1])]
        return (pd.DataFrame(z["features"][ok].astype(np.float32), index=pd.Index(z["ids"][ok].astype("int64"), name=ID),
                             columns=cols), "final/photo_embeddings.npz (MobileNetV3-small, legacy)")
    return None


def _walk_keep(obj, block_names: Sequence[str], track_blocks: Sequence[str], out: dict) -> None:
    if isinstance(obj, dict):
        if isinstance(obj.get("keep"), bool):
            for b in track_blocks:
                out.setdefault(b, obj["keep"])
        for k, v in obj.items():
            if k in block_names:
                if isinstance(v, bool):
                    out[k] = v
                elif isinstance(v, dict) and isinstance(v.get("keep"), bool):
                    out[k] = v["keep"]
            elif isinstance(v, (dict, list)) and k != "keep":
                _walk_keep(v, block_names, [], out)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and item.get("block") in block_names and isinstance(item.get("keep"), bool):
                out[item["block"]] = item["keep"]


def read_keep_decisions(out_dir: Path = OUTPUT_DIR) -> dict[str, bool]:
    """Keep/drop decisions written by the tracks (decision.json / keep.json / decision.csv)."""
    decisions: dict[str, bool] = {}
    for track in sorted(set(TRACK_OF.values())):
        blocks = [b for b, t in TRACK_OF.items() if t == track]
        for fname in ("decision.json", "decisions.json", "keep.json"):
            p = out_dir / track / fname
            if p.exists():
                try:
                    found: dict[str, bool] = {}
                    _walk_keep(json.loads(p.read_text()), BLOCK_ORDER, blocks, found)
                    decisions.update(found)
                except json.JSONDecodeError:
                    warnings.warn(f"unreadable decision file {p}")
        for fname in ("decision.csv", "keep.csv"):
            p = out_dir / track / fname
            if p.exists():
                d = pd.read_csv(p)
                key = "block" if "block" in d.columns else ("name" if "name" in d.columns else None)
                if key and "keep" in d.columns:
                    for b, k in zip(d[key], d["keep"]):
                        if b in BLOCK_ORDER:
                            decisions[b] = str(k).strip().lower() in ("true", "1", "yes")
    return decisions


def load_blocks(names: Sequence[str] | None = None, out_dir: Path = OUTPUT_DIR, cache_dir: Path = CACHE_DIR,
                test_ids: Iterable[int] | None = None, root: Path = ROOT) -> tuple[dict[str, Block], list[str]]:
    """All available blocks (ordered as BLOCK_ORDER) and one message per block (loaded / skipped)."""
    names = list(names or BLOCK_ORDER)
    test_ids = core.load_test_ids() if test_ids is None else test_ids
    decisions = read_keep_decisions(out_dir)
    blocks, msgs = {}, []
    static = {"geo": [out_dir / "geo" / "features_geo.csv"],
              "rawfields": [out_dir / "geo" / "features_rawfields.csv"],
              "photo": [out_dir / "photos" / "features_photo.csv", out_dir / "photos" / "clip_zero_shot_ad.csv"],
              "text_flags": [out_dir / "text" / "features_text.csv"]}
    for name in names:
        try:
            if name in static:
                path = next((p for p in static[name] if p.exists()), None)
                if path is None:
                    msgs.append(f"[skip] {name}: {static[name][0].relative_to(out_dir.parent) if out_dir.parent in static[name][0].parents else static[name][0]} not written yet")
                    continue
                table, notes = read_id_table(path, name, test_ids)
                if path != static[name][0]:
                    notes.insert(0, f"fallback source ({static[name][0].name} missing)")
                if table.shape[1] == 0:
                    msgs.append(f"[skip] {name}: no usable columns in {path.name} ({'; '.join(notes)})")
                    continue
                blocks[name] = Block(name, "tab", str(path.name), table, notes=notes)
            elif name == "text":
                blocks[name] = Block(name, "text", "description (TF-IDF fitted per fold)")
            elif name == "text_emb":
                got = load_text_embedding(cache_dir, out_dir)
                if got is None:
                    msgs.append("[skip] text_emb: no cache/text_emb_<key>.npy"); continue
                blocks[name] = Block(name, "emb", f"text_emb_{got[1]}.npy", got[0])
            elif name == "photo_emb":
                got = load_photo_embedding(cache_dir, root)
                if got is None:
                    msgs.append("[skip] photo_emb: no photo embedding cache"); continue
                blocks[name] = Block(name, "emb", got[1], got[0])
            else:
                raise KeyError(name)
        except (OSError, ValueError, pd.errors.ParserError) as e:   # a half-written file must not stop the run
            msgs.append(f"[skip] {name}: unreadable ({type(e).__name__}: {e})")
            continue
        if name in decisions:
            blocks[name].keep = decisions[name]
        b = blocks[name]
        msgs.append(f"[ok]   {name}: {b.kind}, {len(b.columns) or 'text'} cols from {b.source}"
                    + ("" if b.keep is None else f", track decision keep={b.keep}")
                    + (f" ({'; '.join(b.notes)})" if b.notes else ""))
    return blocks, msgs


def attach_blocks(df: pd.DataFrame, blocks: dict[str, Block] | Iterable[Block]) -> pd.DataFrame:
    """Clean rows + every block column (joined by id; ads missing from a block get NaN) + the
    sanitised description used by TF-IDF. Keeps the row order and a RangeIndex."""
    from . import raw
    bl = blocks.values() if isinstance(blocks, dict) else blocks
    df = df.reset_index(drop=True)
    parts = [df]
    for b in bl:
        if b.table is not None:
            parts.append(core.join_features(df[ID].to_numpy("int64"), b.table))
    out = pd.concat(parts, axis=1)
    if "text_clean" not in out.columns:
        out["text_clean"] = out.get("description", pd.Series("", index=out.index)).fillna("").map(raw.sanitize_text)
    return out


# --------------------------------------------------------------------------- folds
@dataclass
class FoldScheme:
    name: str
    folds: list
    groups: np.ndarray | None = None   # duplicate-group label per row (used for inner folds), None = random

    def key(self, ids: np.ndarray) -> str:
        h = hashlib.sha256(self.name.encode())
        h.update(np.asarray(ids, "int64").tobytes())
        for _, va in self.folds:
            h.update(np.asarray(va, "int64").tobytes())
        if self.groups is not None:   # inner (cross-fitting) folds depend on the duplicate groups too
            h.update(pd.Series(self.groups).astype(str).str.cat(sep="|").encode())
        return h.hexdigest()[:12]


def fold_schemes(ids: np.ndarray, path: Path = FOLDS_GROUP_CSV, full_train: bool = True) -> tuple[dict[str, FoldScheme], list[str]]:
    """``random`` (the Endterm KFold) always; ``group`` (duplicate-group aware) when the leakage track
    wrote folds_group.csv covering these ads. For a subset (quick mode) the folds are restricted to it."""
    ids = np.asarray(ids, "int64")
    msgs = []
    from sklearn.model_selection import KFold as _KF
    rnd = core.cv_folds(len(ids)) if full_train else list(_KF(5, shuffle=True, random_state=SEED).split(ids))
    out = {"random": FoldScheme("random", rnd)}
    if path.exists():
        try:
            f = pd.read_csv(path)
            f[ID] = f[ID].astype("int64")
            f = f.drop_duplicates(ID).set_index(ID)
            fold = f["fold"].reindex(ids)
            if fold.isna().any():
                msgs.append(f"[skip] group folds: {int(fold.isna().sum())} ads missing from {path.name}")
            else:
                v = fold.to_numpy(int)
                ks = sorted(set(v))
                folds = [(np.flatnonzero(v != k), np.flatnonzero(v == k)) for k in ks]
                gcol = "dup_group" if "dup_group" in f.columns else ("group" if "group" in f.columns else None)
                groups = f[gcol].reindex(ids).to_numpy() if gcol else None
                out["group"] = FoldScheme("group", folds, groups)
                msgs.append(f"[ok]   group folds: {len(folds)} folds from {path.name}"
                            + (f", {int(pd.Series(groups).duplicated(keep=False).sum())} ads in duplicate groups" if groups is not None else ""))
        except (OSError, KeyError, ValueError) as e:
            msgs.append(f"[skip] group folds: unreadable ({e})")
    else:
        msgs.append("[skip] group folds: outputs/leakage/folds_group.csv not written yet (random folds only)")
    return out, msgs


def inner_folds(n: int, groups: np.ndarray | None, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-fitting folds inside a training set: GroupKFold by duplicate group when groups are given,
    else KFold(shuffle, seed 42) (= the inner folds of core.base_residual_inputs)."""
    if groups is not None and len(np.unique(groups)) >= k:
        return list(GroupKFold(k, shuffle=True, random_state=SEED).split(np.arange(n), groups=groups))
    return list(KFold(k, shuffle=True, random_state=SEED).split(np.arange(n)))


# --------------------------------------------------------------------------- shared helpers
def _base_model(settings: Settings, num=(), cat=()):
    return core.make_base_model(list(num), list(cat), **settings.hgb_overrides)


def _tab_frame(frame: pd.DataFrame, cols: Sequence[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    E = frame[list(cols)].reset_index(drop=True)
    num, cat = core.split_columns(E)
    return core._prepare_extra(E, num, cat), num, cat


def _oof_log(X: pd.DataFrame, y: np.ndarray, folds, settings: Settings) -> np.ndarray:
    oof = np.empty(len(y))
    for tr, va in folds:
        oof[va] = np.log(_base_model(settings).fit(X.iloc[tr], y[tr]).predict(X.iloc[va]))
    return oof


def _emb_matrix(frame: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    M = frame[list(cols)].to_numpy(np.float64)
    ok = np.isfinite(M).all(1)
    return M, ok


def make_tfidf(settings: Settings) -> FeatureUnion:
    common = dict(sublinear_tf=True, min_df=settings.tfidf_min_df, max_features=settings.tfidf_max_features, dtype=np.float32)
    parts = [("word", TfidfVectorizer(ngram_range=(1, 2), **common))]
    if settings.tfidf_char:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common)))
    return FeatureUnion(parts)


def _ridge(settings: Settings) -> Ridge:
    return Ridge(alpha=settings.text_alpha, solver="sparse_cg", tol=1e-6, max_iter=2000)


# --------------------------------------------------------------------------- level-1 correctors (strategy A)
class TabCorrector:
    """Small regularised HGB on the log residual, inputs = block columns + base log prediction."""

    def __init__(self, cols, settings: Settings):
        self.cols, self.settings = list(cols), settings

    def fit(self, frame, resid):
        E, num, cat = _tab_frame(frame, self.cols)
        E["base_log_pred"] = frame["base_log_pred"].to_numpy()
        self.num_, self.cat_ = num + ["base_log_pred"], cat
        pre = ColumnTransformer([("num", "passthrough", self.num_),
                                 ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cat)])
        self.model_ = Pipeline([("pre", pre), ("hgb", HistGradientBoostingRegressor(**self.settings.corr_hgb, random_state=SEED,
                                                                                    early_stopping=False))]).fit(E, resid)
        return self

    def predict(self, frame):
        E = core._prepare_extra(frame[self.cols].reset_index(drop=True), [c for c in self.num_ if c != "base_log_pred"], self.cat_)
        E["base_log_pred"] = frame["base_log_pred"].to_numpy()
        return self.model_.predict(E)


class TextCorrector:
    """TF-IDF of the sanitised description (word 1-2 grams) + Ridge on the log residual."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def fit(self, frame, resid):
        self.vec_ = make_tfidf(self.settings).fit(frame["text_clean"])
        self.ridge_ = _ridge(self.settings).fit(self.vec_.transform(frame["text_clean"]), resid)
        return self

    def predict(self, frame):
        return self.ridge_.predict(self.vec_.transform(frame["text_clean"]))


class EmbCorrector:
    """Standardised embedding + RidgeCV on the log residual; ads without the embedding get 0."""

    def __init__(self, cols, settings: Settings):
        self.cols, self.settings = list(cols), settings

    def fit(self, frame, resid):
        M, ok = _emb_matrix(frame, self.cols)
        self.ok_ = bool(ok.sum() >= 10)
        if self.ok_:
            self.model_ = Pipeline([("sc", StandardScaler()), ("ridge", RidgeCV(alphas=self.settings.emb_alphas))]).fit(M[ok], resid[ok])
        return self

    def predict(self, frame):
        M, ok = _emb_matrix(frame, self.cols)
        out = np.zeros(len(M))
        if self.ok_ and ok.any():
            out[ok] = self.model_.predict(M[ok])
        return out


def make_corrector(block: Block, settings: Settings):
    if block.kind == "tab":
        return TabCorrector(block.columns, settings)
    if block.kind == "text":
        return TextCorrector(settings)
    return EmbCorrector(block.columns, settings)


def _block_cols(block: Block) -> list[str]:
    return ["text_clean"] if block.kind == "text" else block.columns


def level1(block: Block, frame_tr: pd.DataFrame, resid_tr: np.ndarray, frame_new: pd.DataFrame | None,
           folds, settings: Settings):
    """Inner out-of-fold corrections on the training rows (for the blend weights) and the corrector
    fitted on all training rows (its predictions for ``frame_new`` when given).

    For TF-IDF the vocabulary/IDF is fitted once on all of ``frame_tr`` (unsupervised, no prices) and
    only the Ridge is cross-fitted: 1 vectoriser instead of k+1 per outer fold."""
    if block.kind == "text":
        c = TextCorrector(settings)
        c.vec_ = make_tfidf(settings).fit(frame_tr["text_clean"])
        M = c.vec_.transform(frame_tr["text_clean"])
        z_tr = np.zeros(len(resid_tr))
        for itr, iva in folds:
            z_tr[iva] = _ridge(settings).fit(M[itr], resid_tr[itr]).predict(M[iva])
        c.ridge_ = _ridge(settings).fit(M, resid_tr)
        return z_tr, (None if frame_new is None else c.predict(frame_new)), c
    z_tr = np.zeros(len(resid_tr))
    for itr, iva in folds:
        c = make_corrector(block, settings).fit(frame_tr.iloc[itr].reset_index(drop=True), resid_tr[itr])
        z_tr[iva] = c.predict(frame_tr.iloc[iva].reset_index(drop=True))
    c = make_corrector(block, settings).fit(frame_tr, resid_tr)
    return z_tr, (None if frame_new is None else c.predict(frame_new)), c


def blend_weights(Z: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """Non-negative, no-intercept least squares of the residual on the block corrections."""
    if Z.shape[1] == 0:
        return np.zeros(0)
    return LinearRegression(positive=True, fit_intercept=False).fit(Z, resid).coef_


class ResidualStackModel:
    """Strategy A: base Endterm HGB + blended cross-fitted residual corrections (log scale)."""
    strategy = "A"

    def __init__(self, blocks: Sequence[Block], settings: Settings | None = None):
        self.blocks, self.settings = [b.light() for b in blocks], settings or Settings()

    def fit(self, frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None):
        frame = frame.reset_index(drop=True)
        X = core.encode_features(frame)
        self.base_ = _base_model(self.settings).fit(X, y)
        folds = inner_folds(len(y), groups, self.settings.inner_folds)
        base_oof = _oof_log(X, y, folds, self.settings)
        resid = np.log(y) - base_oof
        fr = frame.assign(base_log_pred=base_oof)
        Z, self.correctors_ = [], []
        for b in self.blocks:
            z, _, c = level1(b, fr[_block_cols(b) + ["base_log_pred", ID]], resid, None, folds, self.settings)
            Z.append(z); self.correctors_.append(c)
        self.weights_ = blend_weights(np.column_stack(Z) if Z else np.zeros((len(y), 0)), resid)
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.reset_index(drop=True)
        base_log = np.log(self.base_.predict(core.encode_features(frame)))
        fr = frame.assign(base_log_pred=base_log)
        delta = np.zeros(len(frame))
        for w, c in zip(self.weights_, self.correctors_):
            if w > 0:
                delta += w * c.predict(fr)
        return base_log + delta

    def predict(self, frame):
        return np.exp(self.predict_log(frame))


# --------------------------------------------------------------------------- block transformers (strategy B)
class TabSelect:
    def __init__(self, block: Block):
        self.cols = block.columns

    def fit_train(self, frame, y, folds, settings):
        return frame[self.cols].reset_index(drop=True)

    def transform(self, frame):
        return frame[self.cols].reset_index(drop=True)


class EmbPCA:
    """PCA of an embedding block, fitted on the training rows that have it (others stay NaN)."""

    def __init__(self, block: Block):
        self.cols, self.prefix = block.columns, f"{block.name}{SEP}pc"

    def fit_train(self, frame, y, folds, settings):
        M, ok = _emb_matrix(frame, self.cols)
        k = int(min(settings.pca_components, max(ok.sum() - 1, 1), len(self.cols)))
        self.pca_ = PCA(k, random_state=SEED).fit(M[ok])
        return self.transform(frame)

    def transform(self, frame):
        M, ok = _emb_matrix(frame, self.cols)
        out = np.full((len(M), self.pca_.n_components_), np.nan)
        if ok.any():
            out[ok] = self.pca_.transform(M[ok])
        return pd.DataFrame(out, columns=[f"{self.prefix}{i}" for i in range(out.shape[1])])


class TextPrior:
    """One column: TF-IDF + RidgeCV prediction of log price. Inner out-of-fold on the training rows
    (so the HGB never sees a value fitted on the same ad), full fit for new rows."""
    col = f"text{SEP}tfidf_log_price"

    def __init__(self, block: Block | None = None):
        pass

    def fit_train(self, frame, y, folds, settings):
        frame = frame.reset_index(drop=True)
        self.vec_ = make_tfidf(settings).fit(frame["text_clean"])   # unsupervised: once on the training rows
        M, ly = self.vec_.transform(frame["text_clean"]), np.log(y)
        z = np.empty(len(y))
        for itr, iva in folds:
            z[iva] = _ridge(settings).fit(M[itr], ly[itr]).predict(M[iva])
        self.ridge_ = _ridge(settings).fit(M, ly)
        return pd.DataFrame({self.col: z})

    def transform(self, frame):
        return pd.DataFrame({self.col: self.ridge_.predict(self.vec_.transform(frame["text_clean"].reset_index(drop=True)))})


def make_block_transformer(block: Block):
    return {"tab": TabSelect, "emb": EmbPCA, "text": TextPrior}[block.kind](block)


class FeatureFusionModel:
    """Strategy B: blocks as extra columns of the Endterm HGB."""
    strategy = "B"

    def __init__(self, blocks: Sequence[Block], settings: Settings | None = None):
        self.blocks, self.settings = [b.light() for b in blocks], settings or Settings()

    def fit(self, frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None):
        frame = frame.reset_index(drop=True)
        folds = inner_folds(len(y), groups, self.settings.inner_folds)
        self.transformers_ = [make_block_transformer(b) for b in self.blocks]
        parts = [t.fit_train(frame, y, folds, self.settings) for t in self.transformers_]
        X, self.num_, self.cat_ = _stack_extra(core.encode_features(frame), parts)
        self.model_ = _base_model(self.settings, self.num_, self.cat_).fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, frame):
        frame = frame.reset_index(drop=True)
        parts = [t.transform(frame) for t in self.transformers_]
        X = _stack_extra_known(core.encode_features(frame), parts, self.num_, self.cat_)
        return self.model_.predict(X)


def _stack_extra(X: pd.DataFrame, parts: list[pd.DataFrame]):
    if not parts:
        return X, [], []
    E = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    num, cat = core.split_columns(E)
    return pd.concat([X, core._prepare_extra(E, num, cat)], axis=1), num, cat


def _stack_extra_known(X, parts, num, cat):
    if not parts:
        return X
    E = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    return pd.concat([X, core._prepare_extra(E, num, cat)], axis=1)


# --------------------------------------------------------------------------- neural fusion (strategy C)
def torch_device(pref: str = "auto") -> str:
    import torch
    if pref == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return pref


def _make_net(d_tab: int, d_txt: int, d_img: int, p: float = 0.15):
    import torch
    from torch import nn

    class FusionNet(nn.Module):
        """Tabular MLP + optional text/photo embedding branches, late fusion, one output (z-scored log price)."""

        def __init__(self):
            super().__init__()
            self.tab = nn.Sequential(nn.Linear(d_tab, 256), nn.GELU(), nn.Dropout(p), nn.Linear(256, 128), nn.GELU())
            self.txt = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_txt, 64), nn.GELU()) if d_txt else None
            self.img = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_img, 64), nn.GELU()) if d_img else None
            d = 128 + (64 if d_txt else 0) + (64 if d_img else 0)
            self.head = nn.Sequential(nn.Dropout(p), nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, xt, xe=None, xi=None):
            h = [self.tab(xt)]
            if self.txt is not None:
                h.append(self.txt(xe))
            if self.img is not None:
                h.append(self.img(xi))
            return self.head(torch.cat(h, 1)).squeeze(1)

    return FusionNet()


class _EmbBranch:
    """Standardised embedding with NaN rows -> 0; returns (matrix, missing flag)."""

    def fit(self, frame, cols):
        self.cols = list(cols)
        M, ok = _emb_matrix(frame, self.cols)
        self.sc_ = StandardScaler().fit(M[ok] if ok.sum() > 1 else np.zeros((2, len(self.cols))))
        return self

    def transform(self, frame):
        M, ok = _emb_matrix(frame, self.cols)
        out = np.zeros_like(M, dtype=np.float32)
        if ok.any():
            out[ok] = self.sc_.transform(M[ok])
        return out, (~ok).astype(np.float32)


class NeuralFusionModel:
    """Strategy C: small multimodal network on log price (tabular + text-embedding + photo-embedding
    branches). TF-IDF ("text") is not an input: the text branch uses the sentence embedding."""
    strategy = "C"

    def __init__(self, blocks: Sequence[Block], settings: Settings | None = None, val_share: float = 0.15):
        self.blocks, self.settings, self.val_share = [b.light() for b in blocks], settings or Settings(), val_share

    def _tabular(self, frame, fit: bool):
        X = core.encode_features(frame)
        tab_cols = [c for b in self.blocks if b.kind == "tab" for c in b.columns]
        if tab_cols:
            if fit:
                _, self.tnum_, self.tcat_ = _tab_frame(frame, tab_cols)
            E = core._prepare_extra(frame[tab_cols].reset_index(drop=True), self.tnum_, self.tcat_)
            X = pd.concat([X, E], axis=1)
        else:
            self.tnum_, self.tcat_ = [], []
        if fit:
            n = len(X)
            num = NUM + BOOL + self.tnum_
            self.pre_ = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                  ("q", QuantileTransformer(n_quantiles=min(1000, n), output_distribution="normal",
                                                            random_state=SEED))]), num),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), CAT + self.tcat_)])
            self.pre_.fit(X)
        return self.pre_.transform(X).astype(np.float32)

    def _inputs(self, frame, fit: bool):
        xt = self._tabular(frame, fit)
        branches = {}
        for name in ("text_emb", "photo_emb"):
            b = next((b for b in self.blocks if b.name == name), None)
            if b is None:
                continue
            if fit:
                setattr(self, f"{name}_", _EmbBranch().fit(frame, b.columns))
            M, miss = getattr(self, f"{name}_").transform(frame)
            branches[name] = M
            xt = np.column_stack([xt, miss])
        return xt, branches.get("text_emb"), branches.get("photo_emb")

    def fit(self, frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None):
        import torch
        frame = frame.reset_index(drop=True)
        xt, xe, xi = self._inputs(frame, fit=True)
        ly = np.log(y)
        self.mu_, self.sd_ = float(ly.mean()), float(ly.std())
        z = ((ly - self.mu_) / self.sd_).astype(np.float32)
        rng = np.random.default_rng(SEED)
        if groups is not None:          # duplicates never straddle the early-stopping split
            ug = np.unique(groups)
            val_g = rng.choice(ug, max(1, int(len(ug) * self.val_share)), replace=False)
            is_val = np.isin(groups, val_g)
        else:
            is_val = np.zeros(len(y), bool)
            is_val[rng.choice(len(y), max(1, int(len(y) * self.val_share)), replace=False)] = True
        dev = torch_device(self.settings.nn_device)
        if dev == "cpu":
            torch.set_num_threads(self.settings.nn_threads)
        self.dims_ = (xt.shape[1], 0 if xe is None else xe.shape[1], 0 if xi is None else xi.shape[1])
        T = lambda a: None if a is None else torch.as_tensor(a, device=dev)
        tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        data = [T(a) for a in (xt, xe, xi)]
        zt = T(z)
        self.states_, self.history_ = [], []
        for s in range(self.settings.nn_seeds):
            torch.manual_seed(SEED + s)
            net = _make_net(*self.dims_).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.settings.nn_lr, weight_decay=self.settings.nn_weight_decay)
            best, best_state, bad, hist = np.inf, None, 0, []
            g = torch.Generator().manual_seed(SEED + s)
            tr_t = torch.as_tensor(tr)
            va_t = torch.as_tensor(va, device=dev)
            for epoch in range(self.settings.nn_epochs):
                net.train()
                perm = tr_t[torch.randperm(len(tr_t), generator=g)].to(dev)
                for i in range(0, len(perm), self.settings.nn_batch):
                    idx = perm[i:i + self.settings.nn_batch]
                    if len(idx) < 2:
                        continue
                    out = net(*[None if d is None else d[idx] for d in data])
                    loss = torch.nn.functional.l1_loss(out, zt[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(torch.nn.functional.l1_loss(net(*[None if d is None else d[va_t] for d in data]), zt[va_t]))
                hist.append(v)
                if v < best - 1e-5:
                    best, bad = v, 0
                    best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.settings.nn_patience:
                        break
            self.states_.append(best_state)
            self.history_.append({"seed": s, "epochs": len(hist), "best_val_l1": best, "val_l1": hist})
        return self

    def predict_log(self, frame):
        import torch
        xt, xe, xi = self._inputs(frame.reset_index(drop=True), fit=False)
        outs = []
        for state in self.states_:
            net = _make_net(*self.dims_)
            net.load_state_dict(state)
            net.eval()
            with torch.no_grad():
                args = [torch.as_tensor(a) if a is not None else None for a in (xt, xe, xi)]
                outs.append(net(*args).numpy())
        return np.mean(outs, 0) * self.sd_ + self.mu_

    def predict(self, frame):
        return np.exp(self.predict_log(frame))


class BlendModel:
    """Fixed-weight geometric blend of two fitted models (weight fixed in advance, not tuned)."""

    def __init__(self, a, b, w: float = 0.5):
        self.a, self.b, self.w = a, b, w
        self.strategy = f"{getattr(a, 'strategy', '?')}+{getattr(b, 'strategy', '?')}"

    def fit(self, frame, y, groups=None):
        self.a.fit(frame, y, groups); self.b.fit(frame, y, groups)
        return self

    def predict(self, frame):
        return np.exp(self.w * np.log(self.a.predict(frame)) + (1 - self.w) * np.log(self.b.predict(frame)))


class NetBlendModel(BlendModel):
    """Strategy Cb: the network (C) and the Endterm base blended 50/50 on log scale (weight fixed in advance)."""

    def __init__(self, blocks: Sequence[Block], settings: Settings | None = None, w: float = 0.5):
        settings = settings or Settings()
        super().__init__(NeuralFusionModel(blocks, settings), BaseModel(settings=settings), w)
        self.blocks, self.settings, self.strategy = [b.light() for b in blocks], settings, "Cb"


class BaseModel:
    """The submitted Endterm model wrapped in the same frame interface."""
    strategy = "base"

    def __init__(self, blocks=(), settings: Settings | None = None):
        self.blocks, self.settings = [], settings or Settings()

    def fit(self, frame, y, groups=None):
        self.model_ = _base_model(self.settings).fit(core.encode_features(frame.reset_index(drop=True)), y)
        return self

    def predict(self, frame):
        return self.model_.predict(core.encode_features(frame.reset_index(drop=True)))


STRATEGIES = {"base": BaseModel, "A": ResidualStackModel, "B": FeatureFusionModel, "C": NeuralFusionModel, "Cb": None}
STRATEGY_LABEL = {"base": "Endterm base", "A": "A residual stacking", "B": "B blocks as HGB features",
                  "C": "C multimodal network", "Cb": "C network + base, 50/50 log-blend"}


def make_model(strategy: str, blocks: Sequence[Block], settings: Settings):
    return STRATEGIES[strategy](blocks, settings)


STRATEGIES["Cb"] = NetBlendModel


# --------------------------------------------------------------------------- CV harness
def result_from_oof(name: str, y: np.ndarray, oof: np.ndarray, folds) -> CVResult:
    return core._result_from_oof(name, y, oof, folds)


def _fit_predict(strategy, blocks, settings, frame, y, groups, tr, va):
    m = make_model(strategy, blocks, settings)
    m.fit(frame.iloc[tr].reset_index(drop=True), y[tr], None if groups is None else groups[tr])
    return m.predict(frame.iloc[va].reset_index(drop=True))


def cv_model(name: str, strategy: str, blocks: Sequence[Block], frame: pd.DataFrame, y: np.ndarray,
             scheme: FoldScheme, settings: Settings, n_jobs: int | None = None,
             blend_with_base: float | None = None) -> CVResult:
    """Generic CV: refit ``strategy`` on every training fold. With ``blend_with_base`` the fold
    prediction is a fixed geometric blend with the base model (weight of the strategy)."""
    n_jobs = settings.n_jobs if n_jobs is None else n_jobs
    if strategy in ("C", "Cb") and torch_device(settings.nn_device) != "cpu":
        n_jobs = 1   # an MPS network stays in the main process
    wanted = set(FEATURES) | {ID, "text_clean"} | {c for b in blocks for c in b.columns}
    cols = [c for c in frame.columns if c in wanted]
    sub = frame[cols]
    preds = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(strategy, blocks, settings, sub, y, scheme.groups, tr, va)
                                    for tr, va in scheme.folds)
    oof = np.empty(len(y))
    for (_, va), p in zip(scheme.folds, preds):
        oof[va] = p
    if blend_with_base is not None:
        base = StackingCV.base_outer_for(scheme, frame, y, settings)
        oof = np.exp(blend_with_base * np.log(oof) + (1 - blend_with_base) * base)
    return result_from_oof(name, y, oof, scheme.folds)


def _base_fold_inputs(X, y, groups, tr, va, settings):
    folds = inner_folds(len(tr), None if groups is None else groups[tr], settings.inner_folds)
    inner = _oof_log(X.iloc[tr].reset_index(drop=True), y[tr], folds, settings)
    outer = np.log(_base_model(settings).fit(X.iloc[tr], y[tr]).predict(X.iloc[va]))
    return inner, outer


class StackingCV:
    """Strategy A under one fold scheme with every expensive piece cached:
    base inner OOF / outer predictions per fold (on disk) and each block's level-1 corrections per fold
    (in memory). Evaluating a block set then only re-blends."""
    _outer_memo: dict = {}

    def __init__(self, frame: pd.DataFrame, y: np.ndarray, scheme: FoldScheme, settings: Settings,
                 cache_dir: Path | None = CACHE_DIR):
        self.frame, self.y, self.scheme, self.settings, self.cache_dir = frame, y, scheme, settings, cache_dir
        self.ids = frame[ID].to_numpy("int64")
        self.level1_: dict[str, list] = {}
        self._base = None

    @property
    def base(self) -> dict:
        if self._base is None:
            self._base = self._compute_base()
        return self._base

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256(json.dumps({"hgb": {**core.HGB_PARAMS, **self.settings.hgb_overrides},
                                       "inner": self.settings.inner_folds}, sort_keys=True).encode())
        h.update(self.scheme.key(self.ids).encode()); h.update(self.y.tobytes())
        return self.cache_dir / f"fusion_base_{self.scheme.name}_{h.hexdigest()[:12]}.npz"

    def _compute_base(self) -> dict:
        p = self._cache_path()
        if p is not None and p.exists():
            z = np.load(p)
            return {"inner": [z[f"inner{k}"] for k in range(len(self.scheme.folds))], "outer": z["outer"]}
        X = core.encode_features(self.frame)
        res = Parallel(n_jobs=self.settings.n_jobs)(delayed(_base_fold_inputs)(X, self.y, self.scheme.groups, tr, va, self.settings)
                                                    for tr, va in self.scheme.folds)
        outer = np.empty(len(self.y))
        for (_, va), (_, o) in zip(self.scheme.folds, res):
            outer[va] = o
        out = {"inner": [r[0] for r in res], "outer": outer}
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.npz")
            np.savez(tmp, outer=outer, **{f"inner{k}": a for k, a in enumerate(out["inner"])})
            os.replace(tmp, p)
        return out

    @classmethod
    def base_outer_for(cls, scheme, frame, y, settings) -> np.ndarray:
        key = (scheme.key(frame[ID].to_numpy("int64")), json.dumps(settings.hgb_overrides, sort_keys=True))
        if key not in cls._outer_memo:
            cls._outer_memo[key] = cls(frame, y, scheme, settings).base["outer"]
        return cls._outer_memo[key]

    def base_result(self, name: str = "Endterm base") -> CVResult:
        return result_from_oof(name, self.y, np.exp(self.base["outer"]), self.scheme.folds)

    def _level1(self, block: Block) -> list:
        if block.name not in self.level1_:
            cols = _block_cols(block) + [ID]
            log_y = np.log(self.y)
            base = self.base
            jobs = []
            for k, (tr, va) in enumerate(self.scheme.folds):
                fr_tr = self.frame.iloc[tr][cols].reset_index(drop=True).assign(base_log_pred=base["inner"][k])
                fr_va = self.frame.iloc[va][cols].reset_index(drop=True).assign(base_log_pred=base["outer"][va])
                folds = inner_folds(len(tr), None if self.scheme.groups is None else self.scheme.groups[tr], self.settings.inner_folds)
                jobs.append(delayed(_level1_fold)(block, fr_tr, log_y[tr] - base["inner"][k], fr_va, folds, self.settings))
            self.level1_[block.name] = Parallel(n_jobs=self.settings.n_jobs)(jobs)
        return self.level1_[block.name]

    def evaluate(self, blocks: Sequence[Block], name: str | None = None) -> tuple[CVResult, np.ndarray]:
        """CV result of base + blended corrections of ``blocks`` and the per-fold weights (folds x blocks)."""
        name = name or f"A: {'+'.join(b.name for b in blocks) or 'base'}"
        log_y = np.log(self.y)
        oof = np.empty(len(self.y))
        W = np.zeros((len(self.scheme.folds), len(blocks)))
        l1 = [self._level1(b) for b in blocks]
        for k, (tr, va) in enumerate(self.scheme.folds):
            delta = np.zeros(len(va))
            if blocks:
                Z_tr = np.column_stack([l[k][0] for l in l1])
                Z_va = np.column_stack([l[k][1] for l in l1])
                W[k] = blend_weights(Z_tr, log_y[tr] - self.base["inner"][k])
                delta = Z_va @ W[k]
            oof[va] = np.exp(self.base["outer"][va] + delta)
        return result_from_oof(name, self.y, oof, self.scheme.folds), W


def _level1_fold(block, fr_tr, resid, fr_va, folds, settings):
    z_tr, z_va, _ = level1(block, fr_tr, resid, fr_va, folds, settings)
    return z_tr, z_va


def _feature_fold(block: Block, frame_tr, y_tr, frame_va, folds, settings):
    t = make_block_transformer(block)
    return t.fit_train(frame_tr, y_tr, folds, settings), t.transform(frame_va)


def _fit_hgb_fold(X_tr, y_tr, X_va, num, cat, settings):
    return _base_model(settings, num, cat).fit(X_tr, y_tr).predict(X_va)


class FeatureCV:
    """Strategy B under one fold scheme; per-fold block features (PCA, text prior) cached in memory."""

    def __init__(self, frame: pd.DataFrame, y: np.ndarray, scheme: FoldScheme, settings: Settings):
        self.frame, self.y, self.scheme, self.settings = frame, y, scheme, settings
        self.X = core.encode_features(frame)
        self.feats_: dict[str, list] = {}

    def _feats(self, block: Block) -> list:
        if block.name not in self.feats_:
            cols = _block_cols(block) + [ID]
            jobs = []
            for tr, va in self.scheme.folds:
                folds = inner_folds(len(tr), None if self.scheme.groups is None else self.scheme.groups[tr], self.settings.inner_folds)
                jobs.append(delayed(_feature_fold)(block, self.frame.iloc[tr][cols].reset_index(drop=True), self.y[tr],
                                                   self.frame.iloc[va][cols].reset_index(drop=True), folds, self.settings))
            n_jobs = 1 if block.kind == "tab" else self.settings.n_jobs
            self.feats_[block.name] = Parallel(n_jobs=n_jobs)(jobs)
        return self.feats_[block.name]

    def evaluate(self, blocks: Sequence[Block], name: str | None = None) -> CVResult:
        name = name or f"B: {'+'.join(b.name for b in blocks) or 'base'}"
        feats = [self._feats(b) for b in blocks]
        jobs = []
        for k, (tr, va) in enumerate(self.scheme.folds):
            X_tr, num, cat = _stack_extra(self.X.iloc[tr].reset_index(drop=True), [f[k][0] for f in feats])
            X_va = _stack_extra_known(self.X.iloc[va].reset_index(drop=True), [f[k][1] for f in feats], num, cat)
            jobs.append(delayed(_fit_hgb_fold)(X_tr, self.y[tr], X_va, num, cat, self.settings))
        preds = Parallel(n_jobs=self.settings.n_jobs)(jobs)
        oof = np.empty(len(self.y))
        for (_, va), p in zip(self.scheme.folds, preds):
            oof[va] = p
        return result_from_oof(name, self.y, oof, self.scheme.folds)


# --------------------------------------------------------------------------- forward selection
def cv_row(res: CVResult, base: CVResult, **extra) -> dict:
    r = res.row(base)
    r["folds_mae_M"] = " / ".join(f"{v:.3f}" for v in res.fold_mae)
    m = rule_margin(base)
    r["passes_1sd_rule"] = bool(base.mean - res.mean > m)
    r["rule_threshold_M"] = round(base.mean - m, 4)
    r.update(extra)
    return r


def forward_select(evaluate: Callable[[tuple[str, ...]], CVResult], candidates: Sequence[str], base: CVResult,
                   log: Callable[[str], None] = print, min_margin: float = 0.0) -> dict:
    """Greedy forward selection.

    At every step each remaining block is added to the current set and the one with the lowest CV MAE
    is taken. The greedy path continues while the CV MAE goes down at all (for the ablation waterfall).
    The team rule decides what is KEPT: a step is accepted only if it lowers the CV MAE of the current
    set by more than one fold SD of the current set (never less than ``min_margin``: the integration run
    passes the fixed Endterm margin RULE_MARGIN_M); the kept set is the path prefix of accepted steps.
    """
    current: tuple[str, ...] = ()
    cur = base
    remaining = list(candidates)
    path, accepted_prefix, stop = [], (), False
    while remaining:
        trials = {b: evaluate(current + (b,)) for b in remaining}
        best = min(trials, key=lambda b: trials[b].mean)
        r = trials[best]
        gain = cur.mean - r.mean
        margin = max(cur.sd, min_margin)
        passes = gain > margin
        log(f"  step {len(path) + 1}: +{best} -> {r.mean:.4f} (gain {gain:+.4f}, step margin {margin:.4f}, "
            f"{'passes' if passes else 'fails'} 1-SD rule)")
        if gain <= 0:
            break
        current = current + (best,)
        path.append({"step": len(path) + 1, "added": best, "blocks": "+".join(current), "cv_mae_M": round(r.mean, 4),
                     "cv_mae_sd_M": round(r.sd, 4), "gain_M": round(gain, 4), "step_sd_M": round(cur.sd, 4), "step_margin_M": round(margin, 4),
                     "folds_better_than_prev": int((r.fold_mae < cur.fold_mae).sum()), "passes_step_rule": bool(passes),
                     "delta_vs_base_M": round(r.mean - base.mean, 4), "set_passes_vs_base": bool(base.mean - r.mean > max(base.sd, min_margin)),
                     "_result": r})
        if passes and not stop:
            accepted_prefix = current
        else:
            stop = True
        cur = r
        remaining.remove(best)
    return {"path": path, "kept": list(accepted_prefix), "greedy": list(current)}


# --------------------------------------------------------------------------- test-time statistics
def bootstrap_metrics(y: np.ndarray, pred: np.ndarray, ref: np.ndarray | None = None, n_boot: int = 2000,
                      seed: int = SEED, alpha: float = 0.05) -> dict:
    """Point metrics + percentile bootstrap CIs (paired resamples of ads). With ``ref`` also the CI of
    the paired difference (pred - ref) of every metric."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    def stats(p):
        ae = np.abs(p - y)
        ape = ae / y
        yb, pb, aeb, apeb = y[idx], p[idx], ae[idx], ape[idx]
        ss_res = ((yb - pb) ** 2).sum(1)
        ss_tot = ((yb - yb.mean(1, keepdims=True)) ** 2).sum(1)
        boot = {"mae_M": aeb.mean(1) / 1e6, "mdape": np.median(apeb, 1) * 100, "r2": 1 - ss_res / ss_tot,
                "within10": (apeb <= 0.10).mean(1), "within20": (apeb <= 0.20).mean(1),
                "rmse_M": np.sqrt(((pb - yb) ** 2).mean(1)) / 1e6}
        return core.metrics(y, p), boot

    point, boot = stats(pred)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {m: {"value": point[m], "ci_low": float(np.percentile(boot[m], lo)), "ci_high": float(np.percentile(boot[m], hi))}
           for m in point}
    if ref is not None:
        rpoint, rboot = stats(np.asarray(ref, float))
        for m in point:
            d = boot[m] - rboot[m]
            out[m]["ref_value"] = rpoint[m]
            out[m]["diff"] = point[m] - rpoint[m]
            out[m]["diff_ci_low"], out[m]["diff_ci_high"] = float(np.percentile(d, lo)), float(np.percentile(d, hi))
    out["n"] = n
    out["n_boot"] = n_boot
    return out


def price_band(y: np.ndarray) -> np.ndarray:
    bins = [0, 25e6, 40e6, 60e6, 100e6, np.inf]
    labels = ["<25M", "25-40M", "40-60M", "60-100M", ">=100M"]
    return pd.cut(np.asarray(y, float), bins, labels=labels, right=False).astype(str)


def segment_table(seg: pd.DataFrame, y: np.ndarray, pred: np.ndarray, ref: np.ndarray | None = None,
                  n_boot: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Error per segment value for every segment column of ``seg`` (row-aligned with y/pred):
    n, MAE with bootstrap CI, MdAPE, median signed log error (bias), and the reference model's MAE."""
    rows = []
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    for col in seg.columns:
        for val, ix in seg.groupby(col, observed=True).indices.items():
            yy, pp = y[ix], pred[ix]
            ae = np.abs(pp - yy) / 1e6
            bi = rng.integers(0, len(ix), size=(n_boot, len(ix)))
            bm = ae[bi].mean(1)
            r = {"segment": col, "value": str(val), "n": len(ix), "mae_M": ae.mean(),
                 "mae_ci_low": np.percentile(bm, 2.5), "mae_ci_high": np.percentile(bm, 97.5),
                 "mdape": core.mdape(yy, pp), "median_log_error": float(np.median(np.log(pp / yy))),
                 "within10": float((np.abs(pp - yy) / yy <= 0.10).mean())}
            if ref is not None:
                rr = np.asarray(ref, float)[ix]
                r["base_mae_M"] = np.abs(rr - yy).mean() / 1e6
                r["base_mdape"] = core.mdape(yy, rr)
                d = (ae - np.abs(rr - yy) / 1e6)[bi].mean(1)
                r["diff_mae_M"] = r["mae_M"] - r["base_mae_M"]
                r["diff_ci_low"], r["diff_ci_high"] = np.percentile(d, 2.5), np.percentile(d, 97.5)
            rows.append(r)
    return pd.DataFrame(rows)


def test_segments(frame: pd.DataFrame, blocks: dict[str, Block], dup_path: Path = DUP_GROUPS_CSV) -> pd.DataFrame:
    """Segment labels of the test rows: city, price band, rooms, photo / text availability, duplicate
    group membership (any duplicate, and duplicate with a twin in the training ads)."""
    seg = pd.DataFrame({"city": frame["city"].astype(str).to_numpy(),
                        "price_band": price_band(frame[core.TARGET].to_numpy()),
                        "rooms": pd.to_numeric(frame["rooms"], errors="coerce").clip(upper=4).fillna(-1).astype(int)
                        .map(lambda r: "4+" if r >= 4 else ("unknown" if r < 0 else str(r))).to_numpy()})
    if "photo_emb" in blocks:
        _, ok = _emb_matrix(frame, blocks["photo_emb"].columns)
        seg["photos"] = np.where(ok, "with photo embedding", "no photos")
    else:
        seg["photos"] = np.where(pd.to_numeric(frame["n_photos"], errors="coerce").fillna(0) > 0, "n_photos>0", "n_photos=0")
    tl = frame["text_clean"].str.len() if "text_clean" in frame else frame["description"].fillna("").str.len()
    seg["text"] = np.where(tl >= 100, "description >=100 chars", "short/no description")
    if dup_path.exists():
        d = pd.read_csv(dup_path)
        d[ID] = d[ID].astype("int64")
        d = d.drop_duplicates(ID).set_index(ID)
        gcol = "group" if "group" in d.columns else ("dup_group" if "dup_group" in d.columns else None)
        if gcol and "group_size" in d.columns:
            g = d.reindex(frame[ID].to_numpy("int64"))
            in_group = g["group_size"].fillna(1).to_numpy() > 1
            seg["duplicate"] = np.where(in_group, "in duplicate group", "unique ad")
            if "split" in d.columns:
                train_groups = set(d.loc[d["split"] == "train", gcol])
                twin = in_group & g[gcol].isin(train_groups).to_numpy()
                seg["train_twin"] = np.where(twin, "has a twin in train", "no train twin")
    return seg


# --------------------------------------------------------------------------- November hold-out
def reduced_features(nov: pd.DataFrame, min_coverage: float = 0.5) -> tuple[list[str], list[str], pd.DataFrame]:
    """Base features the November cards actually carry: numeric/bool filled for >= ``min_coverage`` of
    the rows, categoricals not 'unknown' for >= ``min_coverage``. Returns (num+bool, cat, coverage table)."""
    X = core.encode_features(nov)
    rows = []
    for c in FEATURES:
        if c in CAT:
            cov = float((X[c] != "unknown").mean())
        elif c in BOOL:
            cov = float(nov[c].notna().mean()) if c in nov else 0.0
        else:
            cov = float(X[c].notna().mean())
        rows.append({"feature": c, "kind": "cat" if c in CAT else ("bool" if c in BOOL else "num"), "nov_coverage": cov,
                     "kept": cov >= min_coverage})
    cov = pd.DataFrame(rows)
    keep = cov[cov["kept"]]
    return list(keep.loc[keep["kind"] != "cat", "feature"]), list(keep.loc[keep["kind"] == "cat", "feature"]), cov


def make_reduced_model(num: Sequence[str], cat: Sequence[str], settings: Settings | None = None) -> TransformedTargetRegressor:
    """The Endterm HGB (same params, same preprocessing style) on a reduced column set."""
    settings = settings or Settings()
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))]), list(num)),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), list(cat))])
    hgb = HistGradientBoostingRegressor(**{**core.HGB_PARAMS, **settings.hgb_overrides}, random_state=SEED, early_stopping=False)
    return TransformedTargetRegressor(regressor=Pipeline([("pre", pre), ("model", hgb)]), func=np.log, inverse_func=np.exp,
                                      check_inverse=False)


def load_nov(path: Path = NOV_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df[core.TARGET], errors="coerce") > 0].reset_index(drop=True)
    df, _ = core.fix_coordinates(df)
    return df


def drift_table(y_oct: np.ndarray, df_oct: pd.DataFrame, y_nov: np.ndarray, df_nov: pd.DataFrame) -> pd.DataFrame:
    """Median price and price per m2 by city, October (all clean ads) vs November."""
    rows = []
    for city in sorted(set(df_oct["city"]) | set(df_nov["city"])):
        o, n = (df_oct["city"] == city).to_numpy(), (df_nov["city"] == city).to_numpy()
        if not o.any() or not n.any():
            continue
        po = np.median(y_oct[o] / df_oct.loc[o, "area"].to_numpy(float))
        pn = np.median(y_nov[n] / df_nov.loc[n, "area"].to_numpy(float))
        rows.append({"city": city, "n_oct": int(o.sum()), "n_nov": int(n.sum()),
                     "median_price_oct_M": np.median(y_oct[o]) / 1e6, "median_price_nov_M": np.median(y_nov[n]) / 1e6,
                     "median_price_m2_oct_k": po / 1e3, "median_price_m2_nov_k": pn / 1e3,
                     "price_m2_change_pct": (pn / po - 1) * 100,
                     "median_area_oct": float(df_oct.loc[o, "area"].median()), "median_area_nov": float(df_nov.loc[n, "area"].median())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- persistence
def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=json_default))
    os.replace(tmp, path)


def model_metadata(model, blocks: Sequence[Block], settings: Settings, extra: dict | None = None) -> dict:
    import sklearn
    feats = list(FEATURES) + [c for b in blocks for c in b.columns]
    meta = {"strategy": getattr(model, "strategy", "?"), "strategy_label": STRATEGY_LABEL.get(getattr(model, "strategy", ""), ""),
            "blocks": [b.summary() for b in blocks], "base_features": list(FEATURES),
            "input_columns": feats + (["description"] if any(b.kind == "text" for b in blocks) else []),
            "n_input_columns": len(feats), "settings": asdict(settings), "hgb_params": {**core.HGB_PARAMS, **settings.hgb_overrides},
            "seed": SEED, "sklearn": sklearn.__version__, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "usage": "frame = fusion.attach_blocks(clean_rows, fusion.load_blocks(<block names>)[0]); model.predict(frame) -> KZT"}
    if getattr(model, "weights_", None) is not None:
        meta["blend_weights"] = {b.name: float(w) for b, w in zip(blocks, model.weights_)}
    if getattr(model, "feature_names_", None) is not None:
        meta["model_feature_names"] = model.feature_names_
    meta.update(extra or {})
    return meta
