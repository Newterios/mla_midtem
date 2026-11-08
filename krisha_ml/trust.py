"""Trust layer on top of a price model: prediction intervals, over-pricing flags, SHAP explanations
and comparable ads (lectures 4 and 5).

Every public function takes a *fitted model and feature frames*, so the integration stage can swap
the Endterm base for the final model without touching this module:

* intervals   ``ConformalIntervals(method, levels, mondrian).fit(X, y, ...)`` -> ``.predict(X)``
               conformalized quantile regression (CQR, Romano et al. 2019) or absolute-residual
               conformal around any point model, on log(price), optional Mondrian groups (per city).
* flags       ``price_flags(price, lo, hi)`` -> "over" / "under" / "in range" (+ the Endterm +-20 % rule)
* SHAP        ``shap_explain(model, X)`` -> exact TreeSHAP of a fitted ``Pipeline(pre, HGB)`` (optionally
               wrapped in ``TransformedTargetRegressor``), one-hot / missing-indicator columns summed
               back to source features, lat/lon/district grouped into ``location``.
* comparables ``Comparables().fit(df_train).query(df_query)`` -> 5 most similar training ads per query
               plus a ``few_comparables`` warning.

Conformal quantile used everywhere: the ``ceil((n+1)(1-alpha))``-th smallest calibration score (the
finite-sample-valid split-conformal quantile; MAPIE 1.x computes the same, checked in tests).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

from . import core

LEVELS = (0.8, 0.9)
GLOBAL = "__all__"
ESTIMATE_BAND = "estimate_band"     # Mondrian groups = price band of the point estimate

# Quantile HGB settings compared in scripts/run_trust.py (train-only cross-fitting).
QUANTILE_PARAMS = {
    "endterm": dict(core.HGB_PARAMS),
    "smooth": {"min_samples_leaf": 20, "max_leaf_nodes": 15, "max_iter": 300, "max_depth": None,
               "learning_rate": 0.05, "l2_regularization": 1.0},
}

# Source features that are explained together (SHAP values are additive, so a group = sum of members).
SHAP_GROUPS = {"location": ("lat", "lon", "district"), "area": ("area", "log_area")}


# =========================================================================== conformal core
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score.
    Returns +inf when the calibration set is too small for that level."""
    scores = np.asarray(scores, float)
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return float("inf")
    return float(np.partition(scores, k - 1)[k - 1])


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CQR conformity score (symmetric correction): how far y falls outside [lo, hi] (negative inside)."""
    return np.maximum(lo - y, y - hi)


def calibrate(scores: np.ndarray, alpha: float, groups: np.ndarray | None = None,
              min_group: int | None = None) -> dict[str, float]:
    """Conformal correction per Mondrian group plus the global one (key ``GLOBAL``).
    Groups smaller than ``min_group`` (default: needed for a finite quantile, ceil(1/alpha)) are left
    out and fall back to the global value at prediction time."""
    out = {GLOBAL: conformal_quantile(scores, alpha)}
    if groups is not None:
        min_group = min_group or math.ceil(1 / alpha)
        for g in pd.unique(groups):
            s = scores[groups == g]
            if len(s) >= min_group:
                out[str(g)] = conformal_quantile(s, alpha)
    return out


def _lookup(qhat: dict[str, float], groups: np.ndarray | None, n: int) -> np.ndarray:
    if groups is None:
        return np.full(n, qhat[GLOBAL])
    return np.array([qhat.get(str(g), qhat[GLOBAL]) for g in groups])


# =========================================================================== models
def make_quantile_model(quantile: float, params: dict | None = None) -> Pipeline:
    """HGB with pinball loss on log(price), Endterm preprocessing. Fit with ``y = log(price)``."""
    params = dict(params or QUANTILE_PARAMS["smooth"])
    hgb = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, random_state=core.SEED,
                                        early_stopping=False, **params)
    return Pipeline([("pre", core.make_preprocessor()), ("model", hgb)])


def _quantiles_for(levels: Sequence[float]) -> list[float]:
    qs = sorted({round((1 - lv) / 2, 6) for lv in levels} | {round(1 - (1 - lv) / 2, 6) for lv in levels})
    return qs


def predict_log(model, X: pd.DataFrame) -> np.ndarray:
    """Log-price prediction of a fitted model: a TransformedTargetRegressor with func=log predicts KZT."""
    if isinstance(model, TransformedTargetRegressor):
        return np.log(model.predict(X))
    return np.asarray(model.predict(X), float)


# =========================================================================== intervals
@dataclass
class ConformalIntervals:
    """Conformal prediction intervals for the asking price (computed on log price, returned in KZT).

    method       "cqr"   conformalized quantile regression: HGB quantile models at (1-level)/2 and
                         1-(1-level)/2, corrected by the conformal quantile of CQR scores.
                 "resid" absolute-residual conformal around a point model: log(pred) +- q.
                         Needs the fitted point model and its honest OOF log predictions on (X, y).
    calibration  "crossfit" scores from inner K-fold OOF predictions on all (X, y), final models fitted
                            on all rows (CV+-style; uses every row for both, approximate guarantee).
                 "split"    fit on a random ``1 - cal_size`` part, calibrate on the rest (exact
                            finite-sample marginal guarantee, fewer rows for each). CQR only.
    mondrian     column of X used as Mondrian groups (e.g. "city": separate correction per city), None, or
                 ``ESTIMATE_BAND`` ("resid" only): groups = price band of the point estimate (known at
                 prediction time; OOF estimate for the calibration rows), so expensive flats get their
                 own, wider correction.
    """
    method: str = "cqr"
    levels: tuple[float, ...] = LEVELS
    mondrian: str | None = "city"
    calibration: str = "crossfit"
    quantile_params: dict | None = None
    cal_size: float = 0.25
    n_folds: int = core.N_FOLDS
    seed: int = core.SEED
    # fitted state
    scores_: dict[float, np.ndarray] = field(default_factory=dict, repr=False)
    groups_: np.ndarray | None = field(default=None, repr=False)
    qhat_: dict[float, dict[str, float]] = field(default_factory=dict)
    models_: dict[float, object] = field(default_factory=dict, repr=False)   # quantile -> fitted model
    point_model_: object = field(default=None, repr=False)

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: np.ndarray, point_model=None, point_oof_log: np.ndarray | None = None,
            oof_quantiles: dict[float, np.ndarray] | None = None) -> "ConformalIntervals":
        """Fit on training ads. ``y`` in KZT.

        resid: ``point_model`` fitted on (X, y) and ``point_oof_log`` = honest OOF log predictions on
               (X, y) (``core.oof_log_pred``) are required.
        cqr:   ``oof_quantiles`` {q: OOF log predictions on X} may be passed to skip the inner refits
               (crossfit only; they must come from ``crossfit_quantiles`` with the same settings).
        """
        X = X.reset_index(drop=True)
        log_y = np.log(np.asarray(y, float))
        if self.method == "resid":
            if point_model is None or point_oof_log is None:
                raise ValueError("method='resid' needs point_model and point_oof_log")
            if self.calibration != "crossfit":
                raise ValueError("method='resid' supports calibration='crossfit' only")
            self.point_model_ = point_model
            s = np.abs(log_y - np.asarray(point_oof_log, float))
            self.scores_ = {lv: s for lv in self.levels}
            self._oof_log_ = np.asarray(point_oof_log, float)
            self._set_groups(X)
        elif self.method == "cqr":
            qs = _quantiles_for(self.levels)
            if self.calibration == "crossfit":
                oof = oof_quantiles or crossfit_quantiles(X, log_y, qs, self.quantile_params, self.n_folds, self.seed)
                self.models_ = {q: make_quantile_model(q, self.quantile_params).fit(X, log_y) for q in qs}
                cal_idx = np.arange(len(X))
            elif self.calibration == "split":
                fit_idx, cal_idx = train_test_split(np.arange(len(X)), test_size=self.cal_size, random_state=self.seed)
                self.models_ = {q: make_quantile_model(q, self.quantile_params).fit(X.iloc[fit_idx], log_y[fit_idx]) for q in qs}
                oof = {q: m.predict(X.iloc[cal_idx]) for q, m in self.models_.items()}
            else:
                raise ValueError(f"unknown calibration {self.calibration!r}")
            self.scores_ = {}
            for lv in self.levels:
                lo, hi = self._pair(lv)
                self.scores_[lv] = cqr_scores(oof[lo], oof[hi], log_y[cal_idx])
            self._set_groups(X.iloc[cal_idx])
        else:
            raise ValueError(f"unknown method {self.method!r}")
        self._calibrate()
        return self

    def _pair(self, level: float) -> tuple[float, float]:
        a = round((1 - level) / 2, 6)
        return a, round(1 - a, 6)

    def _group_labels(self, X: pd.DataFrame, center_log: np.ndarray | None) -> np.ndarray | None:
        if not self.mondrian:
            return None
        if self.mondrian == ESTIMATE_BAND:
            if self.method != "resid":
                raise ValueError("mondrian=ESTIMATE_BAND is implemented for method='resid'")
            return np.asarray(price_band(np.exp(center_log)).astype(str))
        return X[self.mondrian].astype(str).to_numpy()

    def _set_groups(self, X_cal: pd.DataFrame) -> None:
        self.groups_ = self._group_labels(X_cal.reset_index(drop=True), getattr(self, "_oof_log_", None))

    def _calibrate(self) -> None:
        self.qhat_ = {lv: calibrate(self.scores_[lv], 1 - lv, self.groups_ if self.mondrian else None) for lv in self.levels}

    def with_mondrian(self, mondrian: str | None, X_cal: pd.DataFrame | None = None) -> "ConformalIntervals":
        """Copy that shares the fitted models but recalibrates with other Mondrian groups (cheap).
        ``X_cal`` (the calibration rows, i.e. the X passed to ``fit`` for crossfit) is needed when
        switching on a grouping the fitted object does not hold."""
        other = copy.copy(self)
        other.mondrian = mondrian
        if mondrian is None:
            other.groups_ = None
        elif X_cal is not None or mondrian == ESTIMATE_BAND:
            other._set_groups(X_cal if X_cal is not None else pd.DataFrame(index=range(len(self.scores_[self.levels[0]]))))
        elif mondrian != self.mondrian:
            raise ValueError("pass X_cal to switch to a new Mondrian column")
        other._calibrate()
        return other

    # ------------------------------------------------------------------ predict
    def predict_log(self, X: pd.DataFrame, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) on the log-price scale."""
        X = X.reset_index(drop=True)
        c = predict_log(self.point_model_, X) if self.method == "resid" else None
        q = _lookup(self.qhat_[level], self._group_labels(X, c), len(X))
        if self.method == "resid":
            return c - q, c + q
        lo_q, hi_q = self._pair(level)
        return self.models_[lo_q].predict(X) - q, self.models_[hi_q].predict(X) + q

    def predict(self, X: pd.DataFrame, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) asking-price bounds in KZT."""
        lo, hi = self.predict_log(X, level)
        return np.exp(lo), np.exp(hi)

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """All levels as columns lo80, hi80, lo90, hi90 (KZT)."""
        cols = {}
        for lv in self.levels:
            lo, hi = self.predict(X, lv)
            tag = int(round(lv * 100))
            cols[f"lo{tag}"], cols[f"hi{tag}"] = lo, hi
        return pd.DataFrame(cols)

    def settings(self) -> dict:
        return {"method": self.method, "calibration": self.calibration, "mondrian": self.mondrian,
                "levels": list(self.levels), "quantile_params": self.quantile_params,
                "qhat_log": {str(lv): {g: round(v, 6) for g, v in d.items()} for lv, d in self.qhat_.items()}}


def crossfit_quantiles(X: pd.DataFrame, log_y: np.ndarray, quantiles: Sequence[float], params: dict | None = None,
                       n_folds: int = core.N_FOLDS, seed: int = core.SEED) -> dict[float, np.ndarray]:
    """Honest K-fold OOF log predictions of HGB quantile models (KFold(shuffle, seed), as core's inner folds)."""
    X = X.reset_index(drop=True)
    out = {q: np.empty(len(X)) for q in quantiles}
    for tr, va in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        for q in quantiles:
            out[q][va] = make_quantile_model(q, params).fit(X.iloc[tr], log_y[tr]).predict(X.iloc[va])
    return out


# =========================================================================== evaluation helpers
PRICE_BANDS_M = (0, 25, 40, 60, 100, np.inf)


def price_band(price_kzt: np.ndarray) -> pd.Categorical:
    labels = ["<25 M", "25-40 M", "40-60 M", "60-100 M", ">=100 M"]
    return pd.cut(np.asarray(price_kzt) / 1e6, PRICE_BANDS_M, right=False, labels=labels)


def coverage_table(price: np.ndarray, lo: np.ndarray, hi: np.ndarray, groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Coverage and width (KZT bounds) overall and per group. ``groups``: {group_type: labels per ad}."""
    price, lo, hi = (np.asarray(a, float) for a in (price, lo, hi))
    frame = pd.DataFrame({"covered": (price >= lo) & (price <= hi), "width_M": (hi - lo) / 1e6,
                          "rel_width": (hi - lo) / price, "above": price > hi, "below": price < lo})
    rows = [("overall", "all", frame)]
    for gtype, labels in groups.items():
        lab = pd.Series(np.asarray(labels), index=frame.index).astype(str)
        rows += [(gtype, g, frame[lab == g]) for g in sorted(lab.unique())]
    return pd.DataFrame([{"group_type": t, "group": g, "n": len(f), "coverage": f["covered"].mean(),
                          "share_above": f["above"].mean(), "share_below": f["below"].mean(),
                          "mean_width_M": f["width_M"].mean(), "median_width_M": f["width_M"].median(),
                          "median_rel_width_pct": 100 * f["rel_width"].median()} for t, g, f in rows])


def coverage_ok(coverage: float, n: int, level: float, z: float = 2.0) -> bool:
    """Coverage is accepted if it is not significantly below nominal: >= level - z * binomial SE."""
    return coverage >= level - z * math.sqrt(level * (1 - level) / n)


# =========================================================================== over-pricing flags
def price_flags(price: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """'over' if the asking price is above the upper bound, 'under' if below the lower, else 'in range'."""
    price = np.asarray(price, float)
    return np.where(price > hi, "over", np.where(price < lo, "under", "in range"))


def endterm_flags(price: np.ndarray, pred: np.ndarray, tol: float = 0.20) -> np.ndarray:
    """The Endterm rule: over if price > (1 + tol) * estimate, under if price < (1 - tol) * estimate."""
    price, pred = np.asarray(price, float), np.asarray(pred, float)
    return np.where(price > (1 + tol) * pred, "over", np.where(price < (1 - tol) * pred, "under", "in range"))


# =========================================================================== SHAP
def _split_model(model) -> tuple[Pipeline, object, object]:
    reg = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    if not isinstance(reg, Pipeline) or len(reg.steps) != 2:
        raise TypeError("shap_explain expects Pipeline([('pre', ColumnTransformer), ('model', tree model)]), "
                        "optionally inside a TransformedTargetRegressor")
    return reg, reg.steps[0][1], reg.steps[1][1]


def source_feature_map(pre) -> list[str]:
    """For every output column of a fitted ColumnTransformer, the input column it comes from
    (one-hot dummies, '_infrequent_sklearn' and missing indicators map back to their source)."""
    out_names = pre.get_feature_names_out()
    sources = []
    by_tr = {name: [str(c) for c in cols] for name, _, cols in pre.transformers_ if name != "remainder"}
    for full in out_names:
        tr, rest = full.split("__", 1)
        if rest.startswith("missingindicator_"):
            rest = rest[len("missingindicator_"):]
        cands = [c for c in by_tr[tr] if rest == c or rest.startswith(c + "_")]
        if not cands:
            raise ValueError(f"cannot map output column {full!r} to a source feature")
        sources.append(max(cands, key=len))
    return sources


@dataclass
class ShapResult:
    """Log-scale SHAP values per (grouped) source feature. ``base + values.sum(1) == log_pred``."""
    values: pd.DataFrame          # n x groups
    base_value: float
    log_pred: np.ndarray
    data: pd.DataFrame            # the source rows (for display)

    def explain_one(self, i: int) -> pd.DataFrame:
        """One ad: contribution per group (log), multiplicative effect in % and the feature values."""
        v = self.values.iloc[i]
        tab = pd.DataFrame({"group": v.index, "shap_log": v.to_numpy(), "effect_pct": 100 * np.expm1(v.to_numpy())})
        tab["value"] = [_group_value(self.data.iloc[i], g) for g in tab["group"]]
        return tab.reindex(tab["shap_log"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _group_value(row: pd.Series, group: str) -> str:
    members = SHAP_GROUPS.get(group, (group,))
    vals = [row[m] for m in members if m in row.index]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    return ", ".join(fmt(v) for v in vals)


def shap_explain(model, X: pd.DataFrame, groups: dict[str, Sequence[str]] | None = SHAP_GROUPS) -> ShapResult:
    """Exact TreeSHAP of a fitted tree pipeline on the log-price scale.

    The TreeExplainer runs on the very model that predicts (no surrogate): SHAP values of the one-hot
    dummies and missing indicators of a feature are summed back to that feature (additivity makes the
    sum the feature's contribution), then ``groups`` are summed the same way."""
    import shap   # heavy import, only when needed
    _, pre, tree = _split_model(model)
    Xt = pre.transform(X)
    explainer = shap.TreeExplainer(tree)
    raw_vals = explainer.shap_values(Xt)
    base = float(np.ravel(explainer.expected_value)[0])
    src = source_feature_map(pre)
    per_feat = pd.DataFrame(raw_vals, columns=src).T.groupby(level=0, sort=False).sum().T
    if groups:
        gmap = {m: g for g, members in groups.items() for m in members}
        per_feat = per_feat.T.groupby(per_feat.columns.map(lambda c: gmap.get(c, c)), sort=False).sum().T
    log_pred = np.asarray(tree.predict(Xt), float)
    gap = np.abs(base + per_feat.to_numpy().sum(1) - log_pred).max()
    if gap > 1e-6:
        raise AssertionError(f"SHAP additivity broken: max gap {gap:.2e}")
    return ShapResult(per_feat.reset_index(drop=True), base, log_pred, X.reset_index(drop=True))


def make_ordinal_model(**hgb_overrides) -> TransformedTargetRegressor:
    """Endterm HGB with ordinal-encoded categoricals (one column per feature). Used as a check that the
    one-hot SHAP sums tell the same story as a model whose categoricals are single columns."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), core.NUM + core.BOOL),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=10), core.CAT),
    ])
    hgb = HistGradientBoostingRegressor(**{**core.HGB_PARAMS, **hgb_overrides}, random_state=core.SEED, early_stopping=False)
    return TransformedTargetRegressor(regressor=Pipeline([("pre", pre), ("model", hgb)]), func=np.log,
                                      inverse_func=np.exp, check_inverse=False)


# =========================================================================== comparables
COMP_FEATURES = ("log_area", "rooms", "year_built", "floor", "y_km", "x_km")


@dataclass
class Comparables:
    """K most similar training ads in the same city (standardised log area, rooms, year, floor and
    coordinates in km, scaler fitted per city). A query gets ``few_comparables=True`` when its k-th
    neighbour is farther than the ``warn_quantile`` of the leave-one-out k-th-neighbour distance of the
    training ads of that city."""
    k: int = 5
    warn_quantile: float = 0.95
    coord_weight: float = 1.0
    city_col: str = "city"
    # fitted state
    threshold_: dict[str, float] = field(default_factory=dict)
    _fit: dict = field(default_factory=dict, repr=False)

    def _raw(self, df: pd.DataFrame, city: str) -> np.ndarray:
        st = self._fit[city]
        lat = df["lat"].fillna(st["lat0"]).to_numpy(float)
        lon = df["lon"].fillna(st["lon0"]).to_numpy(float)
        return np.column_stack([
            np.log(df["area"].to_numpy(float)),
            df["rooms"].to_numpy(float),
            df["year_built"].fillna(st["med"]["year_built"]).to_numpy(float),
            df["floor"].fillna(st["med"]["floor"]).to_numpy(float),
            (lat - st["lat0"]) * 111.2,                                        # km north of the city median
            (lon - st["lon0"]) * 111.2 * np.cos(np.radians(st["lat0"])),      # km east
        ])

    def _matrix(self, df: pd.DataFrame, city: str) -> np.ndarray:
        st = self._fit[city]
        Z = (self._raw(df, city) - st["mu"]) / st["sd"]
        Z[:, -2:] *= self.coord_weight
        return Z

    def fit(self, df: pd.DataFrame) -> "Comparables":
        """df: training ads (clean columns id, price, city, area, rooms, year_built, floor, lat, lon)."""
        from sklearn.neighbors import NearestNeighbors
        df = df.reset_index(drop=True)
        self._fit, self.threshold_ = {}, {}
        for city, part in df.groupby(self.city_col, sort=True):
            part = part.reset_index(drop=True)
            st = {"lat0": float(part["lat"].median()), "lon0": float(part["lon"].median()),
                  "med": {c: float(part[c].median()) for c in ("year_built", "floor")}}
            self._fit[city] = st
            raw_m = self._raw(part, city)
            st["mu"], st["sd"] = raw_m.mean(0), raw_m.std(0) + 1e-12
            Z = self._matrix(part, city)
            nn = NearestNeighbors(n_neighbors=min(self.k + 1, len(part))).fit(Z)
            d, _ = nn.kneighbors(Z)                    # column 0 is the ad itself (distance 0)
            self.threshold_[city] = float(np.quantile(d[:, self.k], self.warn_quantile))
            st.update(nn=nn, rows=part[["id", "price", "area", "rooms", "year_built", "floor", "district", "lat", "lon"]].copy())
        return self

    def query(self, df: pd.DataFrame, exclude_self: bool = True) -> pd.DataFrame:
        """Long table: one row per (query ad, neighbour rank 1..k) with the neighbour's id, price, price
        per m2, features, distance, the query's 5th-neighbour distance and the ``few_comparables`` flag.
        ``exclude_self`` drops a neighbour with the query's own id (when querying training ads)."""
        df = df.reset_index(drop=True)
        out = []
        for city, part in df.groupby(self.city_col, sort=False):
            if city not in self._fit:
                raise KeyError(f"no training ads for city {city!r}")
            st, pos = self._fit[city], part.index.to_numpy()
            extra = 1 if exclude_self else 0
            d, j = st["nn"].kneighbors(self._matrix(part, city), n_neighbors=min(self.k + extra, len(st["rows"])))
            ids = st["rows"]["id"].to_numpy()
            for r, p in enumerate(pos):
                keep = [c for c in range(d.shape[1]) if not (exclude_self and ids[j[r, c]] == df.at[p, "id"])][: self.k]
                nb = st["rows"].iloc[j[r, keep]].reset_index(drop=True)
                nb.insert(0, "rank", np.arange(1, len(keep) + 1))
                nb.insert(0, "query_id", df.at[p, "id"])
                nb.insert(0, "_pos", p)
                nb["distance"] = d[r, keep]
                nb["dist_k"] = d[r, keep][-1]
                nb["few_comparables"] = bool(d[r, keep][-1] > self.threshold_[city])
                out.append(nb.rename(columns={"id": "comp_id", "price": "comp_price"}))
        res = pd.concat(out, ignore_index=True)
        res["comp_price_m2"] = res["comp_price"] / res["area"]
        return res.sort_values(["_pos", "rank"], kind="stable").drop(columns="_pos").reset_index(drop=True)

    def summary(self, comps: pd.DataFrame, query_area: pd.Series | None = None) -> pd.DataFrame:
        """Per query ad: k-th distance, warning, median neighbour price and price per m2, and (if the
        query areas are given, indexed by query id) a comparables estimate = median price/m2 x area."""
        g = comps.groupby("query_id", sort=False)
        s = pd.DataFrame({"dist_k": g["dist_k"].first(), "few_comparables": g["few_comparables"].first(),
                          "comp_median_price": g["comp_price"].median(),
                          "comp_median_price_m2": g["comp_price_m2"].median()})
        if query_area is not None:
            s["comp_estimate"] = s["comp_median_price_m2"] * query_area.reindex(s.index).to_numpy()
        return s.reset_index()


def comparables_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Columns Comparables needs, from a clean-ads frame (Dataset.df)."""
    cols = ["id", "price", "city", "area", "rooms", "year_built", "floor", "district", "lat", "lon"]
    return df[[c for c in cols if c in df.columns]].copy()
