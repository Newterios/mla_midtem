"""Tests of krisha_ml.trust (fast: small fits on a subset of the training ads)."""
import numpy as np
import pandas as pd
import pytest

from krisha_ml import core, trust


@pytest.fixture(scope="module")
def small():
    t = core.load_train()
    idx = np.random.default_rng(0).choice(len(t), 1200, replace=False)
    return t.X.iloc[idx].reset_index(drop=True), t.y[idx], t.df.iloc[idx].reset_index(drop=True)


def test_conformal_quantile_finite_sample():
    s = np.arange(1, 11, dtype=float)                    # n=10, alpha=0.1 -> ceil(11*0.9)=10th smallest
    assert trust.conformal_quantile(s, 0.1) == 10.0
    assert trust.conformal_quantile(s, 0.2) == 9.0       # ceil(11*0.8)=9
    assert trust.conformal_quantile(s[:5], 0.1) == np.inf


def test_calibrate_mondrian_and_fallback():
    s = np.r_[np.zeros(50), np.ones(50)]
    g = np.array(["a"] * 50 + ["b"] * 50)
    q = trust.calibrate(s, 0.1, g)
    assert q["a"] == 0.0 and q["b"] == 1.0 and q[trust.GLOBAL] == 1.0
    assert np.array_equal(trust._lookup(q, np.array(["a", "c"]), 2), [0.0, 1.0])   # unknown group -> global


def test_split_cqr_matches_mapie(small):
    from mapie.regression import ConformalizedQuantileRegressor
    X, y, _ = small
    ci = trust.ConformalIntervals(method="cqr", levels=(0.9,), mondrian=None, calibration="split").fit(X, y)
    lo, hi = ci.predict_log(X.iloc[:200], 0.9)
    fit_idx, cal_idx = trust.train_test_split(np.arange(len(X)), test_size=0.25, random_state=core.SEED)
    est = [ci.models_[0.05], ci.models_[0.95], ci.models_[0.95]]    # median slot unused for the bounds
    mapie = ConformalizedQuantileRegressor(est, confidence_level=0.9, prefit=True)
    mapie.conformalize(X.iloc[cal_idx], np.log(y[cal_idx]))
    _, pis = mapie.predict_interval(X.iloc[:200], symmetric_correction=True)
    assert np.allclose(lo, pis[:, 0, 0]) and np.allclose(hi, pis[:, 1, 0])


def test_resid_intervals_and_flags(small):
    X, y, _ = small
    model = core.make_base_model().fit(X, y)
    oof = core.oof_log_pred(X, y)
    ci = trust.ConformalIntervals(method="resid", levels=(0.8, 0.9), mondrian="city").fit(X, y, model, oof)
    lo8, hi8 = ci.predict(X, 0.8)
    lo9, hi9 = ci.predict(X, 0.9)
    assert np.all(lo9 <= lo8) and np.all(hi8 <= hi9)
    assert set(ci.qhat_[0.9]) == {trust.GLOBAL, "Astana", "Almaty"}
    flags = trust.price_flags(np.array([1, 5, 10.0]), np.array([2, 2, 2.0]), np.array([8, 8, 8.0]))
    assert list(flags) == ["under", "in range", "over"]
    assert list(trust.endterm_flags(np.array([130, 100, 70.0]), np.full(3, 100.0))) == ["over", "in range", "under"]


def test_shap_additive_and_grouped(small):
    X, y, _ = small
    model = core.make_base_model().fit(X, y)
    res = trust.shap_explain(model, X.iloc[:100])
    assert "location" in res.values.columns and "lat" not in res.values.columns and "district" not in res.values.columns
    assert res.values.shape[1] == len(core.FEATURES) - 3   # lat/lon/district -> 1, area/log_area -> 1
    assert np.allclose(res.base_value + res.values.sum(axis=1), np.log(model.predict(X.iloc[:100])))
    one = res.explain_one(0)
    assert one["shap_log"].abs().is_monotonic_decreasing


def test_comparables_same_city_and_not_self(small):
    _, _, df = small
    comp = trust.Comparables(k=5).fit(trust.comparables_frame(df))
    q = df.iloc[:20]
    res = comp.query(q)
    assert len(res) == 20 * 5
    assert list(res["query_id"].unique()) == list(q["id"])                     # query order kept
    assert not (res["query_id"] == res["comp_id"]).any()
    city = dict(zip(df["id"], df["city"]))
    assert (res["query_id"].map(city) == res["comp_id"].map(city)).all()
    assert (res.groupby("query_id")["distance"].apply(lambda d: d.is_monotonic_increasing)).all()
    # the warning flags about (1 - warn_quantile) of the training ads
    allq = comp.summary(comp.query(df))
    assert 0.02 < allq["few_comparables"].mean() < 0.09


def test_resid_estimate_band_mondrian(small):
    X, y, _ = small
    model = core.make_base_model().fit(X, y)
    oof = core.oof_log_pred(X, y)
    ci = trust.ConformalIntervals("resid", (0.9,), trust.ESTIMATE_BAND).fit(X, y, model, oof)
    assert trust.GLOBAL in ci.qhat_[0.9] and len(ci.qhat_[0.9]) >= 4
    lo, hi = ci.predict_log(X, 0.9)
    c = np.log(model.predict(X))
    band = np.asarray(trust.price_band(np.exp(c)).astype(str))
    q = np.array([ci.qhat_[0.9].get(b, ci.qhat_[0.9][trust.GLOBAL]) for b in band])
    assert np.allclose(hi - c, q) and np.allclose(c - lo, q)
