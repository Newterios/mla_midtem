"""Fast checks of the shared foundation (about 15 s). Run: python -m pytest -q tests/test_core.py"""
import numpy as np
import pandas as pd
import pytest

from krisha_ml import core, raw


@pytest.fixture(scope="module")
def train():
    return core.load_train()


def test_split_and_features(train):
    assert len(train) == 3852 and len(core.load_test_ids()) == 963
    assert not set(train.ids) & core.load_test_ids()
    assert list(train.X.columns) == core.FEATURES and len(core.FEATURES) == 24
    assert {"price", "price_m2", "has_detail"}.isdisjoint(core.FEATURES)
    assert (train.X["city"] != "").all() and train.X[core.BOOL].isin([0, 1]).all().all()


def test_coordinate_fix_sets_ten_ads_missing():
    _, fixed = core.fix_coordinates(pd.read_csv(core.CLEAN_CSV))
    assert int(fixed.sum()) == 10
    clean = core.load_clean()
    for city, (la0, la1, lo0, lo1) in core.CITY_BOX.items():
        c = clean[(clean["city"] == city) & clean["lat"].notna()]
        assert c["lat"].between(la0, la1).all() and c["lon"].between(lo0, lo1).all()


def test_folds_are_the_endterm_folds(train):
    folds = core.cv_folds(len(train))
    assert len(folds) == 5
    assert np.array_equal(np.sort(np.concatenate([va for _, va in folds])), np.arange(len(train)))


def test_base_cv_reproduces_endterm():
    base = core.cv_evaluate("base", n_jobs=5)
    assert base.mean == pytest.approx(core.ENDTERM_REFERENCE["cv_mae_M"], abs=1e-3)
    assert base.sd == pytest.approx(core.ENDTERM_REFERENCE["cv_mae_sd_M"], abs=1e-3)


def test_extra_block_and_zero_residual(train):
    noise = pd.DataFrame({"id": train.ids, "noise": np.random.default_rng(0).normal(size=len(train))})
    with_noise = core.cv_evaluate("noise", extra=noise, n_jobs=5)
    base = core.base_cv()
    assert not with_noise.passes_rule(base)          # a random column must not pass the 1-SD rule
    zero = core.cv_evaluate_residual(lambda d_tr, r, d_va: np.zeros(len(d_va)), name="zero")
    np.testing.assert_allclose(zero.fold_mae, base.fold_mae, rtol=1e-9)


def test_join_features_order_and_missing():
    extra = pd.DataFrame({"id": [3, 1], "a": [30.0, 10.0]})
    out = core.join_features([1, 2, 3], extra)
    assert out["a"].tolist()[0] == 10.0 and np.isnan(out["a"][1]) and out["a"][2] == 30.0
    with pytest.raises(ValueError):
        core.join_features([1], pd.DataFrame({"id": [1, 1], "a": [0, 1]}))


def test_metrics_known_values():
    m = core.metrics([100e6, 200e6], [110e6, 160e6])
    assert m["mae_M"] == pytest.approx(25.0) and m["mdape"] == pytest.approx(15.0)
    assert m["within10"] == 0.5 and m["within20"] == 1.0


def test_sanitize_text_drops_prices_contacts_digits():
    s = raw.sanitize_text("Срочно! Цена 45 млн тг. Звоните +7 701 123 45 67, @ivan_home. Евроремонт www.x.kz")
    assert s == "срочно звоните евроремонт"
    assert "[contact removed]" in raw.mask_contacts("тел 8 701 123 45 67")


def test_raw_details_table():
    r = raw.load_raw_details()
    assert r["id"].is_unique and len(r) >= 4815
    assert set(core.load_clean()["id"]) <= set(r["id"])
    assert r["photo_urls"].map(len).max() <= 5
    assert {"flooring", "door", "security", "furniture", "internet", "phone", "mortgaged_text"} <= set(r.columns)
    mh = raw.multi_hot(r["security"], "sec_", min_count=50)
    assert len(mh) == len(r) and mh.isin([0, 1]).all().all()
