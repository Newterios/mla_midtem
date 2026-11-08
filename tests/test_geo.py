"""Tests for krisha_ml.geo (need the cached OSM layers in cache/osm/; skipped otherwise)."""
import numpy as np
import pandas as pd
import pytest

from krisha_ml import core, geo

needs_osm = pytest.mark.skipif(
    not all((geo.OSM_DIR / f"{c}_{l}.json").exists() for l, (_, _, cs) in geo.LAYERS.items() for c in cs),
    reason="OSM cache missing: run scripts/run_geo.py once")


def _ads():
    return pd.DataFrame({"id": [1, 2, 3, 4], "city": ["Astana", "Almaty", "Almaty", "Astana"],
                         "lat": [51.128207, 43.238045, np.nan, 51.20], "lon": [71.430411, 76.945465, np.nan, 71.43]})


@needs_osm
def test_geo_features_shape_and_semantics():
    f = geo.geo_features(_ads())
    assert list(f["id"]) == [1, 2, 3, 4]
    assert f.loc[:1, "geo_dist_centre_km"].max() < 1e-3                    # centre points
    assert abs(f.loc[3, "geo_dist_centre_km"] - 8.0) < 0.2                  # 0.0718 deg lat ~ 8.0 km
    assert f.loc[2].drop("id").isna().all()                                 # no coordinates -> NaN
    assert np.isnan(f.loc[0, "geo_dist_metro_km"]) and f.loc[1, "geo_dist_metro_km"] < 1.0  # metro Almaty only
    km = f.filter(like="_km")
    assert (km.isna() | (km >= 0)).all().all()


@needs_osm
def test_haversine_matches_balltree():
    pts = geo.layer_points("school", "Astana")
    q = np.array([[51.15, 71.40]])
    d_tree = geo.BallTree(np.radians(pts), metric="haversine").query(np.radians(q))[0][0, 0] * geo.EARTH_KM
    la1, lo1, la2, lo2 = map(np.radians, (q[0, 0], q[0, 1], pts[:, 0], pts[:, 1]))
    hav = 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    assert abs(d_tree - hav.min() * geo.EARTH_KM) < 1e-6


def test_rawfield_features_keyed_and_typed():
    r = geo.rawfield_features()
    clean_ids = set(core.load_clean()["id"])
    assert r["id"].is_unique and clean_ids <= set(r["id"])
    sec = r.filter(like="raw_sec_")
    assert sec.shape[1] >= 5
    assert sec[r["raw_sec_n_items"].isna()].isna().all().all()             # missing security stays missing
    assert set(r["raw_exchange"].dropna().unique()) <= {0.0, 1.0}
    assert not set(r.columns) & set(core.FEATURES)
