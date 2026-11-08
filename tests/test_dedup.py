"""Tests of the duplicate finder and the CV schemes (synthetic data, plus the saved group folds)."""
import numpy as np
import pandas as pd
import pytest

from krisha_ml import cv_schemes as cs
from krisha_ml import dedup


def _cards(n):
    return pd.DataFrame({"id": np.arange(1, n + 1, dtype="int64"), "city": "Astana", "rooms": 2.0,
                         "floor": np.arange(1, n + 1, dtype=float), "total_floors": 12.0, "year_built": 2020.0,
                         "area": 50.0 + 3 * np.arange(n), "price": 30e6 + 1e6 * np.arange(n),
                         "lat": 51.1 + 0.01 * np.arange(n), "lon": 71.4, "street": "s", "house_num": np.arange(n).astype(str),
                         "description": [f"unique text number {i} " * 10 for i in range(n)]})


def _hashes(rows):
    return pd.DataFrame(rows, columns=["ad_id", "k", "phash", "dhash"]).astype({"ad_id": "int64", "phash": np.uint64, "dhash": np.uint64})


def test_photo_pair_found_and_stock_image_ignored():
    df = _cards(10)
    df.loc[1, ["area", "floor"]] = df.loc[0, ["area", "floor"]].to_numpy()  # ads 1, 2: same flat, share a photo
    stock = 0xFFFF0000FFFF0000
    rows = [(1, 1, 0x0123456789ABCDEF, 0x0F0F0F0F0F0F0F0F), (2, 1, 0x0123456789ABCDEE, 0x0F0F0F0F0F0F0F0E)]
    rows += [(a, 2, stock, stock) for a in range(3, 10)]          # one image on 7 ads -> hub
    rows += [(a, 3, (a * 0x9E3779B97F4A7C15) % 2**64, (a * 0xC2B2AE3D27D4EB4F) % 2**64) for a in range(1, 11)]
    rep = dedup.find_duplicates(df, hashes=_hashes(rows), use_emb=False)
    assert set(zip(rep.pairs["id_a"], rep.pairs["id_b"])) == {(1, 2)}
    assert rep.pairs.loc[0, "tier"] == "same_flat"
    g = rep.groups.set_index("id")
    assert g.loc[1, "group"] == g.loc[2, "group"] == 1 and g.loc[3, "group_size"] == 1


def test_photo_link_needs_plausible_cards():
    df = _cards(4)
    df.loc[1, ["area", "rooms"]] = [120.0, 4.0]   # shares a photo with ad 1 but a different flat
    rows = [(1, 1, 0x0123456789ABCDEF, 0x0F0F0F0F0F0F0F0F), (2, 1, 0x0123456789ABCDEF, 0x0F0F0F0F0F0F0F0F)]
    assert dedup.find_duplicates(df, hashes=_hashes(rows), use_emb=False).pairs.empty


def test_structured_match():
    df = _cards(5)
    df.loc[4, ["floor", "area", "price", "lat", "description"]] = [1.0, 50.3, 30.5e6, 51.1005, "other"]  # twin of ad 1
    pairs = dedup.structured_pairs(df)
    assert list(zip(pairs["id_a"], pairs["id_b"])) == [(1, 5)]
    df.loc[4, "price"] = 40e6
    assert dedup.structured_pairs(df).empty


def test_merge_labels_and_group_folds():
    a = np.array(["x", "x", "y", None, "z", "w"], dtype=object)
    b = np.array([None, None, 1, 1, 2, None], dtype=object)
    m = cs.merge_labels(a, b)
    assert m[0] == m[1] and m[2] == m[3] and m[0] != m[2] and m[4] != m[5]
    groups = np.repeat(np.arange(40), 3)
    folds = cs.group_folds(groups, seed=42)
    fv = cs.fold_vector(folds, len(groups))
    assert (pd.Series(fv).groupby(groups).nunique() == 1).all()
    assert sorted(np.concatenate([va for _, va in folds]).tolist()) == list(range(len(groups)))


def test_saved_group_folds_cover_train():
    if not cs.FOLDS_GROUP_CSV.exists():
        pytest.skip("run scripts/leakage_audit.py first")
    folds = cs.load_group_folds()
    f = pd.read_csv(cs.FOLDS_GROUP_CSV)
    assert len(folds) == 5 and sum(len(va) for _, va in folds) == 3852
    assert (f.groupby("dup_group")["fold"].nunique() == 1).all()


def test_repair_folds_moves_whole_groups_only():
    fv = np.array([0, 1, 1, 2, 3, 4, 0])
    groups = np.array([7, 7, 7, 8, 9, 9, 10])
    out = cs.repair_folds(fv, groups)
    assert out[:3].tolist() == [1, 1, 1]                 # majority fold of group 7
    assert out[4] == out[5] and out[3] == 2 and out[6] == 0
    assert (out != fv).sum() == 2
