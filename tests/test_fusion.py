"""Tests of krisha_ml.fusion on small synthetic frames (no track outputs needed), plus one quick
end-to-end run of scripts/run_final.py on tiny subsets of the real data (skipped if data are missing)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
from krisha_ml import core, fusion  # noqa: E402
from krisha_ml.fusion import Block, Settings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORDS = ["евроремонт", "новый", "дом", "центр", "парк", "вид", "старый", "ремонт", "требует", "срочно", "мебель", "школа"]


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def synth():
    """160 fake ads with the 24 base columns, a description and a price that depends on area,
    a 'geo' value and a text word, so the blocks carry real signal."""
    rng = np.random.default_rng(0)
    n = 160
    area = rng.uniform(30, 150, n)
    geo = rng.normal(size=n)
    lux = rng.random(n) < 0.4
    df = pd.DataFrame({
        "id": np.arange(1000, 1000 + n, dtype="int64"), "area": area, "log_area": np.log(area), "rooms": np.clip(area // 30, 1, 5),
        "kitchen_area": rng.uniform(5, 20, n), "floor": rng.integers(1, 20, n), "total_floors": 20, "year_built": rng.integers(1960, 2025, n),
        "ceiling_height": 2.8, "n_photos": rng.integers(0, 30, n), "lat": 51.1 + rng.normal(0, .02, n), "lon": 71.4 + rng.normal(0, .02, n),
        "city": rng.choice(["Astana", "Almaty"], n), "district": rng.choice(["a", "b", "c"], n), "building_type": "monolithic",
        "condition": rng.choice(["good", "fresh_renovation"], n), "bathroom": "combined", "furnished": "unknown", "balcony": "unknown",
        "parking": "unknown", "former_dorm": "unknown", "seller_type": rng.choice(["owner", "specialist"], n),
        "has_complex": True, "mortgaged": False, "is_urgent": False,
        "description": [" ".join(rng.choice(WORDS, 8)) + (" евроремонт" * 3 if l else " требует ремонт") for l in lux]})
    df["price"] = 4e5 * area * np.exp(0.25 * geo + 0.3 * lux + rng.normal(0, 0.05, n))
    geo_tab = pd.DataFrame({"id": df["id"], "dist": geo, "noise": rng.normal(size=n)}).set_index("id")
    geo_tab.columns = [f"geo__{c}" for c in geo_tab.columns]
    emb = rng.normal(size=(n, 8)).astype(np.float32)
    emb[:, 0] += lux
    emb[::10] = np.nan                                    # ads without the embedding
    emb_tab = pd.DataFrame(emb, index=pd.Index(df["id"], name="id"), columns=[f"photo_emb__e{i}" for i in range(8)])
    blocks = {"geo": Block("geo", "tab", "synthetic", geo_tab), "text": Block("text", "text", "description"),
              "photo_emb": Block("photo_emb", "emb", "synthetic", emb_tab)}
    frame = fusion.attach_blocks(df, blocks)
    return frame, df["price"].to_numpy(float), blocks


@pytest.fixture
def quick():
    s = Settings.quick()
    s.n_boot = 100
    return s


# --------------------------------------------------------------------------- loading
def test_read_id_table_drops_oof_trainonly_and_constant_columns(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"id": [1, 2, 3, 4, 4], "good": [1., 2., 3., 4., 4.], "oof_delta_x": [.1, .2, .3, .4, .4],
                  "trainonly": [1., 2., np.nan, np.nan, np.nan], "const": [5, 5, 5, 5, 5], "fold": [0, 1, 2, 3, 3],
                  "flag": ["True", "False", "True", "False", "False"]}).to_csv(p, index=False)
    t, notes = fusion.read_id_table(p, "blk", test_ids={3, 4})
    assert list(t.columns) == ["blk__good", "blk__flag"]
    assert t.index.is_unique and t.index.dtype == "int64"
    assert t["blk__flag"].tolist() == [1.0, 0.0, 1.0, 0.0]          # bool strings -> numeric branch
    assert any("duplicated" in n for n in notes)


def test_read_id_table_requires_id(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"x": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="id"):
        fusion.read_id_table(p, "b")


@pytest.mark.parametrize("payload,expected", [
    ({"keep": False}, {"geo": False, "rawfields": False}),
    ({"geo": {"keep": True}, "rawfields": False}, {"geo": True, "rawfields": False}),
    ({"blocks": [{"block": "geo", "keep": True}]}, {"geo": True}),
    ({"cv_mae": 6.5}, {}),
])
def test_keep_decisions(tmp_path, payload, expected):
    (tmp_path / "geo").mkdir()
    (tmp_path / "geo" / "decision.json").write_text(json.dumps(payload))
    assert fusion.read_keep_decisions(tmp_path) == expected


def test_load_blocks_degrades_gracefully(tmp_path):
    (tmp_path / "geo").mkdir()
    pd.DataFrame({"id": [1, 2, 3], "d": [0.1, 0.2, 0.3]}).to_csv(tmp_path / "geo" / "features_geo.csv", index=False)
    (tmp_path / "geo" / "features_rawfields.csv").write_text("id,a\n1,\"broken")     # half-written file
    blocks, msgs = fusion.load_blocks(["geo", "rawfields", "photo", "text_emb", "photo_emb", "text"], out_dir=tmp_path,
                                      cache_dir=tmp_path / "nocache", test_ids={3}, root=tmp_path)
    assert list(blocks) == ["geo", "text"]
    assert blocks["geo"].columns == ["geo__d"]
    skipped = [m for m in msgs if m.startswith("[skip]")]
    assert len(skipped) == 4 and any("photo" in m for m in skipped)


def test_attach_blocks_keeps_order_and_missing_as_nan(synth):
    frame, _, blocks = synth
    shuffled = frame[["id", "description"] + list(core.FEATURES)].iloc[::-1].reset_index(drop=True)
    out = fusion.attach_blocks(shuffled, blocks)
    assert (out["id"].to_numpy() == shuffled["id"].to_numpy()).all()
    assert np.allclose(out["geo__dist"].to_numpy(), blocks["geo"].table.loc[shuffled["id"], "geo__dist"].to_numpy())
    assert out["photo_emb__e0"].isna().sum() == 16
    assert out["text_clean"].str.len().gt(0).all()


def test_fold_schemes_group_restricts_and_validates(tmp_path):
    ids = np.arange(100, dtype="int64")
    pd.DataFrame({"id": ids, "fold": ids % 5, "dup_group": ids // 2}).to_csv(tmp_path / "fg.csv", index=False)
    schemes, _ = fusion.fold_schemes(ids[:50], tmp_path / "fg.csv", full_train=False)
    assert set(schemes) == {"random", "group"}
    va = np.concatenate([v for _, v in schemes["group"].folds])
    assert sorted(va) == list(range(50))
    schemes, msgs = fusion.fold_schemes(np.arange(200, dtype="int64"), tmp_path / "fg.csv", full_train=False)
    assert set(schemes) == {"random"} and any("missing" in m for m in msgs)


def test_inner_folds_keep_groups_together():
    groups = np.repeat(np.arange(20), 3)
    for tr, va in fusion.inner_folds(60, groups, 5):
        assert not set(groups[tr]) & set(groups[va])


# --------------------------------------------------------------------------- models
@pytest.mark.parametrize("strategy", ["base", "A", "B", "C", "Cb"])
def test_models_fit_predict(synth, quick, strategy):
    frame, y, blocks = synth
    model = fusion.make_model(strategy, list(blocks.values()), quick)
    model.fit(frame.iloc[:120], y[:120])
    pred = model.predict(frame.iloc[120:])
    assert pred.shape == (40,) and np.isfinite(pred).all() and (pred > 0).all()


def test_residual_stack_without_blocks_is_base(synth, quick):
    frame, y, _ = synth
    a = fusion.ResidualStackModel([], quick).fit(frame.iloc[:120], y[:120])
    b = fusion.BaseModel(settings=quick).fit(frame.iloc[:120], y[:120])
    assert np.allclose(a.predict(frame.iloc[120:]), b.predict(frame.iloc[120:]))


def test_blend_weights_are_nonnegative_and_find_signal():
    rng = np.random.default_rng(1)
    z_good, z_bad = rng.normal(size=500), rng.normal(size=500)
    resid = 0.8 * z_good - 0.5 * z_bad + rng.normal(0, .01, 500)
    w = fusion.blend_weights(np.column_stack([z_good, z_bad]), resid)
    assert w[0] == pytest.approx(0.8, abs=0.02) and w[1] == 0.0
    assert fusion.blend_weights(np.zeros((5, 0)), np.zeros(5)).shape == (0,)


def test_stacking_cv_matches_refit_model_and_caches(synth, quick, tmp_path):
    frame, y, blocks = synth
    scheme = fusion.FoldScheme("random", fusion.core.cv_folds(len(y)))
    st = fusion.StackingCV(frame, y, scheme, quick, cache_dir=tmp_path)
    base = st.base_result()
    res, W = st.evaluate([blocks["geo"], blocks["text"]])
    assert W.shape == (5, 2) and (W >= 0).all()
    assert res.mean < base.mean                          # the synthetic blocks carry signal
    assert len(list(tmp_path.glob("fusion_base_random_*.npz"))) == 1
    zero, _ = st.evaluate([])
    assert np.allclose(zero.oof_pred, base.oof_pred)
    # B on the same folds runs and returns one prediction per ad
    fb = fusion.FeatureCV(frame, y, scheme, quick).evaluate([blocks["geo"], blocks["photo_emb"], blocks["text"]])
    assert np.isfinite(fb.oof_pred).all() and len(fb.fold_mae) == 5


# --------------------------------------------------------------------------- selection
def _fake_result(mae: float, sd: float) -> core.CVResult:
    folds = mae + sd * np.array([-1.4142, -0.7071, 0, 0.7071, 1.4142])   # population SD == sd
    return core.CVResult("x", folds, folds, folds, np.zeros(1))


def test_forward_select_applies_one_sd_rule():
    base = _fake_result(6.6, 0.35)
    table = {("a",): 6.1, ("b",): 6.5, ("c",): 6.7, ("a", "b"): 5.9, ("a", "c"): 6.2, ("a", "b", "c"): 5.95}
    ev = lambda names: _fake_result(table[tuple(sorted(names))], 0.3)
    out = fusion.forward_select(ev, ["a", "b", "c"], base, log=lambda m: None)
    assert out["kept"] == ["a"]                 # 6.6 -> 6.1 passes (0.5 > 0.35); 6.1 -> 5.9 fails (0.2 < 0.3)
    assert out["greedy"] == ["a", "b"]          # greedy path stops when nothing lowers the MAE
    assert [p["passes_step_rule"] for p in out["path"]] == [True, False]


def test_forward_select_nothing_passes():
    base = _fake_result(6.6, 0.35)
    out = fusion.forward_select(lambda n: _fake_result(6.5 if len(n) == 1 else 6.45, .3), ["a", "b"], base, log=lambda m: None)
    assert out["kept"] == [] and out["greedy"] == ["a", "b"]


# --------------------------------------------------------------------------- test statistics
def test_bootstrap_ci_contains_point_and_paired_diff():
    rng = np.random.default_rng(2)
    y = rng.uniform(2e7, 9e7, 400)
    pred = y * np.exp(rng.normal(0, .1, 400))
    worse = y * np.exp(rng.normal(0, .2, 400))
    b = fusion.bootstrap_metrics(y, pred, worse, n_boot=300)
    for m in ("mae_M", "mdape", "r2", "within10"):
        assert b[m]["ci_low"] <= b[m]["value"] <= b[m]["ci_high"]
    assert b["mae_M"]["value"] == pytest.approx(core.metrics(y, pred)["mae_M"])
    assert b["mae_M"]["diff_ci_high"] < 0                     # clearly better than the noisier reference
    same = fusion.bootstrap_metrics(y, pred, pred, n_boot=50)
    assert same["mae_M"]["diff_ci_low"] == same["mae_M"]["diff_ci_high"] == 0


def test_segment_table_and_segments(synth, tmp_path):
    frame, y, blocks = synth
    dup = pd.DataFrame({"id": frame["id"], "group": frame["id"], "group_size": 1, "split": "test"})
    dup.loc[:1, ["group", "group_size"]] = [[1000, 2], [1000, 2]]
    dup.loc[0, "split"] = "train"
    dup.to_csv(tmp_path / "dup.csv", index=False)
    seg = fusion.test_segments(frame, blocks, tmp_path / "dup.csv")
    assert {"city", "price_band", "rooms", "photos", "text", "duplicate", "train_twin"} <= set(seg.columns)
    assert (seg["duplicate"] == "in duplicate group").sum() == 2 and (seg["train_twin"] == "has a twin in train").sum() == 2
    t = fusion.segment_table(seg, y, y * 1.1, y, n_boot=50)
    assert t.groupby("segment")["n"].sum().eq(len(y)).all()
    assert np.allclose(t["mdape"], 10.0) and np.allclose(t["base_mae_M"], 0)


def test_price_band_edges():
    assert list(fusion.price_band(np.array([1e6, 25e6, 59.9e6, 1e8]))) == ["<25M", "25-40M", "40-60M", ">=100M"]


# --------------------------------------------------------------------------- November
def test_reduced_features_drop_missing_fields(synth, quick):
    frame, y, _ = synth
    nov = frame.copy()
    nov[["lat", "lon", "kitchen_area", "n_photos"]] = np.nan
    nov["balcony"] = "unknown"
    num, cat, cov = fusion.reduced_features(nov)
    assert not {"lat", "lon", "kitchen_area", "n_photos", "balcony", "parking"} & set(num + cat)
    assert {"area", "rooms", "has_complex"} <= set(num) and "city" in cat
    X = core.encode_features(frame)
    m = fusion.make_reduced_model(num, cat, quick).fit(X[num + cat].iloc[:120], y[:120])
    assert np.isfinite(m.predict(X[num + cat].iloc[120:])).all()


def test_drift_table():
    o = pd.DataFrame({"city": ["A", "A", "B"], "area": [50., 50., 100.]})
    n = pd.DataFrame({"city": ["A", "B"], "area": [50., 100.]})
    d = fusion.drift_table(np.array([5e7, 5e7, 1e8]), o, np.array([5.5e7, 1e8]), n)
    assert d.set_index("city").loc["A", "price_m2_change_pct"] == pytest.approx(10.0)


def test_metadata_is_json_serialisable(synth, quick, tmp_path):
    frame, y, blocks = synth
    m = fusion.ResidualStackModel([blocks["geo"]], quick).fit(frame.iloc[:100], y[:100])
    meta = fusion.model_metadata(m, [blocks["geo"]], quick)
    fusion.write_json(tmp_path / "m.json", meta)
    back = json.loads((tmp_path / "m.json").read_text())
    assert back["strategy"] == "A" and "geo__dist" in back["input_columns"] and "geo" in back["blend_weights"]


# --------------------------------------------------------------------------- end-to-end smoke
@pytest.mark.skipif(not (core.CLEAN_CSV.exists() and core.SPLIT_CSV.exists()), reason="clean data not present")
def test_run_final_quick_end_to_end(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "run_final.py"), "--quick", "--quick-n", "200", "--out", str(tmp_path),
           "--blocks", "geo", "text", "photo_emb"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_final", ROOT / "scripts" / "run_final.py")
    run_final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_final)
    assert "test" not in run_final.parse_args([]).stages   # the one test evaluation is never a default stage
    stages = ["compare", "select", "ablate", "decide", "fit", "test", "nov", "figures", "export"]
    r = subprocess.run(cmd[:2] + stages + cmd[2:], capture_output=True, text=True, timeout=900,
                       env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for f in ("cv_comparison.csv", "forward_selection.csv", "selection.json", "test_metrics.json", "test_segments.csv",
              "models/final_model.joblib", "models/final_model_meta.json", "fig_model_comparison.png"):
        assert (tmp_path / f).exists(), f
    tm = json.loads((tmp_path / "test_metrics.json").read_text())
    assert tm["quick"] is True and tm["metrics"]["n_boot"] == 200
    # export: the layout the notebook, the report and the slides read (scripts/final_build_common.py)
    for f in ("fusion/selection.csv", "fusion/selection.json", "final_eval/test_metrics.json", "final_eval/results_table.csv",
              "final_eval/nov_holdout.json", "final_eval/error_by_segment.csv", "final_eval/worst_cases.csv"):
        assert (tmp_path / f).exists(), f
    ev = json.loads((tmp_path / "final_eval" / "test_metrics.json").read_text())
    assert ev["chosen"]["mae_M"] == pytest.approx(tm["metrics"]["mae_M"]["value"]) and len(ev["chosen"]["bootstrap95_mae"]) == 2
    assert {"cv_mae_M", "group_cv_mae_M", "chosen", "reason"} <= set(json.loads((tmp_path / "fusion" / "selection.json").read_text()))
    # the test stage refuses a second evaluation without --allow-retest
    r2 = subprocess.run(cmd[:2] + ["test"] + cmd[2:], capture_output=True, text=True, timeout=600)
    assert r2.returncode != 0 and "evaluated once" in (r2.stdout + r2.stderr)
    assert not (ROOT / "models" / "final_model_quick.joblib").exists()


def test_saved_model_keeps_no_block_tables(synth, quick, tmp_path):
    import joblib
    frame, y, blocks = synth
    m = fusion.make_model("Cb", list(blocks.values()), quick).fit(frame.iloc[:120], y[:120])
    assert all(b.table is None and b.columns == blocks[b.name].columns for b in m.blocks)
    joblib.dump(m, tmp_path / "m.joblib")
    back = joblib.load(tmp_path / "m.joblib")
    assert np.allclose(back.predict(frame.iloc[120:]), m.predict(frame.iloc[120:]))
    assert back.blocks[0].digest() == blocks[back.blocks[0].name].digest()
