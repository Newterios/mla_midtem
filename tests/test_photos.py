"""Fast checks of krisha_ml.photos (no network, < 10 s). Cache-dependent tests skip when caches are absent."""
import numpy as np
import pandas as pd
import pytest

from krisha_ml import core
from krisha_ml import photos as ph


def test_condition_labels_collapse():
    df = pd.DataFrame({"condition": ["fresh_renovation", "good", "needs_repair", "rough", "free_layout", "unknown", "other"]})
    assert ph.condition_labels(df).tolist() == [0, 1, 2, 3, 3, -1, -1]


def test_ad_mean_embedding_order_weights_and_missing():
    z = {"emb": np.array([[1, 0], [0, 1], [1, 1]], np.float16), "ad_id": np.array([5, 5, 7])}
    X = ph.ad_mean_embedding(z, [7, 9, 5])
    assert np.allclose(X[0], np.array([1, 1]) / np.sqrt(2)) and np.isnan(X[1]).all()
    assert np.allclose(X[2], np.array([1, 1]) / np.sqrt(2))
    Xw = ph.ad_mean_embedding(z, [5], weights=np.array([1.0, 0.0, 1.0]))
    assert Xw[0, 0] > 0.99


def test_aggregate_zero_shot_shapes():
    rng = np.random.default_rng(0)
    rooms = rng.dirichlet(np.ones(len(ph.ROOM_PROMPTS)), 6)
    qual = rng.dirichlet(np.ones(len(ph.QUALITY_PROMPTS)), 6)
    per = pd.DataFrame(np.hstack([rooms, qual]), columns=[f"room_{r}" for r in ph.ROOM_PROMPTS] + [f"q_{g}" for g in ph.QUALITY_PROMPTS])
    out = ph.aggregate_zero_shot(per, np.array([1, 1, 1, 2, 2, 3]))
    assert out["id"].tolist() == [1, 2, 3] and out["n_photos_ok"].tolist() == [3, 2, 1]
    assert np.allclose(out[[f"clip_mean_{g}" for g in ph.QUALITY_PROMPTS]].sum(axis=1), 1)
    assert out["clip_interior_share"].between(0, 1).all()


def test_probe_oof_never_predicts_with_own_label():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8)); y = rng.integers(0, 4, 200); y[:20] = -1; X[-5:] = np.nan
    folds = core.cv_folds(200)
    P = ph.probe_oof(X, y, folds, C=0.1)
    assert np.isnan(P[-5:]).all() and np.isfinite(P[:-5]).all() and np.allclose(P[:-5].sum(axis=1), 1)
    # changing a validation row's label must not change its own OOF prediction
    y2 = y.copy(); va0 = folds[0][1]; i = va0[(y[va0] >= 0) & (va0 < 195)][0]; y2[i] = (y[i] + 1) % 4
    assert np.allclose(ph.probe_oof(X, y2, folds, 0.1)[i], P[i])


def test_gradcam_on_tiny_effnet():
    timm = pytest.importorskip("timm")
    import torch
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4).eval()
    cam, c, prob = ph.gradcam(m, torch.randn(1, 3, 64, 64))
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1 and 0 <= c < 4 and np.isclose(prob.sum(), 1, atol=1e-5)


@pytest.mark.skipif(not ph.IMG_INDEX.exists(), reason="image cache not built")
def test_image_cache_covers_only_clean_ads_and_features_file():
    idx = pd.read_csv(ph.IMG_INDEX)
    assert set(idx["ad_id"]) <= set(core.load_clean()["id"])
    f = ph.OUT / "features_photo.csv"
    if f.exists():
        F = pd.read_csv(f)
        assert len(F) == 4815 and F["id"].is_unique
        assert set(F.loc[F["split"] == "test", "id"]) == core.load_test_ids()
