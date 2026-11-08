"""Tests of the demo app's inference API (app/predict.py). No Gradio, no CLIP download: the photo
model is a stub, and the final model path points to a temp dir so the tests do not depend on
whether models/final_model.joblib exists yet."""
from __future__ import annotations

import io
import math

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app import predict as P
from krisha_ml import core

VALID = {"city": "Astana", "district": "Esilyskiy", "area": 62, "rooms": 2, "floor": 5, "total_floors": 12,
         "year_built": 2019, "building_type": "monolithic", "condition": "fresh_renovation",
         "lat": 51.128, "lon": 71.43, "asking_price_m": 45}


class StubPhotoModel:
    """Says every photo is a bare-concrete shell with 90 % confidence."""
    classes = P.PHOTO_CLASSES
    source = "stub"

    def predict_proba(self, images):
        return np.tile([0.04, 0.04, 0.02, 0.90], (len(images), 1))


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    missing = tmp_path_factory.mktemp("models") / "final_model.joblib"
    return P.Predictor(final_model_path=missing, photo_model=StubPhotoModel())


def _png(color=(200, 180, 160), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- validation
def test_validate_valid_input_coerces_types():
    x = P.validate({**VALID, "area": "62,5", "rooms": "2", "mortgaged": "да", "description": "  текст  "})
    assert x.area == 62.5 and x.rooms == 2 and isinstance(x.rooms, int)
    assert x.mortgaged is True and x.description == "текст"
    assert x.bathroom == "unknown" and x.seller_type == "owner"      # defaults


@pytest.mark.parametrize("patch, fragment", [
    ({"city": "Shymkent"}, "city must be Astana or Almaty"),
    ({"district": "Medeuskiy"}, "is not in Astana"),
    ({"area": -5}, "Total area"),
    ({"area": "abc"}, "must be a number"),
    ({"rooms": 2.5}, "whole number"),
    ({"floor": 15, "total_floors": 9}, "cannot be above total floors"),
    ({"year_built": 1700}, "Year built"),
    ({"kitchen_area": 80}, "Kitchen area"),
    ({"lat": 43.2, "lon": 76.9}, "outside Astana"),
    ({"lon": None}, "both latitude and longitude"),
    ({"condition": "luxury"}, "condition must be one of"),
    ({"asking_price_m": -5}, "Asking price"),
    ({"description": "x" * (P.MAX_DESCRIPTION + 1)}, "too long"),
], ids=lambda v: v if isinstance(v, str) else None)
def test_validate_rejects_bad_input(patch, fragment):
    with pytest.raises(P.ValidationError) as e:
        P.validate({**VALID, **patch})
    assert any(fragment in msg for msg in e.value.errors), e.value.errors


def test_validate_optional_numbers_left_empty_arrive_as_zero():
    """An empty optional number box in the UI is delivered as 0 once its widget has been rendered
    (Gradio Number, e.g. inside the opened "More details" accordion): 0 means "not given", not an error."""
    x = P.validate({**VALID, "asking_price_m": 0, "kitchen_area": 0, "ceiling_height": 0, "n_photos": 0, "lat": 0, "lon": 0})
    assert x.asking_price_m is None and x.kitchen_area is None and x.ceiling_height is None
    assert x.n_photos is None and x.lat is None and x.lon is None


def test_validate_reports_all_errors_at_once():
    with pytest.raises(P.ValidationError) as e:
        P.validate({"city": "Astana"})
    assert len(e.value.errors) >= 5          # area, rooms, floor, total floors, year
    assert all(isinstance(m, str) and m for m in e.value.errors)


def test_predict_raises_validation_error_for_raw_dict(predictor):
    with pytest.raises(P.ValidationError):
        predictor.predict({**VALID, "rooms": 0})


# --------------------------------------------------------------------------- prediction
def test_predict_valid_input_full_result(predictor):
    r = predictor.predict(P.validate(VALID))
    assert 10e6 < r.price_kzt < 150e6 and math.isfinite(r.price_kzt)
    assert r.model_name.startswith("Endterm base")
    lo80, hi80 = r.intervals["80"]
    lo90, hi90 = r.intervals["90"]
    assert lo90 <= lo80 < r.price_kzt < hi80 <= hi90
    assert r.verdict["code"] in {"fair", "high", "low", "over", "under"}
    assert r.verdict["gap_pct"] == pytest.approx(100 * (45e6 / r.price_kzt - 1))
    # SHAP: additive on the log scale and sorted by size
    assert r.shap is not None and {"location", "area"} <= set(r.shap["group"])
    assert r.shap["shap_log"].abs().is_monotonic_decreasing
    # comparables: 5 training ads from the same city, never test ads
    assert r.comparables is not None and len(r.comparables) == 5
    assert set(r.comparables["ad id"]).isdisjoint(core.load_test_ids())
    assert isinstance(r.few_comparables, bool)
    assert r.photo == {"status": "no photos", "n_photos": 0}
    d = r.to_dict()
    assert isinstance(d["comparables"], list) and isinstance(d["shap"], list)


def test_predict_matches_endterm_model_on_a_training_ad(predictor):
    """Form -> features mapping reproduces the Endterm encoding exactly."""
    tr = core.load_train()
    df = tr.df
    ok = (df[["floor", "total_floors", "lat", "kitchen_area", "ceiling_height", "n_photos"]].notna().all(axis=1)
          & (df["floor"] <= df["total_floors"]) & df["district"].notna()
          & df["condition"].isin(P.CHOICES["condition"]) & df["bathroom"].isin(P.CHOICES["bathroom"])
          & df["building_type"].isin(P.CHOICES["building_type"]) & (df["kitchen_area"] < df["area"])
          & df["ceiling_height"].between(2, 6) & df["kitchen_area"].between(2, 200))
    i = int(np.flatnonzero(ok.to_numpy())[0])
    row = df.iloc[i]
    raw = {k: row[k] for k in ["city", "district", "area", "rooms", "floor", "total_floors", "year_built",
                               "building_type", "condition", "kitchen_area", "ceiling_height", "bathroom",
                               "furnished", "balcony", "parking", "former_dorm", "seller_type", "n_photos",
                               "lat", "lon", "has_complex", "mortgaged", "is_urgent"]}
    raw = {k: (v.item() if hasattr(v, "item") else v) for k, v in raw.items()}
    r = predictor.predict(raw)
    expected = predictor.base_model.predict(tr.X.iloc[[i]])[0]
    assert r.price_kzt == pytest.approx(expected, rel=1e-9)


def test_verdict_rules():
    iv = {"80": (90.0, 110.0), "90": (80.0, 120.0)}
    assert P.verdict_for(100, 100, iv)["code"] == "fair"
    assert P.verdict_for(115, 100, iv)["code"] == "high"
    assert P.verdict_for(130, 100, iv)["code"] == "over"
    assert P.verdict_for(85, 100, iv)["code"] == "low"
    assert P.verdict_for(70, 100, iv)["code"] == "under"
    assert P.verdict_for(125, 100, None)["code"] == "over"          # Endterm +-20 % rule
    assert P.verdict_for(110, 100, None)["code"] == "fair"


def test_text_red_flags_and_form_mismatch(predictor):
    text = "Продается по переуступке, только наличный расчет. Черновая отделка. Звоните +7 701 123 45 67"
    r = predictor.predict({**VALID, "description": text})
    keys = {f["key"] for f in r.text_flags}
    assert {"assignment_of_rights", "cash_only", "rough_finish", "contacts", "mismatch"} <= keys
    assert r.text_flags[0]["severity"] == "red"                      # red flags first


def test_no_description_no_flags(predictor):
    assert predictor.predict(VALID).text_flags == []


def test_unusual_flat_warns(predictor):
    r = predictor.predict({**VALID, "area": 600, "rooms": 9, "asking_price_m": None})
    assert r.verdict is None
    assert any("extrapolation" in w for w in r.warnings)
    assert r.few_comparables is True
    assert any("Few comparables" in w for w in r.warnings)


# --------------------------------------------------------------------------- photos
def test_missing_photos_is_not_an_error(predictor):
    for photos in (None, [], [None]):
        r = predictor.predict(VALID, photos=photos)
        assert r.photo["status"] == "no photos"
        assert not any("Photo" in w for w in r.warnings)


def test_bad_images_are_skipped_without_crashing(predictor, tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    r = predictor.predict(VALID, photos=[str(broken), b"plain bytes", str(tmp_path / "missing.png")])
    assert r.photo["status"] == "no readable photos"
    assert sum("not a readable image" in w for w in r.warnings) == 3
    assert r.price_kzt > 0


def test_photo_condition_mismatch_warning(predictor, tmp_path):
    good = tmp_path / "room.png"
    good.write_bytes(_png())
    r = predictor.predict(VALID, photos=[str(good), _png((10, 10, 10)), b"junk"])   # form: fresh renovation
    assert r.photo["status"] == "ok" and r.photo["n_photos"] == 2
    assert r.photo["predicted"] == "shell"
    assert sum(r.photo["proba"].values()) == pytest.approx(1)
    assert "fresh_renovation" in r.photo["mismatch"]
    assert any(w.startswith("Photo vs stated condition") for w in r.warnings)
    assert any("not a readable image" in w for w in r.warnings)
    # agreeing condition: no mismatch
    r2 = predictor.predict({**VALID, "condition": "rough"}, photos=[str(good)])
    assert r2.photo["mismatch"] is None and r2.photo["comment"].startswith("Photos agree")


def test_too_many_photos_uses_first_three():
    loaded, warns = P.load_photos([_png()] * 5)
    assert len(loaded) == P.MAX_PHOTOS and any("only the first" in w for w in warns)


def test_tiny_and_array_images():
    loaded, warns = P.load_photos([np.zeros((10, 10, 3), np.uint8), np.full((40, 50, 3), 200, np.uint8)])
    assert len(loaded) == 1 and "too small" in warns[0]
    assert P.square_224(loaded[0].image).shape == (224, 224, 3)


# --------------------------------------------------------------------------- degradation and final model
def test_missing_modules_degrade_with_notes(monkeypatch, tmp_path):
    real = P._import_optional
    monkeypatch.setattr(P, "_import_optional",
                        lambda name: (None, "ModuleNotFoundError: stub") if name in ("krisha_ml.trust", "krisha_ml.text")
                        else real(name))
    pr = P.Predictor(final_model_path=tmp_path / "none.joblib", photo_model=None)
    r = pr.predict({**VALID, "description": "срочно, торг"}, photos=[_png()])
    assert r.price_kzt > 0 and r.intervals is not None               # intervals need no trust module
    assert r.shap is None and r.comparables is None
    notes = " ".join(r.notes)
    assert "SHAP reasons: module not available yet" in notes
    assert "Comparable listings: module not available yet" in notes
    assert "Text red flags: module not available yet" in notes
    assert "Photo condition: module not available yet" in notes
    st = pr.status()
    assert "not available" in st["shap"] and "disabled" in st["photos"]


def test_final_model_bundle_is_used(tmp_path, predictor):
    path = tmp_path / "final_model.joblib"
    joblib.dump({"model": predictor.base_model, "name": "Stub final"}, path)
    pr = P.Predictor(final_model_path=path, photo_model=None)
    r = pr.predict(VALID)
    assert pr.is_final and r.model_name == "Stub final"
    assert r.price_kzt == pytest.approx(predictor.predict(VALID).price_kzt)


def test_final_model_with_extra_columns_gets_imputed_inputs(tmp_path):
    tr = core.load_train()
    extra = pd.DataFrame({"geo_dist_centre_km": np.random.default_rng(0).random(len(tr)), "raw_unknown_x": 1.0})
    X = pd.concat([tr.X, extra], axis=1)
    model = core.make_base_model(extra_num=list(extra.columns), max_iter=30).fit(X, tr.y)
    path = tmp_path / "final_model.joblib"
    joblib.dump(model, path)
    pr = P.Predictor(final_model_path=path, photo_model=None)
    r = pr.predict(VALID)
    assert r.model_name == "Final model" and r.price_kzt > 0
    assert any("raw_unknown_x" in n for n in r.notes)


def test_corrupt_final_model_falls_back(tmp_path):
    path = tmp_path / "final_model.joblib"
    path.write_bytes(b"not a pickle")
    pr = P.Predictor(final_model_path=path, photo_model=None)
    assert not pr.is_final and "could not be loaded" in pr.status()["model"]
    assert pr.predict(VALID).model_name.startswith("Endterm base")


class _FakeBlock:
    def __init__(self, name, columns=()):
        self.name, self.columns = name, list(columns)


class FakeFusionModel:
    """Shaped like a krisha_ml.fusion model: predicts on a one-row frame of clean + block columns."""
    strategy = "Cb"
    blocks = [_FakeBlock("geo", ["geo__geo_dist_center_km"]), _FakeBlock("rawfields", ["rawfields__raw_door"]),
              _FakeBlock("text_flags", ["text_flags__flag_urgent", "text_flags__flag_bargain"]),
              _FakeBlock("text_emb", [f"text_emb__e{i}" for i in range(4)]),
              _FakeBlock("photo_emb", [f"photo_emb__e{i}" for i in range(3)])]
    seen = []

    def predict(self, frame):
        FakeFusionModel.seen.append(frame.copy())
        return np.array([5.0e7])


class BrokenFusionModel(FakeFusionModel):
    def predict(self, frame):
        raise KeyError("text_emb__e0")


def test_fusion_final_model_gets_a_form_built_frame(tmp_path):
    path = tmp_path / "final_model.joblib"
    joblib.dump(FakeFusionModel(), path)
    pr = P.Predictor(final_model_path=path, photo_model=None, text_encoder=False)
    st = pr.status()["model"]
    assert pr.is_final and "always missing in the app: rawfields" in st
    FakeFusionModel.seen.clear()
    r = pr.predict({**VALID, "description": "Срочно продам, торг уместен"})
    assert r.price_kzt == pytest.approx(5.0e7) and r.model_name.startswith("Final")
    assert r.base_price_kzt is not None and r.base_price_kzt > 0
    frame = FakeFusionModel.seen[-1]
    assert len(frame) == 1 and {"area", "city", "text_flags__flag_urgent", "photo_emb__e0", "rawfields__raw_door"} <= set(frame.columns)
    assert frame["text_flags__flag_urgent"].iloc[0] == 1 and frame["photo_emb__e0"].isna().all()
    assert any("left missing" in n and "rawfields" in n for n in r.notes)
    assert r.intervals and r.intervals["80"][0] < r.price_kzt < r.intervals["80"][1]


def test_broken_fusion_model_falls_back_to_base(tmp_path):
    path = tmp_path / "final_model.joblib"
    joblib.dump(BrokenFusionModel(), path)
    pr = P.Predictor(final_model_path=path, photo_model=None, text_encoder=False)
    r = pr.predict(VALID)
    assert r.model_name.startswith("Endterm base") and r.price_kzt > 0
    assert any("failed on this input" in n for n in r.notes)
