"""Tests for the Final deliverable builders (scripts/build_final_*.py, scripts/final_build_common.py).

Fixture outputs live in tmp_path; the real outputs/ are only read (never written) by the
"real outputs" tests, which write their deliverables to tmp_path too.
"""
import ast
import json
import sys
from pathlib import Path

import nbformat
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_final_notebook as bnb  # noqa: E402
import build_final_report as brep  # noqa: E402
import build_final_slides as bsl  # noqa: E402
import final_build_common as fc  # noqa: E402

CV_COLS = "variant,cv_mae_M,cv_mae_sd_M,cv_mdape,cv_r2,folds_mae_M,delta_vs_base_M,folds_better,passes_1sd_rule"


@pytest.fixture
def out(tmp_path) -> Path:
    o = tmp_path / "outputs"
    (o / "core").mkdir(parents=True)
    (o / "core" / "base_cv.json").write_text(json.dumps({
        "cv_mae_M": 6.616, "cv_mae_sd_M": 0.3535, "cv_mdape": 8.934, "cv_r2": 0.8569,
        "endterm_reference": {"cv_mae_M": 6.616, "cv_mae_sd_M": 0.3535, "cv_mdape": 8.934, "cv_r2": 0.8569,
                              "test_mae_M": 6.29, "test_mdape": 9.13, "test_r2": 0.855}}))
    (o / "core" / "endterm_repro_test.json").write_text(json.dumps({"test": {"mae_M": 6.29, "rmse_M": 13.9, "mdape": 9.13, "r2": 0.855,
                                                                              "within10": 0.54, "within20": 0.84}}))
    (o / "geo").mkdir()
    (o / "geo" / "cv.csv").write_text("\n".join([CV_COLS,
        "base (Endterm),6.616,0.3535,8.934,0.857,a,0.0,0,False",
        "base + geo (OSM),6.531,0.25,8.8,0.86,a,-0.085,3,False",
        "base + distance to centre only,6.10,0.24,8.6,0.87,a,-0.516,5,True"]))
    (o / "trust").mkdir()
    (o / "trust" / "chosen_settings.json").write_text(json.dumps({
        "intervals": {"chosen_variant": "cqr crossfit", "oof": {"0.9": {"coverage": 0.903}, "0.8": {"coverage": 0.801}}},
        "shap": {"top_features": [{"feature": "area"}, {"feature": "lon"}]}}))
    return o


# --------------------------------------------------------------------------- resolver
def test_values_resolves_json_paths_with_dotted_keys_and_list_indices(out):
    v = fc.Values(out=out)
    assert v.get("trust/chosen_settings:intervals.oof.0.9.coverage") == 0.903
    assert v.get("trust/chosen_settings:shap.top_features.1.feature") == "lon"
    assert v.get("trust/chosen_settings:shap.top_features.5.feature") is fc.MISSING
    assert v.get("nothing/here:x") is fc.MISSING


def test_values_aliases_and_derived_threshold(out):
    v = fc.Values(out=out)
    assert v.get("base.cv_mae_M") == 6.616
    assert v.get("base.threshold_M") == pytest.approx(6.616 - 0.3535)
    assert v.get("endterm.test_mae_M") == 6.29
    assert v.get("final.test_mae_M") is fc.MISSING


def test_values_table_best_match_and_counts(out):
    v = fc.Values(out=out)
    assert v.get("geo/cv@best:variant") == "base + distance to centre only"   # base row excluded
    assert v.get("geo/cv@best:passes_1sd_rule") is True
    assert v.get("geo/cv[geo (OSM)]:cv_mae_M") == 6.531
    assert v.get("geo/cv[no such]:cv_mae_M") is fc.MISSING
    assert v.get("geo/cv@n") == 2 and v.get("geo/cv@n_pass") == 1
    assert v.get("text/cv@best:variant") is fc.MISSING


@pytest.mark.parametrize("value,spec,expected", [
    (6.6159, None, "6.62"), (0.35352, None, "0.354"), (3852, None, "3,852"), (True, None, "yes"),
    (0.9031, ".1%", "90.3%"), (-0.135, "+.3f", "-0.135"), (0.9031, "pct", "90.3"), ("x", ".2f", "x"),
])
def test_fmt_value(value, spec, expected):
    assert fc.fmt_value(value, spec) == expected


def test_fill_keeps_or_marks_missing(out):
    v = fc.Values(out=out)
    s = "MAE {{base.cv_mae_M|.3f}}, final {{final.chosen}}"
    assert fc.fill(s, v) == "MAE 6.616, final {{final.chosen}}"
    assert fc.fill(s, v, missing="pending") == "MAE 6.616, final pending"
    assert fc.placeholders(fc.fill(s, v)) == ["{{final.chosen}}"]


def test_pending_note_and_blocks_summary(out):
    v = fc.Values(out=out)
    assert fc.pending_note(["geo/cv.csv"], v) is None
    note = fc.pending_note(["geo/cv.csv", "fusion/selection.csv"], v)
    assert "fusion/selection.csv" in note and "geo/cv.csv" not in note
    tab = fc.blocks_summary(v)
    assert tab.loc["location (OSM + raw fields)", "variants kept"] == "1/2"
    assert tab.loc["text", "best variant"] == "pending"


# --------------------------------------------------------------------------- notebook
def _sections(nb):
    return [c.source.split("\n")[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## ")]


def test_notebook_builds_valid_with_no_outputs(tmp_path):
    nb, unfilled = bnb.build(fc.Values(out=tmp_path / "empty"))
    nbformat.validate(nb)
    heads = _sections(nb)
    assert [h.split(".")[0] for h in heads] == [f"## {i}" for i in range(1, 12)]
    assert any("**Pending.**" in c.source for c in nb.cells if c.cell_type == "markdown")
    assert "{{base.cv_mae_M|.3f}}" in unfilled
    for c in nb.cells:
        if c.cell_type == "code":
            ast.parse(c.source)          # every code cell is valid Python
            assert not c.outputs         # built, not executed


def test_notebook_fills_from_fixture_outputs(out):
    nb, unfilled = bnb.build(fc.Values(out=out))
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "**6.616 ± 0.353 M KZT**" in text
    assert "**base + distance to centre only**" in text
    assert not any(u.startswith("{{base.") for u in unfilled)


def test_notebook_main_writes_valid_file(tmp_path):
    path = tmp_path / "nb.ipynb"
    assert bnb.main(["--out", str(path)]) == 0
    nbformat.validate(nbformat.read(path, as_version=4))


def test_notebook_helpers_degrade_gracefully(out, monkeypatch, tmp_path):
    ns: dict = {}
    monkeypatch.chdir(SCRIPTS.parent)
    exec(compile(bnb.NB_SETUP, "setup", "exec"), ns)
    ns["OUT"] = out                                  # point the helpers at the fixture outputs
    assert ns["read_csv"]("missing/table.csv") is None
    assert ns["read_json"]("missing/x.json") is None
    assert ns["show_figure"]("missing/fig.png") is False
    view = ns["cv_view"](ns["read_csv"]("geo/cv.csv"))
    assert view.index[0] == "base + distance to centre only" and view.iloc[0]["kept (1-SD rule)"] == "yes"
    assert ns["THRESHOLD"] == pytest.approx(6.616 - 0.3535, abs=1e-3)   # from the real outputs/core


# --------------------------------------------------------------------------- report
def test_report_with_no_outputs_is_all_pending(tmp_path):
    text, unfilled, pending = brep.build(fc.Values(out=tmp_path / "empty"))
    assert text.startswith("# Final report")
    assert "_Pending: `outputs/geo/cv.csv`" in text
    assert "geo/cv.csv" in pending and unfilled


def test_report_tables_from_fixture(out):
    text, _, pending = brep.build(fc.Values(out=out))
    assert "base + distance to centre only" in text and "6.100 ± 0.240" in text
    assert "geo/cv.csv" not in pending and "core/base_cv.json" not in pending


def test_report_main_strict_fails_while_pending(tmp_path):
    path = tmp_path / "r.md"
    assert brep.main(["--out", str(path)]) == 0
    _, unfilled, pending = brep.build()
    assert brep.main(["--out", str(path), "--strict"]) == (1 if (unfilled or pending) else 0)
    assert path.read_text(encoding="utf-8").startswith("# Final report")


# --------------------------------------------------------------------------- slides
def test_slides_build_with_no_outputs(tmp_path):
    deck = bsl.build(fc.Values(out=tmp_path / "empty"))
    assert deck.n == 15 and len(deck.prs.slides) == 15
    assert deck.missing_figures
    texts = " ".join(sh.text_frame.text for s in deck.prs.slides for sh in s.shapes if sh.has_text_frame)
    assert "{{" not in texts and "pending" in texts


def test_slides_use_existing_figure(out):
    from PIL import Image
    Image.new("RGB", (800, 400), "white").save(out / "geo" / "residual_map.png")
    deck = bsl.build(fc.Values(out=out))
    slide4 = deck.prs.slides[3]
    assert any(sh.shape_type == 13 for sh in slide4.shapes)   # 13 = PICTURE
    assert "geo/residual_map.png" not in deck.missing_figures


def test_slides_main_writes_file(tmp_path):
    path = tmp_path / "deck.pptx"
    assert bsl.main(["--out", str(path)]) == 0
    from pptx import Presentation
    assert len(Presentation(str(path)).slides) == 15
