"""Collect the Endterm tables into ENDTERM_REPORT.md.

    .venv/bin/python tools/build_report.py

Reads endterm/results_table.csv, summary.json, segment_profiles.csv and
error_by_segment.csv and nn_gradient_flow.csv written by the Endterm notebook.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "endterm"
res = pd.read_csv(E / "results_table.csv", index_col="model")
summ = json.loads((E / "summary.json").read_text())
seg = pd.read_csv(E / "segment_profiles.csv", index_col=0)
err = pd.read_csv(E / "error_by_segment.csv", index_col=0)
grad = pd.read_csv(E / "nn_gradient_flow.csv", index_col=[0, 1])
mid = res.loc["Midterm linear (log price)"]
b = summ["bootstrap95_mae"]; bm = summ["bootstrap95_mdape"]
params = summ["hgb_params"] if "boosting" in summ["best_model"].lower() else summ.get("rf_params")
goal = "reached" if summ["test_mdape"] < 10 else "not reached"
out = f"""# Endterm report — apartment asking-price prediction (SE-2424, Balgabekov & Kaiyrbekuly)

Full text, code and figures: `Endterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb`. This file collects the tables.

**Chosen by 5-fold CV MAE:** {summ['best_model']} — test MAE **{summ['test_mae_M']:.2f} M KZT** (95 % bootstrap {b[0]:.2f}–{b[1]:.2f}), test MdAPE **{summ['test_mdape']:.2f} %** ({bm[0]:.2f}–{bm[1]:.2f}), R² {summ['test_r2']:.3f}. Midterm linear model on the same {summ['n_test']} test ads: {mid['Test MAE (M KZT)']:.2f} M, {mid['Test MdAPE %']:.2f} %, R² {mid['Test R²']:.3f}. Registration goal (median error < 10 %): {goal}.
Settings: {params}.

## Every model, same folds, same test split
{res.to_markdown()}

## K-Means segments (training rows; price columns computed after clustering)
{seg.to_markdown()}

## Error of the chosen model by band, city, rooms and segment (test rows)
{err.to_markdown()}

## Neural network: gradient flow with depth (Practice 8A measurement, fold 1)
Untrained networks of 64 units per hidden layer, one `backward()` on the inner-training rows; then trained with early stopping on the same fold.
{grad.reset_index().to_markdown(index=False, floatfmt=".4g")}

## Other numbers
- K-Means: k = {summ['segments_k']}; DBSCAN eps = {summ['dbscan_eps']:.2f}, {summ['dbscan_noise_train']} noise ads in training; PCA: {summ['pca_components_90']} components for 90 % of the variance of the 10 numeric fields.
- Over-pricing flag (> 20 % above the estimate): {summ['flagged_overpriced_share']:.0%} of the test ads.
- Features ({len(summ['features'])}): {', '.join(summ['features'])}.
"""
(ROOT / "ENDTERM_REPORT.md").write_text(out, encoding="utf-8"); print("ENDTERM_REPORT.md written,", len(res), "models")
