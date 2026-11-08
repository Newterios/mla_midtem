"""Trust layer on the Endterm base model: intervals, over-pricing flags, SHAP, comparables.

Run from the repo root:  python scripts/run_trust.py [--stage all|intervals|flags|shap|comparables]
Training ads only (core.load_train(), the Endterm 5 folds). The 963 test ads are never touched.
Outputs: outputs/trust/ (tables, figures, chosen_settings.json); caches: cache/trust_*.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")          # 5 parallel folds x 2 threads on the M4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from krisha_ml import core, trust  # noqa: E402

OUT = core.OUTPUT_DIR / "trust"
OUT.mkdir(parents=True, exist_ok=True)
core.CACHE_DIR.mkdir(exist_ok=True)
LEVELS = trust.LEVELS
BAND_ORDER = ["<25 M", "25-40 M", "40-60 M", "60-100 M", ">=100 M"]
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================================================================== 1. intervals
def _variants() -> list[dict]:
    v = [{"method": "resid", "calibration": "crossfit", "params": "-"}]
    for p in trust.QUANTILE_PARAMS:
        v += [{"method": "cqr", "calibration": c, "params": p} for c in ("crossfit", "split")]
    return v


def _vname(v: dict, mondrian: str | None) -> str:
    base = "abs-residual (base HGB)" if v["method"] == "resid" else f"CQR {v['params']} {v['calibration']}"
    tag = {None: "global", "city": "Mondrian city", trust.ESTIMATE_BAND: "Mondrian estimate band"}[mondrian]
    return f"{base} | {tag}"


def _mondrians(v: dict) -> tuple:
    return ("city", None, trust.ESTIMATE_BAND) if v["method"] == "resid" else ("city", None)


def _resid_band_fold(k: int, tr: np.ndarray, va: np.ndarray, train: core.Dataset, resid_in: dict) -> dict:
    """Abs-residual intervals with Mondrian groups = price band of the base estimate (added after the first
    run; the other variants are cached in trust_interval_oof.npz)."""
    X_tr, X_va = train.X.iloc[tr].reset_index(drop=True), train.X.iloc[va].reset_index(drop=True)
    point = core.make_base_model().fit(X_tr, train.y[tr])
    ci = trust.ConformalIntervals("resid", LEVELS, trust.ESTIMATE_BAND).fit(X_tr, train.y[tr], point, resid_in["inner"][k])
    name = _vname({"method": "resid"}, trust.ESTIMATE_BAND)
    return {(name, lv): (ci.predict_log(X_va, lv), ci.qhat_[lv]) for lv in LEVELS}


def _outer_fold(k: int, tr: np.ndarray, va: np.ndarray, train: core.Dataset, resid_in: dict) -> dict:
    """All interval variants for outer fold k: fit on rows tr, predict rows va (log bounds)."""
    X_tr, X_va, y_tr = train.X.iloc[tr].reset_index(drop=True), train.X.iloc[va].reset_index(drop=True), train.y[tr]
    out = {}
    for v in _variants():
        if v["method"] == "resid":
            point = core.make_base_model().fit(X_tr, y_tr)
            assert np.allclose(np.log(point.predict(X_va)), resid_in["outer"][va]), "base refit differs from cache"
            ci = trust.ConformalIntervals("resid", LEVELS, "city").fit(X_tr, y_tr, point, resid_in["inner"][k])
        else:
            ci = trust.ConformalIntervals("cqr", LEVELS, "city", v["calibration"], trust.QUANTILE_PARAMS[v["params"]]).fit(X_tr, y_tr)
        for mond in ("city", None):
            c = ci if mond == "city" else ci.with_mondrian(None)
            for lv in LEVELS:
                out[(_vname(v, mond), lv)] = (c.predict_log(X_va, lv), c.qhat_[lv])
    return out


def stage_intervals(train: core.Dataset) -> dict:
    cache = core.CACHE_DIR / "trust_interval_oof.npz"
    folds = core.cv_folds(len(train))
    names = [_vname(v, m) for v in _variants() for m in ("city", None)]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        bounds = {(n, lv): (z[f"{i}_{lv}_lo"], z[f"{i}_{lv}_hi"]) for i, n in enumerate(names) for lv in LEVELS}
        qhat = json.loads(str(z["qhat"]))
        log("intervals: loaded cached OOF bounds")
    else:
        t0 = time.time()
        resid_in = core.base_residual_inputs(train)
        res = Parallel(n_jobs=5)(delayed(_outer_fold)(k, tr, va, train, resid_in) for k, (tr, va) in enumerate(folds))
        bounds, qhat = {}, {}
        for key in res[0]:
            lo, hi = np.empty(len(train)), np.empty(len(train))
            for (_, va), r in zip(folds, res):
                lo[va], hi[va] = r[key][0]
            bounds[key] = (lo, hi)
            qhat[f"{key[0]} @ {key[1]}"] = [r[key][1] for r in res]
        arrays = {f"{i}_{lv}_{s}": bounds[(n, lv)][j] for i, n in enumerate(names) for lv in LEVELS for j, s in enumerate(("lo", "hi"))}
        np.savez(cache, qhat=json.dumps(qhat), **arrays)
        log(f"intervals: {len(names)} variants x {len(LEVELS)} levels in {time.time() - t0:.0f} s")
    band_cache = core.CACHE_DIR / "trust_interval_oof_band.npz"
    band_name = _vname({"method": "resid"}, trust.ESTIMATE_BAND)
    if band_cache.exists():
        z = np.load(band_cache, allow_pickle=False)
        bounds.update({(band_name, lv): (z[f"{lv}_lo"], z[f"{lv}_hi"]) for lv in LEVELS})
        qhat.update(json.loads(str(z["qhat"])))
    else:
        resid_in = core.base_residual_inputs(train)
        res = Parallel(n_jobs=5)(delayed(_resid_band_fold)(k, tr, va, train, resid_in) for k, (tr, va) in enumerate(folds))
        q_new = {}
        for lv in LEVELS:
            lo, hi = np.empty(len(train)), np.empty(len(train))
            for (_, va), r in zip(folds, res):
                lo[va], hi[va] = r[(band_name, lv)][0]
            bounds[(band_name, lv)] = (lo, hi)
            q_new[f"{band_name} @ {lv}"] = [r[(band_name, lv)][1] for r in res]
        qhat.update(q_new)
        np.savez(band_cache, qhat=json.dumps(q_new), **{f"{lv}_{s}": bounds[(band_name, lv)][j] for lv in LEVELS for j, s in enumerate(("lo", "hi"))})
        log("intervals: computed the estimate-band Mondrian variant")
    names = names + [band_name]

    city = train.df["city"].to_numpy()
    band = trust.price_band(train.y).astype(str)
    pband = trust.price_band(core.base_cv().oof_pred).astype(str)   # band of the base estimate: known at prediction time
    fold_of = np.empty(len(train), dtype=int)
    for k, (_, va) in enumerate(folds):
        fold_of[va] = k + 1
    rows, summary = [], []
    for n in names:
        for lv in LEVELS:
            lo, hi = np.exp(bounds[(n, lv)][0]), np.exp(bounds[(n, lv)][1])
            tab = trust.coverage_table(train.y, lo, hi, {"city": city, "price band": band, "estimate band": pband, "fold": fold_of})
            tab.insert(0, "level", lv); tab.insert(0, "variant", n)
            rows.append(tab)
            cities = tab[tab["group_type"] == "city"]
            folds_cov = tab.loc[tab["group_type"] == "fold", "coverage"]
            ok_city = all(trust.coverage_ok(r.coverage, r.n, lv) for r in cities.itertuples())
            ok_est = all(trust.coverage_ok(r.coverage, r.n, lv)
                         for r in tab[tab["group_type"] == "estimate band"].itertuples())
            summary.append({"variant": n, "level": lv, "coverage": tab.iloc[0]["coverage"],
                            "coverage_fold_sd": folds_cov.std(ddof=0),
                            **{f"coverage_{r.group}": r.coverage for r in cities.itertuples()},
                            "min_band_coverage": tab.loc[tab["group_type"] == "price band", "coverage"].min(),
                            "min_estimate_band_coverage": tab.loc[tab["group_type"] == "estimate band", "coverage"].min(),
                            "mean_width_M": tab.iloc[0]["mean_width_M"], "median_width_M": tab.iloc[0]["median_width_M"],
                            "median_rel_width_pct": tab.iloc[0]["median_rel_width_pct"],
                            "share_above": tab.iloc[0]["share_above"], "share_below": tab.iloc[0]["share_below"],
                            "valid_per_city": ok_city, "valid_per_estimate_band": ok_est})
    cov = pd.concat(rows, ignore_index=True)
    summ = pd.DataFrame(summary)
    # selection: valid per-city coverage at BOTH levels, then narrowest mean width at 90 %
    # selection (train-only): coverage not significantly below nominal in every city AND every band of the
    # point estimate (both known when a new ad is priced), at 80 % and 90 %; then the narrowest mean 90 % width.
    w90 = summ[summ["level"] == 0.9].set_index("variant")["mean_width_M"]
    valid_city = summ.groupby("variant")["valid_per_city"].all()
    valid = (summ.assign(v=summ["valid_per_city"] & summ["valid_per_estimate_band"]).groupby("variant")["v"].all())
    summ["valid_both_levels"] = summ["variant"].map(valid)
    chosen_city_only = w90[valid_city[w90.index]].idxmin()
    chosen = w90[valid[w90.index]].idxmin() if valid.any() else chosen_city_only
    summ["chosen"] = summ["variant"] == chosen
    log(f"city-only rule would choose: {chosen_city_only}")
    cov.round(4).to_csv(OUT / "coverage_table.csv", index=False)
    summ.round(4).sort_values(["level", "mean_width_M"]).to_csv(OUT / "interval_methods.csv", index=False)
    log(f"chosen interval method: {chosen}")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summ[["variant", "level", "coverage", "coverage_Almaty", "coverage_Astana", "min_band_coverage",
                    "min_estimate_band_coverage", "mean_width_M", "median_rel_width_pct", "valid_per_city", "valid_per_estimate_band"]].round(3).sort_values(["level", "mean_width_M"]).to_string(index=False))
    _plot_intervals(cov, summ, chosen)
    return {"bounds": bounds, "chosen": chosen, "chosen_city_only": chosen_city_only, "summary": summ, "coverage": cov, "qhat": qhat}


def _plot_intervals(cov: pd.DataFrame, summ: pd.DataFrame, chosen: str) -> None:
    # (a) width vs coverage of all variants at 90 %
    s = summ[summ["level"] == 0.9].sort_values("mean_width_M")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = ["#1f77b4" if v else "#bbbbbb" for v in s["valid_both_levels"]]
    ax.barh(s["variant"], s["mean_width_M"], color=colors)
    for y, (w, c) in enumerate(zip(s["mean_width_M"], s["coverage"])):
        ax.text(w + 0.3, y, f"{w:.1f} M  (cov {100 * c:.1f} %)", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean width of the 90 % interval, M KZT (out-of-fold, 3,852 training ads)")
    ax.set_title("90 % interval methods (grey = coverage not valid in a city or estimate band)", fontsize=10)
    ax.set_xlim(0, s["mean_width_M"].max() * 1.35)
    fig.savefig(OUT / "interval_methods_width.png"); plt.close(fig)
    # (b) chosen method: coverage per city and per price band at both levels
    c = cov[(cov["variant"] == chosen) & cov["group_type"].isin(["overall", "city", "estimate band"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    order = ["all", "Almaty", "Astana"] + [b for b in BAND_ORDER]
    groups = [g for g in order if g in set(c[c["level"] == 0.9]["group"])]
    x = np.arange(len(groups))
    for i, lv in enumerate(LEVELS):
        vals = c[c["level"] == lv].set_index("group").loc[groups, "coverage"]
        ax.bar(x + (i - 0.5) * 0.38, 100 * vals, width=0.38, label=f"{int(lv * 100)} % interval", color=["#9ecae1", "#3182bd"][i])
        ax.axhline(100 * lv, color=["#9ecae1", "#3182bd"][i], ls="--", lw=1)
    ax.set_xticks(x, [g if g in ("all", "Almaty", "Astana") else f"estimate {g}" for g in groups], rotation=20)
    ax.set_ylim(60, 100); ax.set_ylabel("coverage, % of ads inside the interval")
    ax.set_title(f"Out-of-fold coverage, chosen method: {chosen}", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(OUT / "coverage_chosen.png"); plt.close(fig)


def photo_duplicate_ids(train: core.Dataset) -> set[int]:
    """Training ads sharing an exact photo pHash with another training ad (relisted flats / agency stock
    photos). Empty if the photo hash cache is missing."""
    path = core.CACHE_DIR / "photo_hashes.csv"
    if not path.exists():
        return set()
    h = pd.read_csv(path, dtype={"ad_id": "int64", "phash": str})
    h = h[h["ad_id"].isin(train.ids)]
    n_ads = h.groupby("phash")["ad_id"].nunique()
    return set(h.loc[h["phash"].isin(n_ads[n_ads > 1].index), "ad_id"])


def duplicate_sensitivity(train: core.Dataset, iv: dict) -> dict:
    """Coverage/width of the chosen method with and without ads that have a photo duplicate."""
    dup = np.isin(train.ids, list(photo_duplicate_ids(train)))
    out = {"n_ads_with_photo_duplicate": int(dup.sum())}
    for lv in LEVELS:
        lo, hi = (np.exp(a) for a in iv["bounds"][(iv["chosen"], lv)])
        cov = (train.y >= lo) & (train.y <= hi)
        out[str(lv)] = {"coverage_all": round(float(cov.mean()), 4), "coverage_without_duplicates": round(float(cov[~dup].mean()), 4),
                        "coverage_duplicates_only": round(float(cov[dup].mean()), 4) if dup.any() else None}
    log(f"duplicate sensitivity: {out}")
    return out


# =========================================================================== 2. flags
def stage_flags(train: core.Dataset, iv: dict) -> dict:
    base = core.base_cv()
    lo, hi = (np.exp(a) for a in iv["bounds"][(iv["chosen"], 0.9)])
    flag_iv = trust.price_flags(train.y, lo, hi)
    flag_et = trust.endterm_flags(train.y, base.oof_pred)
    city, band = train.df["city"].to_numpy(), trust.price_band(train.y).astype(str)
    eband = trust.price_band(base.oof_pred).astype(str)   # band of the estimate: what the interval width depends on
    frame = pd.DataFrame({"id": train.ids, "city": city, "band": band, "estimate_band": eband, "price": train.y, "base_pred": base.oof_pred,
                          "lo80": np.exp(iv["bounds"][(iv["chosen"], 0.8)][0]), "hi80": np.exp(iv["bounds"][(iv["chosen"], 0.8)][1]),
                          "lo90": lo, "hi90": hi, "flag_interval90": flag_iv, "flag_endterm20": flag_et})
    frame["point_inside_90"] = (frame["base_pred"] >= lo) & (frame["base_pred"] <= hi)
    frame.to_csv(OUT / "oof_intervals_flags.csv", index=False)

    def shares(f: pd.DataFrame) -> dict:
        return {"n": len(f), "over_interval90": (f["flag_interval90"] == "over").mean(),
                "under_interval90": (f["flag_interval90"] == "under").mean(),
                "over_endterm20": (f["flag_endterm20"] == "over").mean(),
                "under_endterm20": (f["flag_endterm20"] == "under").mean(),
                "over_both": ((f["flag_interval90"] == "over") & (f["flag_endterm20"] == "over")).mean()}
    rows = [{"group_type": "overall", "group": "all", **shares(frame)}]
    rows += [{"group_type": "city", "group": g, **shares(f)} for g, f in frame.groupby("city", sort=True)]
    for gt, col in (("price band", "band"), ("estimate band", "estimate_band")):
        rows += [{"group_type": gt, "group": b, **shares(frame[frame[col] == b])} for b in BAND_ORDER if (frame[col] == b).any()]
    tab = pd.DataFrame(rows)
    tab.round(4).to_csv(OUT / "overpricing_shares.csv", index=False)
    cross = pd.crosstab(frame["flag_endterm20"], frame["flag_interval90"], margins=True)
    cross.to_csv(OUT / "overpricing_crosstab.csv")
    # how far above the estimate the price has to be to get the interval flag
    over = frame[frame["flag_interval90"] == "over"]
    hi_ratio = frame["hi90"] / frame["base_pred"] - 1
    res = {"share_over_interval90": float(tab.iloc[0]["over_interval90"]), "share_under_interval90": float(tab.iloc[0]["under_interval90"]),
           "share_over_endterm20": float(tab.iloc[0]["over_endterm20"]), "share_under_endterm20": float(tab.iloc[0]["under_endterm20"]),
           "share_over_both": float(tab.iloc[0]["over_both"]),
           "jaccard_over": float(((frame.flag_interval90 == "over") & (frame.flag_endterm20 == "over")).sum()
                                 / ((frame.flag_interval90 == "over") | (frame.flag_endterm20 == "over")).sum()),
           "upper90_above_estimate_pct_median": float(100 * hi_ratio.median()),
           "upper90_above_estimate_pct_p10_p90": [float(100 * hi_ratio.quantile(0.1)), float(100 * hi_ratio.quantile(0.9))],
           "upper90_above_estimate_pct_median_by_estimate_band": {b: round(float(100 * hi_ratio[frame["estimate_band"] == b].median()), 2) for b in BAND_ORDER},
           "over_share_by_estimate_band": {b: {"interval90": round(float((frame.loc[frame["estimate_band"] == b, "flag_interval90"] == "over").mean()), 4),
                                               "endterm20": round(float((frame.loc[frame["estimate_band"] == b, "flag_endterm20"] == "over").mean()), 4)} for b in BAND_ORDER},
           "base_point_inside_90_share": float(frame["point_inside_90"].mean()),
           "n_over_interval90": int(len(over))}
    log(f"flags: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})}")
    print(tab.round(3).to_string(index=False)); print(cross)
    # figure: required mark-up for the flag vs. the fixed +20 % rule, per band of the estimate (the interval
    # width is set by the estimate band, so this is the grouping the flag threshold actually follows)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    data = [100 * hi_ratio[frame["estimate_band"] == b] for b in BAND_ORDER]
    ax.boxplot(data, tick_labels=BAND_ORDER, showfliers=False)
    ax.axhline(20, color="#d62728", ls="--", lw=1.2, label="Endterm rule: +20 %")
    ax.set_xlabel("band of the model estimate (out-of-fold)"); ax.set_ylabel("upper 90 % bound above the estimate, %")
    ax.set_title("How much above the estimate an ad must be to get the interval over-pricing flag", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(OUT / "flag_threshold_by_band.png"); plt.close(fig)
    return res


# =========================================================================== 3. SHAP
def stage_shap(train: core.Dataset) -> dict:
    import shap
    t0 = time.time()
    model = core.make_base_model().fit(train.X, train.y)
    res = trust.shap_explain(model, train.X)
    imp = res.values.abs().mean().sort_values(ascending=False)
    # ordinal twin: same params, categoricals as single ordinal columns
    ord_cv = core.cv_evaluate("Endterm HGB, ordinal-encoded categoricals",
                              model_factory=lambda n, c: trust.make_ordinal_model(), n_jobs=5)
    base = core.base_cv()
    ord_model = trust.make_ordinal_model().fit(train.X, train.y)
    ord_res = trust.shap_explain(ord_model, train.X)
    ord_imp = ord_res.values.abs().mean()
    table = pd.DataFrame({"mean_abs_shap_log": imp, "mean_abs_shap_log_ordinal_twin": ord_imp.reindex(imp.index)})
    table["rank"] = np.arange(1, len(table) + 1)
    table["rank_ordinal_twin"] = table["mean_abs_shap_log_ordinal_twin"].rank(ascending=False).astype(int)
    table.round(5).to_csv(OUT / "shap_global_importance.csv", index_label="feature")
    spearman = float(table[["mean_abs_shap_log", "mean_abs_shap_log_ordinal_twin"]].corr("spearman").iloc[0, 1])

    # global beeswarm (numeric features coloured by value; categoricals and groups grey)
    cols = list(imp.index)
    data = pd.DataFrame({c: pd.to_numeric(train.X[c], errors="coerce") if c in core.NUM + core.BOOL else np.nan for c in cols})
    data["area"] = train.X["area"]
    expl = shap.Explanation(values=res.values[cols].to_numpy(), base_values=np.full(len(train), res.base_value),
                            data=data[cols].to_numpy(dtype=float), feature_names=cols)
    plt.figure()
    shap.plots.beeswarm(expl, max_display=15, show=False, plot_size=(8.5, 6))
    plt.title("SHAP of the Endterm base model (log-price scale), 3,852 training ads", fontsize=10)
    plt.xlabel("SHAP value: contribution to log(price)  (+0.1 = about +10.5 % price)")
    plt.savefig(OUT / "shap_beeswarm.png"); plt.close("all")

    # per-ad waterfall: an ad flagged as over-priced, explained by the fold model that did not see it
    flags = pd.read_csv(OUT / "oof_intervals_flags.csv")
    over = flags[flags["flag_interval90"] == "over"].assign(r=lambda d: d["price"] / d["base_pred"])
    ex_id = int(over.sort_values("r").iloc[len(over) // 2]["id"])     # median over-priced ad, not an extreme one
    pos = int(np.flatnonzero(train.ids == ex_id)[0])
    tr = next(tr for tr, va in core.cv_folds(len(train)) if pos in va)
    fold_model = core.make_base_model().fit(train.X.iloc[tr], train.y[tr])
    one = trust.shap_explain(fold_model, train.X.iloc[[pos]])
    tab = one.explain_one(0)
    tab.to_csv(OUT / f"shap_example_{ex_id}.csv", index=False)
    e1 = shap.Explanation(values=one.values.iloc[0].to_numpy(), base_values=one.base_value,
                          data=np.array([trust._group_value(one.data.iloc[0], g) for g in one.values.columns], dtype=object),
                          feature_names=list(one.values.columns))
    plt.figure()
    shap.plots.waterfall(e1, max_display=12, show=False)
    pred = np.exp(one.log_pred[0]) / 1e6
    row = flags.set_index("id").loc[ex_id]
    plt.title(f"Ad {ex_id}: estimate {pred:.1f} M KZT (log {one.log_pred[0]:.2f}), asking {row['price'] / 1e6:.1f} M, "
              f"90 % interval {row['lo90'] / 1e6:.1f}-{row['hi90'] / 1e6:.1f} M -> over-priced", fontsize=9)
    plt.savefig(OUT / "shap_waterfall_example.png"); plt.close("all")
    out = {"top_features": [{"feature": f, "mean_abs_shap_log": round(float(v), 4)} for f, v in imp.head(8).items()],
           "ordinal_twin_cv": ord_cv.row(base), "importance_spearman_onehot_vs_ordinal": round(spearman, 3),
           "example_ad": ex_id, "seconds": round(time.time() - t0, 1)}
    log(f"shap: {json.dumps(out)[:900]}")
    return out


# =========================================================================== 4. comparables
def stage_comparables(train: core.Dataset, example_id: int | None = None) -> dict:
    folds = core.cv_folds(len(train))
    frame = trust.comparables_frame(train.df)
    parts = []
    for tr, va in folds:
        comp = trust.Comparables().fit(frame.iloc[tr])
        q = frame.iloc[va]
        s = comp.summary(comp.query(q, exclude_self=False), q.set_index("id")["area"])
        s["threshold"] = s["query_id"].map(dict(zip(q["id"], q["city"]))).map(comp.threshold_)
        parts.append(s)
    oof = pd.concat(parts).set_index("query_id").reindex(train.ids).reset_index().rename(columns={"index": "id", "query_id": "id"})
    flags = pd.read_csv(OUT / "oof_intervals_flags.csv").set_index("id").reindex(train.ids)
    base = core.base_cv()
    ape = np.abs(base.oof_pred - train.y) / train.y
    oof["base_abs_err_M"] = np.abs(base.oof_pred - train.y) / 1e6
    oof["base_ape"] = ape
    oof["rel_width90"] = ((flags["hi90"] - flags["lo90"]) / flags["base_pred"]).to_numpy()
    oof.round(5).to_csv(OUT / "comparables_oof.csv", index=False)
    warned = oof["few_comparables"].astype(bool).to_numpy()
    cmp = pd.DataFrame({g: {"n": int(m.sum()), "share": float(m.mean()), "base_mae_M": float(oof.loc[m, "base_abs_err_M"].mean()),
                            "base_mdape_pct": float(100 * np.median(ape[m])),
                            "median_rel_width90_pct": float(100 * oof.loc[m, "rel_width90"].median()),
                            "over_or_under_flag_share": float((flags["flag_interval90"].to_numpy()[m] != "in range").mean())}
                        for g, m in (("few comparables", warned), ("enough comparables", ~warned))}).T
    cmp.round(4).to_csv(OUT / "comparables_warning_check.csv")
    comp_est = core.metrics(train.y, oof["comp_estimate"].to_numpy())
    # deployable object on all training ads + one example
    comp = trust.Comparables().fit(frame)
    ex_id = int(example_id if example_id is not None else train.ids[0])
    example = comp.query(frame[frame["id"] == ex_id])
    example.round(3).to_csv(OUT / "comparables_example.csv", index=False)
    # figure: distribution of the 5th-neighbour distance and the warning thresholds
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, city in zip(axes, sorted(comp.threshold_)):
        d = oof.loc[train.df["city"].to_numpy() == city, "dist_k"]
        ax.hist(d, bins=50, color="#6baed6")
        ax.axvline(comp.threshold_[city], color="#d62728", ls="--", label=f"warning threshold {comp.threshold_[city]:.2f}")
        ax.set_title(city, fontsize=10); ax.set_xlabel("distance to the 5th comparable (standardised units)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("training ads (out-of-fold query)")
    fig.savefig(OUT / "comparables_distance.png"); plt.close(fig)
    out = {"thresholds": {c: round(v, 4) for c, v in comp.threshold_.items()}, "oof_warning_share": float(warned.mean()),
           "warning_check": cmp.round(4).to_dict(orient="index"),
           "comparables_estimate_cv": {k: round(v, 4) for k, v in comp_est.items()}, "example_ad": ex_id}
    log(f"comparables: {json.dumps(out)}")
    return out


# =========================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "intervals", "flags", "shap", "comparables"])
    stage = ap.parse_args().stage
    train = core.load_train()
    settings_path = OUT / "chosen_settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    iv = stage_intervals(train)
    if stage in ("all", "intervals"):
        ch = iv["summary"][iv["summary"]["chosen"]].set_index("level")
        v, mond = {_vname(v, m): (v, m) for v in _variants() for m in _mondrians(v)}[iv["chosen"]]
        final = trust.ConformalIntervals(v["method"], LEVELS, mond, v["calibration"], trust.QUANTILE_PARAMS.get(v["params"]))
        if v["method"] == "resid":
            final.fit(train.X, train.y, core.make_base_model().fit(train.X, train.y), core.oof_log_pred(train.X, train.y))
        else:
            final.fit(train.X, train.y)
        settings["intervals"] = {"chosen_variant": iv["chosen"], **final.settings(),
                                 "selection_rule": "OOF coverage >= level - 2 binomial SE in every city and every band of the point "
                                                   "estimate, at 80 % and 90 %; then the narrowest mean 90 % width "
                                                   "(train rows, Endterm folds)",
                                 "chosen_by_city_only_rule": iv["chosen_city_only"],
                                 "duplicate_sensitivity": duplicate_sensitivity(train, iv),
                                 "oof": {str(lv): {k: round(float(x), 4) for k, x in ch.loc[lv].items() if k not in ("variant", "chosen")} for lv in LEVELS}}
    if stage in ("all", "flags"):
        settings["flags"] = {"rule": "over if asking price > upper bound of the chosen 90 % interval, under if < lower bound",
                             **stage_flags(train, iv)}
    if stage in ("all", "shap"):
        settings["shap"] = {"explainer": "shap.TreeExplainer on the fitted HGB inside the base pipeline (exact, interventional=False); "
                                         "one-hot dummies and missing indicators summed to the source feature",
                            "groups": {k: list(v) for k, v in trust.SHAP_GROUPS.items()}, **stage_shap(train)}
        settings_path.write_text(json.dumps(settings, indent=2, default=str))
    if stage in ("all", "comparables"):
        c = trust.Comparables()
        settings["comparables"] = {"k": c.k, "features": list(trust.COMP_FEATURES), "same_city_only": True,
                                   "warning": f"5th-neighbour distance > {c.warn_quantile:.0%} quantile of the leave-one-out "
                                              "5th-neighbour distance of the city's training ads",
                                   **stage_comparables(train, settings.get("shap", {}).get("example_ad"))}
    settings_path.write_text(json.dumps(settings, indent=2, default=str))
    log(f"wrote {settings_path}")


if __name__ == "__main__":
    main()
