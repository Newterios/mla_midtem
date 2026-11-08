"""Track "geo": OpenStreetMap location features + unused raw ad fields, judged by the 1-SD rule.

Run from the repo root:  .venv/bin/python scripts/run_geo.py [--plots-only | --importance-only]
(on a busy machine run --importance-only with OMP_NUM_THREADS=1: 5 single-threaded fold workers)
Training rows only (core.load_train); the 963 test ads are never touched.

Outputs (outputs/geo/):
  features_geo.csv        id-keyed OSM features for all 4,815 clean ads
  features_rawfields.csv  id-keyed raw-field features for all 4,815 clean ads
  osm_poi_summary.csv     objects/points per city and OSM layer
  cv.csv, cv.json         base, base+geo, base+rawfields, base+both (+ a centre-distance-only diagnostic)
  oof_predictions.csv     OOF predictions of every variant (train ads)
  permutation_importance.csv, permutation_importance.png
  residual_map.png        base OOF residual maps + residual vs distance to centre
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from krisha_ml import core, geo

OUT = core.OUTPUT_DIR / "geo"
N_JOBS = 2
N_REPEATS = 5


def build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = core.load_clean()
    geo.fetch_all()
    geo.poi_summary().to_csv(OUT / "osm_poi_summary.csv", index=False)
    g = geo.geo_features(clean)
    rawf = geo.rawfield_features()
    rawf = rawf[rawf[core.ID].isin(clean[core.ID])].reset_index(drop=True)
    assert len(g) == len(clean) == g[core.ID].nunique() and len(rawf) == len(clean)
    g.to_csv(OUT / "features_geo.csv", index=False)   # full precision: a rounded table shifts HGB bins and the CV numbers
    rawf.to_csv(OUT / "features_rawfields.csv", index=False)
    return g, rawf


def run_cv(g: pd.DataFrame, rawf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = core.base_cv()
    both = g.merge(rawf, on=core.ID, validate="1:1")
    variants = {
        "base + geo (OSM)": g,
        "base + raw fields": rawf,
        "base + geo + raw fields": both,
        # diagnostic only (one column, chosen a priori as the classic location signal)
        "base + distance to centre only": g[[core.ID, "geo_dist_centre_km"]],
    }
    results = {"base (Endterm)": base}
    for name, extra in variants.items():
        results[name] = core.cv_evaluate(name, extra=extra, n_jobs=N_JOBS)
        print(results[name], flush=True)
    rows = [base.row(base)] + [r.row(base) for n, r in results.items() if n != "base (Endterm)"]
    rows[0]["variant"] = "base (Endterm)"
    table = pd.DataFrame(rows)
    table["n_extra_cols"] = [0] + [v.shape[1] - 1 for v in variants.values()]
    table["rule_threshold_M"] = round(base.mean - base.sd, 4)
    table.assign(folds_mae_M=table["folds_mae_M"].map(lambda v: " / ".join(f"{x:.3f}" for x in v))) \
         .to_csv(OUT / "cv.csv", index=False)
    (OUT / "cv.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=bool))
    train = core.load_train()
    pd.DataFrame({"id": train.ids, "price": train.y,
                  **{n: r.oof_pred for n, r in results.items()}}).to_csv(OUT / "oof_predictions.csv", index=False)
    return table, results


# --------------------------------------------------------------------------- permutation importance
def _fold_importance(train, tr, va, E, groups, seed):
    num, cat = core.split_columns(E)
    Ep = core._prepare_extra(E, num, cat)
    X = pd.concat([train.X, Ep], axis=1)
    model = core.make_base_model(num, cat).fit(X.iloc[tr], train.y[tr])
    X_va, y_va = X.iloc[va].reset_index(drop=True), train.y[va]
    base_mae = np.mean(np.abs(model.predict(X_va) - y_va)) / 1e6
    rng = np.random.default_rng(seed)
    out = []
    for gname, cols in groups.items():
        for rep in range(N_REPEATS):
            Xp = X_va.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]      # permute the group jointly (keeps within-group structure)
            out.append({"feature": gname, "repeat": rep,
                        "mae_increase_M": np.mean(np.abs(model.predict(Xp) - y_va)) / 1e6 - base_mae})
    return out


def permutation_importance(g: pd.DataFrame, rawf: pd.DataFrame, n_jobs: int = N_JOBS) -> pd.DataFrame:
    """Per-fold permutation importance on the validation fold (train ads only), model = base+geo+raw."""
    train = core.load_train()
    E = core.join_features(train.ids, g.merge(rawf, on=core.ID))
    geo_cols = [c for c in E.columns if c.startswith("geo_")]
    raw_cols = [c for c in E.columns if c.startswith("raw_")]
    groups = {c: [c] for c in geo_cols + raw_cols}
    groups.update({"[group] all geo": geo_cols, "[group] all raw fields": raw_cols,
                   "[ref] lat+lon": ["lat", "lon"], "[ref] district": ["district"], "[ref] area": ["area", "log_area"]})
    parts = Parallel(n_jobs=n_jobs)(delayed(_fold_importance)(train, tr, va, E, groups, core.SEED + k)
                                    for k, (tr, va) in enumerate(core.cv_folds(len(train))))
    df = pd.DataFrame([dict(r, fold=k) for k, p in enumerate(parts) for r in p])
    per_fold = df.groupby(["feature", "fold"])["mae_increase_M"].mean().unstack()
    imp = pd.DataFrame({"mae_increase_M": per_fold.mean(axis=1), "sd_over_folds_M": per_fold.std(axis=1, ddof=0),
                        "folds_positive": (per_fold > 0).sum(axis=1)}).sort_values("mae_increase_M", ascending=False)
    imp.round(4).to_csv(OUT / "permutation_importance.csv")
    return imp


def plot_importance(imp: pd.DataFrame) -> None:
    top = imp.head(25).iloc[::-1]
    colors = ["#888888" if f.startswith("[ref]") else "#d95f02" if "raw" in f else "#1b9e77" for f in top.index]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top.index, top["mae_increase_M"], xerr=top["sd_over_folds_M"], color=colors, capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("CV MAE increase when permuted, M KZT (mean over 5 validation folds, ±SD)")
    ax.set_title("Permutation importance, base + geo + raw fields\n(green = OSM geo, orange = raw fields, grey = base reference)")
    fig.tight_layout()
    fig.savefig(OUT / "permutation_importance.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- residual figure
def plot_residuals(g: pd.DataFrame, oof: pd.DataFrame) -> None:
    """oof: oof_predictions.csv (id, price, one OOF column per variant)."""
    train = core.load_train()
    d = train.df[[core.ID, "city", "lat", "lon"]].merge(g[[core.ID, "geo_dist_centre_km"]], on=core.ID, validate="1:1")
    d = d.merge(oof, on=core.ID, validate="1:1")
    d["resid_base"] = np.log(d["price"]) - np.log(d["base (Endterm)"])
    d["resid_geo"] = np.log(d["price"]) - np.log(d["base + geo (OSM)"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for j, city in enumerate(["Astana", "Almaty"]):
        c = d[(d["city"] == city) & d["lat"].notna()]
        ax = axes[0, j]
        sc = ax.scatter(c["lon"], c["lat"], c=c["resid_base"].clip(-0.4, 0.4), cmap="RdBu_r", s=7, vmin=-0.4, vmax=0.4)
        clat, clon = geo.CENTRES[city]
        ax.plot(clon, clat, marker="*", color="black", ms=14, label="centre")
        for r_km in (2, 5, 10):
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(clon + r_km / (111.32 * np.cos(np.radians(clat))) * np.cos(t), clat + r_km / 110.57 * np.sin(t),
                    color="grey", lw=0.7, ls="--")
        ax.set_aspect(1 / np.cos(np.radians(clat)))
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_title(f"{city}: base OOF log residual\n(red = asks more than predicted; rings 2 / 5 / 10 km)", fontsize=10)
        ax.legend(loc="lower right")
        fig.colorbar(sc, ax=ax, label="log(price) - log(OOF prediction), clipped ±0.4")
        ax = axes[1, j]
        bins = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 25, 60])
        c = c.assign(bin=pd.cut(c["geo_dist_centre_km"], bins))
        for col, lab, colr in [("resid_base", "base (Endterm)", "#7570b3"), ("resid_geo", "base + geo", "#1b9e77")]:
            s = c.groupby("bin", observed=True)[col].agg(["mean", "sem", "size"])
            mids = [iv.mid for iv in s.index]
            ax.errorbar(mids, s["mean"], yerr=1.96 * s["sem"], marker="o", capsize=3, label=lab, color=colr)
            ax.scatter(c["geo_dist_centre_km"], c[col], s=2, alpha=0.12, color=colr)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("distance to city centre, km")
        ax.set_ylabel("OOF log residual (mean per bin, 95 % CI)")
        ax.set_title(f"{city}: residual vs distance to centre")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "residual_map.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--plots-only" in sys.argv:       # redraw figures from saved outputs, no refitting
        plot_residuals(pd.read_csv(OUT / "features_geo.csv"), pd.read_csv(OUT / "oof_predictions.csv"))
        if (OUT / "permutation_importance.csv").exists():
            plot_importance(pd.read_csv(OUT / "permutation_importance.csv", index_col=0))
        return
    if "--importance-only" in sys.argv:  # permutation importance only (features from the OSM cache)
        g, rawf = build_features()
        imp = permutation_importance(g, rawf, n_jobs=5)
        plot_importance(imp)
        print(imp.head(25).to_string(), flush=True)
        return
    g, rawf = build_features()
    print("geo cols:", g.shape[1] - 1, "raw cols:", rawf.shape[1] - 1, flush=True)
    table, results = run_cv(g, rawf)
    print(table.drop(columns=["folds_mae_M"]).to_string(), flush=True)
    plot_residuals(g, pd.read_csv(OUT / "oof_predictions.csv"))
    imp = permutation_importance(g, rawf)
    plot_importance(imp)
    print(imp.head(20).to_string(), flush=True)


if __name__ == "__main__":
    main()
