"""Leakage & evaluation audit (Final, lectures 2 and 6).

1. Duplicate / re-posted listings among all 4,815 clean ads (krisha_ml.dedup): pairs, groups,
   how many groups cross the fixed train/test split, how many test ads have a twin in train.
   (Test ads enter only this descriptive count: their card incl. asking price, photos and text are
   matched, but no model is fitted, tuned or scored on them. The CV folds below are built from a
   SECOND duplicate search over the training ads only, so no test row influences any choice.)
2. CV schemes on the 3,852 TRAINING ads with the unchanged Endterm base model:
   random KFold (Endterm), GroupKFold by duplicate group, GroupKFold by building, spatial ~1 km
   blocks; each with several fold seeds (seed spread = noise of the comparison).
3. Twin analysis: OOF error of training ads whose twin sits in another fold, random vs group folds.
   (random, random+dupfix and placebo: all seeds, paired by seed; group/spatial schemes: the first 10 seeds)
4. Recommended Final folds -> outputs/leakage/folds_group.csv; figures.

Every CV run is cached in cache/leakage_cv/<scheme>_s<seed>.json, so the script is resumable.
Usage:  python scripts/leakage_audit.py [--seeds 42 1 2 3 4] [--n-jobs 2] [--no-cv]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from krisha_ml import core, cv_schemes as cs, dedup, raw  # noqa: E402

OUT = core.OUTPUT_DIR / "leakage"
LOCAL = OUT / "local"                      # gitignored: contains listing photos
CV_CACHE = core.CACHE_DIR / "leakage_cv"
SCHEMES = {"random": "Random KFold (Endterm)", "random_dupfix": "Same KFold, duplicate groups kept together",
           "dup": "GroupKFold by duplicate group",
           "building": "GroupKFold by building (+dups)", "spatial1km": "Spatial 1 km blocks (+dups)",
           "spatial3km": "Spatial 3 km blocks (+dups)",
           "placebo": "Placebo: same KFold, 26 random non-duplicate ads moved"}
SHORT = {"random": "Random KFold\n(Endterm scheme)", "random_dupfix": "Random KFold,\nduplicates together", "dup": "GroupKFold\nduplicate groups",
         "building": "GroupKFold\nbuildings", "spatial1km": "Spatial blocks\n1 km", "spatial3km": "Spatial blocks\n3 km",
         "placebo": "Placebo: 26 random\nads moved"}
N_PLACEBO = 26   # = ads moved by repairing the Endterm folds (asserted in main)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- 1. duplicates
def duplicate_section(df: pd.DataFrame, test_ids: set[int]) -> tuple[dedup.DuplicateReport, dict]:
    rep = dedup.find_duplicates(df)
    split = pd.Series(np.where(df[core.ID].isin(test_ids), "test", "train"), index=df[core.ID].to_numpy())
    pairs = rep.pairs.assign(split_a=split.reindex(rep.pairs["id_a"]).to_numpy(), split_b=split.reindex(rep.pairs["id_b"]).to_numpy())
    pairs.to_csv(OUT / "duplicate_pairs.csv", index=False)
    rep.image_pairs.to_csv(OUT / "duplicate_image_pairs.csv", index=False)

    summary = {"n_ads": int(len(df)), "n_pairs": int(len(pairs)),
               "pairs_by_signal": pairs["signals"].value_counts().to_dict(),
               "pairs_by_tier": pairs["tier"].value_counts().to_dict(),
               "image_pairs_matched": int(len(rep.image_pairs)), "image_pairs_stock_hub": int(rep.image_pairs["hub"].sum())}
    for tier in (None, "same_flat"):
        g = dedup.group_ids(df[core.ID].to_numpy(), rep.pairs, tier=tier).assign(split=split.reindex(df[core.ID]).to_numpy())
        if tier is None:
            g.to_csv(OUT / "duplicate_groups.csv", index=False)
        multi = g[g["group_size"] > 1]
        per_group = multi.groupby("group")["split"].agg(lambda s: set(s))
        cross = per_group.map(lambda s: s == {"train", "test"})
        test_twin = multi[(multi["split"] == "test") & multi["group"].isin(per_group[cross].index)]
        key = "all_links" if tier is None else "same_flat"
        if tier is None:   # ids only: lets the integration stage also report test MAE without these ads
            tier_by_group = rep.pairs.assign(group=g.set_index(core.ID)["group"].reindex(rep.pairs["id_a"]).to_numpy()) \
                .groupby("group")["tier"].agg(lambda t: "same_flat" if (t == "same_flat").any() else "near")
            test_twin.assign(link_tier=tier_by_group.reindex(test_twin["group"]).to_numpy())[[core.ID, "group", "link_tier"]] \
                .to_csv(OUT / "test_ads_with_train_twin.csv", index=False)
        summary[key] = {"groups": int(per_group.size), "ads_in_groups": int(len(multi)),
                        "group_sizes": multi.groupby("group").size().value_counts().sort_index().to_dict(),
                        "groups_crossing_split": int(cross.sum()),
                        "test_ads_with_train_twin": int(len(test_twin)),
                        "test_ads_with_train_twin_pct": round(100 * len(test_twin) / len(test_ids), 2),
                        "train_ads_in_groups": int((multi["split"] == "train").sum())}
    summary["example_pairs"] = pairs.sort_values(["tier", "signals"], ascending=[False, True]).groupby("signals").head(2)[
        ["id_a", "id_b", "signals", "tier", "split_a", "split_b"]].to_dict("records")
    return rep, summary


# --------------------------------------------------------------------------- 2. CV schemes
def scheme_groups(train: core.Dataset, dup_groups: np.ndarray) -> dict[str, np.ndarray]:
    rt = raw.load_raw_details().set_index("id")["complex_id"]
    complex_id = rt.reindex(train.ids).to_numpy(dtype=object)
    return {"dup": dup_groups,
            "building": cs.merge_labels(cs.building_labels(train.df, complex_id), dup_groups),
            "spatial1km": cs.merge_labels(cs.spatial_blocks(train.df, 1.0), dup_groups),
            "spatial3km": cs.merge_labels(cs.spatial_blocks(train.df, 3.0), dup_groups)}


def scheme_folds(scheme: str, seed: int, n: int, groups: dict[str, np.ndarray]):
    if scheme == "random":
        return cs.random_folds(n, seed)
    if scheme == "placebo":   # falsification test: is the dupfix change just fold-composition noise?
        fv = cs.fold_vector(cs.random_folds(n, seed), n)
        rng = np.random.default_rng(1000 + seed)
        out_of_groups = np.flatnonzero(pd.Series(groups["dup"]).map(pd.Series(groups["dup"]).value_counts()).to_numpy() == 1)
        idx = rng.choice(out_of_groups, N_PLACEBO, replace=False)
        fv[idx] = (fv[idx] + rng.integers(1, cs.N_FOLDS, N_PLACEBO)) % cs.N_FOLDS
        return cs.folds_from_vector(fv)
    if scheme == "random_dupfix":
        return cs.folds_from_vector(cs.repair_folds(cs.fold_vector(cs.random_folds(n, seed), n), groups["dup"]))
    return cs.group_folds(groups[scheme], seed)


def seeds_for(scheme: str, seeds: list[int]) -> list[int]:
    """Fold seeds per scheme. The machine is shared by several tracks (one CV run took 1-13 min), so
    only the paired random / random+dupfix / placebo comparison gets all seeds; the others get 10."""
    if scheme.startswith("random") or scheme == "placebo":
        return seeds
    return seeds[:10]


def run_cv(scheme: str, seed: int, folds, train: core.Dataset, n_jobs: int) -> dict:
    path = CV_CACHE / f"{scheme}_s{seed}.json"
    fv = cs.fold_vector(folds, len(train))
    if path.exists():
        rec = json.loads(path.read_text())
        if rec.get("fold_hash") == int(pd.util.hash_array(fv).sum() % 2**31):
            return rec
    t = time.time()
    res = cs.cv_with_folds(f"{scheme} s{seed}", folds, train, n_jobs=n_jobs)
    rec = {"scheme": scheme, "seed": seed, **res.row(), "fold_sizes": [int(len(va)) for _, va in folds],
           "fold_hash": int(pd.util.hash_array(fv).sum() % 2**31), "seconds": round(time.time() - t, 1),
           "oof_pred": res.oof_pred.round(0).tolist()}
    CV_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, path)
    log(f"{scheme} seed {seed}: {res}  ({rec['seconds']} s)")
    return rec


# --------------------------------------------------------------------------- 3. twin analysis
def twin_analysis(train: core.Dataset, dup_groups: np.ndarray, rec_random: dict, rec_dup: dict) -> dict:
    """Training ads in duplicate groups: error with a twin in another fold (Endterm folds) vs the same
    ads once their group is kept together (repaired folds): a paired, ad-level comparison."""
    folds_r = cs.fold_vector(cs.random_folds(len(train), 42), len(train))
    g = pd.Series(dup_groups)
    size = g.map(g.value_counts()).to_numpy()
    in_group = size > 1
    other_fold = np.zeros(len(train), bool)
    for _, idx in g[in_group].groupby(g[in_group]).groups.items():
        idx = np.asarray(idx)
        for i in idx:
            other_fold[i] = (folds_r[idx[idx != i]] != folds_r[i]).any()
    y = train.y
    ape = lambda p: np.abs(np.asarray(p) - y) / y * 100
    a_r, a_d = ape(rec_random["oof_pred"]), ape(rec_dup["oof_pred"])
    ae_r, ae_d = np.abs(np.asarray(rec_random["oof_pred"]) - y) / 1e6, np.abs(np.asarray(rec_dup["oof_pred"]) - y) / 1e6
    out = {"train_ads_in_dup_groups": int(in_group.sum()),
           "train_ads_with_twin_in_other_fold_random42": int(other_fold.sum())}
    for name, m in [("twin_in_other_fold", other_fold), ("no_group", ~in_group)]:
        out[name] = {"n": int(m.sum()), "random_mae_M": round(float(ae_r[m].mean()), 3), "repaired_mae_M": round(float(ae_d[m].mean()), 3),
                     "random_mdape": round(float(np.median(a_r[m])), 2), "repaired_mdape": round(float(np.median(a_d[m])), 2)}
    # how good is "predict the twin's price"? (upper bound of what a memorising model can gain)
    twin_price = np.full(len(train), np.nan)
    for _, idx in g[in_group].groupby(g[in_group]).groups.items():
        idx = np.asarray(idx)
        for i in idx:
            twin_price[i] = np.median(y[idx[idx != i]])
    m = in_group
    out["twin_price_as_prediction"] = {"n": int(m.sum()), "mdape": round(float(np.median(np.abs(twin_price[m] - y[m]) / y[m] * 100)), 2),
                                       "mae_M": round(float(np.mean(np.abs(twin_price[m] - y[m])) / 1e6), 3)}
    return out


# --------------------------------------------------------------------------- 4. figures
def cv_figure(table: pd.DataFrame, base_sd: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, muted = "#2a78d6", "#0b0b0b", "#52514e"
    agg = table.groupby("scheme", sort=False).agg(mean=("cv_mae_M", "mean"), lo=("cv_mae_M", "min"), hi=("cv_mae_M", "max"),
                                                  fold_sd=("cv_mae_sd_M", "mean")).reindex(list(SCHEMES))
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    x = np.arange(len(agg))
    ax.bar(x, agg["mean"], width=0.55, color=blue, alpha=0.85, zorder=2)
    ax.errorbar(x, agg["mean"], yerr=agg["fold_sd"], fmt="none", ecolor=muted, capsize=5, lw=1.2, zorder=3, label="± mean fold SD")
    for xi, s in zip(x, agg.index):
        v = table.loc[table["scheme"] == s, "cv_mae_M"].to_numpy()
        ax.scatter(np.full(len(v), xi) + np.linspace(-0.12, 0.12, len(v)), v, s=14, color=ink, zorder=4,
                   label="one fold seed" if xi == 0 else None)
        ax.text(xi, agg.loc[s, "mean"] + agg.loc[s, "fold_sd"] + 0.08, f"{agg.loc[s, 'mean']:.2f}", ha="center", color=ink, fontsize=10)
    r = agg.loc["random", "mean"]
    ax.axhline(r, color=muted, lw=1, ls="--", zorder=1)
    ax.axhspan(r - base_sd, r + base_sd, color=muted, alpha=0.08, zorder=0, label="random KFold ± 1 base SD (rule band)")
    ax.set_xticks(x, [SHORT[s] for s in agg.index], fontsize=9)
    e42 = table.loc[(table["scheme"] == "random") & (table["seed"] == 42), "cv_mae_M"].iloc[0]
    ax.scatter([0.3], [e42], marker="D", s=36, color="#eb6834", zorder=5, label=f"Endterm folds, seed 42 ({e42:.2f}, HP tuned on them)")
    ax.set_ylabel("5-fold CV MAE, M KZT (training ads)")
    ax.set_ylim(0, max(agg["hi"].max(), (agg["mean"] + agg["fold_sd"]).max()) * 1.15)
    ax.set_title("Endterm base model under different CV schemes (3,852 training ads; bar = mean over fold seeds)", fontsize=11, color=ink)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def duplicate_grid(pairs: pd.DataFrame, img_pairs: pd.DataFrame, path: Path, n_rows: int = 6) -> None:
    """Side-by-side photos of example duplicate pairs (LOCAL ONLY: listing photos are not redistributed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    photo_pairs = pairs[pairs["signals"].str.contains("phash|emb")]
    ex = pd.concat([photo_pairs[photo_pairs["tier"] == "same_flat"].head(n_rows // 2),
                    photo_pairs[photo_pairs["tier"] == "near"].head(n_rows - n_rows // 2)])
    fig, axes = plt.subplots(len(ex), 2, figsize=(6, 2.6 * len(ex)), dpi=110)
    for row, (_, p) in zip(np.atleast_2d(axes), ex.iterrows()):
        m = img_pairs[((img_pairs["ad_a"] == p["id_a"]) & (img_pairs["ad_b"] == p["id_b"]))
                      | ((img_pairs["ad_a"] == p["id_b"]) & (img_pairs["ad_b"] == p["id_a"]))].sort_values("hub").iloc[0]
        for ax, a, k in zip(row, (m["ad_a"], m["ad_b"]), (m["k_a"], m["k_b"])):
            ax.imshow(Image.open(dedup.PHOTO_DIR / f"{a}_{k}.jpg").convert("RGB"))
            ax.set_title(f"ad {a} photo {k}", fontsize=7)
            ax.axis("off")
        row[0].text(-0.05, 0.5, f"{p['tier']}\n{p['signals']}", transform=row[0].transAxes, ha="right", va="center", fontsize=7)
    fig.suptitle("Example duplicate pairs (pHash/dHash/embedding match)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, *range(1, 20)])
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--no-cv", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); LOCAL.mkdir(exist_ok=True)

    df, test_ids, train = core.load_clean(), core.load_test_ids(), core.load_train()
    t = time.time()
    rep, summary = duplicate_section(df, test_ids)
    log(f"duplicates: {summary['n_pairs']} pairs, {summary['all_links']['groups']} groups ({time.time() - t:.1f} s)")
    duplicate_grid(rep.pairs, rep.image_pairs, LOCAL / "duplicate_examples.png")

    # duplicate groups for the folds: a separate search over the training ads only (TF-IDF vocabulary,
    # stock-image hubs and card matches never see a test ad)
    rep_train = dedup.find_duplicates(train.df)
    dup_train = dedup.group_ids(train.ids, rep_train.pairs)["group"].to_numpy()
    dup_all = dedup.group_ids(train.ids, rep.pairs)["group"].to_numpy()
    summary["train_only_groups_equal_all_ads_groups"] = bool(
        pd.crosstab(pd.factorize(dup_train)[0], pd.factorize(dup_all)[0]).gt(0).sum(axis=1).eq(1).all()
        and pd.Series(dup_train).nunique() == pd.Series(dup_all).nunique())
    groups = scheme_groups(train, dup_train)
    summary["n_groups_per_scheme"] = {k: int(pd.Series(v).nunique()) for k, v in groups.items()}

    # recommended folds: the Endterm folds with every duplicate group moved whole into one fold
    fv = cs.fold_vector(cs.random_folds(len(train), 42), len(train))
    fv_fixed = cs.repair_folds(fv, dup_train)
    summary["recommended_folds_moved_ads"] = int((fv != fv_fixed).sum())
    assert summary["recommended_folds_moved_ads"] == N_PLACEBO
    fv = fv_fixed
    gsize = pd.Series(dup_train).map(pd.Series(dup_train).value_counts()).to_numpy()
    pd.DataFrame({"id": train.ids, "fold": fv, "dup_group": dup_train, "dup_group_size": gsize}).to_csv(OUT / "folds_group.csv", index=False)
    assert all(pd.Series(fv).groupby(dup_train).nunique() == 1)

    if not args.no_cv:
        recs = [run_cv(s, seed, scheme_folds(s, seed, len(train), groups), train, args.n_jobs)
                for s in SCHEMES for seed in seeds_for(s, args.seeds)]
        table = pd.DataFrame([{k: v for k, v in r.items() if k != "oof_pred"} for r in recs])
        table.to_csv(OUT / "cv_schemes_runs.csv", index=False)
        base = core.base_cv()
        agg = table.groupby("scheme", sort=False).agg(cv_mae_mean=("cv_mae_M", "mean"), seed_sd=("cv_mae_M", "std"),
                                                      fold_sd_mean=("cv_mae_sd_M", "mean"), mdape=("cv_mdape", "mean"),
                                                      r2=("cv_r2", "mean"), n_seeds=("seed", "size")).reindex(list(SCHEMES))
        agg["delta_vs_random_M"] = agg["cv_mae_mean"] - agg.loc["random", "cv_mae_mean"]
        agg.insert(0, "description", [SCHEMES[s] for s in agg.index])
        agg.round(4).to_csv(OUT / "cv_schemes_summary.csv")
        summary["cv_schemes"] = agg.round(4).reset_index().to_dict("records")
        r42 = next(r for r in recs if r["scheme"] == "random" and r["seed"] == 42)
        d42 = next(r for r in recs if r["scheme"] == "random_dupfix" and r["seed"] == 42)
        paired = table.pivot(index="seed", columns="scheme", values="cv_mae_M")
        diff = paired["random_dupfix"] - paired["random"]
        summary["paired_dupfix_minus_random_M"] = {"per_seed": diff.round(4).to_dict(), "mean": round(float(diff.mean()), 4),
                                                   "sd": round(float(diff.std()), 4)}
        assert abs(r42["cv_mae_M"] - round(base.mean, 4)) < 1e-4, "random seed 42 must equal the Endterm base CV"
        summary["recommended_folds"] = {"file": "outputs/leakage/folds_group.csv", "scheme": "Endterm KFold(5, shuffle, 42), duplicate groups moved whole into one fold",
                                        "base_cv_mae_M": d42["cv_mae_M"], "base_cv_sd_M": d42["cv_mae_sd_M"],
                                        "base_folds_mae_M": d42["folds_mae_M"], "fold_sizes": d42["fold_sizes"],
                                        "rule_margin_M": round(base.sd, 4),
                                        "rule_threshold_M": round(d42["cv_mae_M"] - base.sd, 4),
                                        "rule_note": "keep a variant only if its CV MAE on these folds < base on these folds - fixed Endterm margin (0.3535 M)"}
        pl = paired["placebo"] - paired["random"]
        summary["paired_placebo_minus_random_M"] = {"per_seed": pl.round(4).to_dict(), "mean": round(float(pl.mean()), 4),
                                                    "sd": round(float(pl.std()), 4)}
        dp = paired["random_dupfix"] - paired["placebo"]
        summary["paired_dupfix_minus_placebo_M"] = {"per_seed": dp.round(4).to_dict(), "mean": round(float(dp.mean()), 4),
                                                    "sd": round(float(dp.std()), 4)}
        summary["twin_analysis"] = twin_analysis(train, dup_train, r42, d42)
        cv_figure(table, base.sd, OUT / "cv_schemes.png")
    path = OUT / "leakage_summary.json"
    if args.no_cv and path.exists():   # keep the CV sections of the last full run
        summary = {**json.loads(path.read_text()), **summary}
    path.write_text(json.dumps(summary, indent=2, default=str))
    log(f"done -> {OUT.relative_to(core.ROOT)}")


if __name__ == "__main__":
    main()
