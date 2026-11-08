"""Text NLP track: run the CV experiments and write outputs/text/.

Stages (run any subset, default all except finetune):
  tfidf     TF-IDF residual corrections (word, char, word+char, price-weighted) + TF-IDF OOF as an HGB feature
  emb       per sentence-transformer: Ridge residual correction and fold-fitted PCA(32) features
  combo     TF-IDF + best embedding in one Ridge
  flags     regex flags: stats vs OOF base residuals, flags as features / residual Ridge, similarity scores
  terms     top TF-IDF terms pushing the price up / down (train rows, honest OOF residuals)
  finetune  rubert-tiny2 fine-tuned on the fold residuals (GPU, slow)
  report    cv.csv figure + features_text.csv
  dupcheck  gain of the best text correction on near-duplicate vs other descriptions

Every CV row is appended to outputs/text/cv.csv (re-running a variant replaces its row); OOF
log-scale corrections of residual variants are cached as cache/text_oof/<slug>.npy.
Usage: python scripts/run_text.py [stage ...] [--jobs 2]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from krisha_ml import core, text  # noqa: E402

OUT = core.OUTPUT_DIR / "text"
OOF_DIR = core.CACHE_DIR / "text_oof"
CV_CSV = OUT / "cv.csv"
EMB_KEYS = ["rubert-tiny2", "user-base", "e5-base"]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def record(res: core.CVResult, kind: str, notes: str = "", residual: bool = False, seconds: float = 0.0) -> dict:
    base = core.base_cv()
    row = res.row(base)
    row.update(kind=kind, notes=notes, seconds=round(seconds, 1))
    row["folds_mae_M"] = " / ".join(f"{v:.3f}" for v in res.fold_mae)
    OUT.mkdir(parents=True, exist_ok=True)
    cv = pd.read_csv(CV_CSV) if CV_CSV.exists() else pd.DataFrame()
    if len(cv):
        cv = cv[cv["variant"] != row["variant"]]
    pd.concat([cv, pd.DataFrame([row])], ignore_index=True).to_csv(CV_CSV, index=False)
    if residual:
        OOF_DIR.mkdir(parents=True, exist_ok=True)
        np.save(OOF_DIR / f"{slug(res.name)}.npy", text.residual_delta(res))
    print(f"{row['variant']:<55} {row['cv_mae_M']:.4f} +- {row['cv_mae_sd_M']:.4f}  delta {row['delta_vs_base_M']:+.4f}"
          f"  better {row['folds_better']}/5  rule {row['passes_1sd_rule']}  {notes}", flush=True)
    return row


def run_resid(name: str, make, kind: str, jobs: int):
    log = text.FitLog(str(core.CACHE_DIR / f"alphas_{slug(name)}.txt")).reset()
    t0 = time.time()
    res = core.cv_evaluate_residual(make(log), name, n_jobs=jobs)
    alphas = log.read()
    note = f"alpha per fold (GCV) {', '.join(f'{a:g}' for a in alphas)}" if alphas else ""
    return record(res, kind, note, residual=True, seconds=time.time() - t0)


def run_feat(name: str, kind: str, jobs: int, notes: str = "", **kw):
    t0 = time.time()
    res = core.cv_evaluate(name, n_jobs=jobs, **kw)
    return record(res, kind, notes, seconds=time.time() - t0)


# --------------------------------------------------------------------------- stages
def stage_tfidf(jobs):
    run_resid("TF-IDF word 1-2 + Ridge residual", lambda l: text.tfidf_corrector(char=False, log=l), "tfidf", jobs)
    run_resid("TF-IDF char 3-5 + Ridge residual", lambda l: text.tfidf_corrector(word=False, log=l), "tfidf", jobs)
    run_resid("TF-IDF word+char + Ridge residual", lambda l: text.tfidf_corrector(log=l), "tfidf", jobs)
    run_resid("TF-IDF word+char + Ridge residual, price-weighted",
              lambda l: text.tfidf_corrector(weighted=True, log=l), "tfidf", jobs)
    run_feat("TF-IDF word+char OOF residual as 1 HGB feature", "tfidf", jobs,
             "inner 5-fold stacking feature", feature_fn=text.oof_text_feature(text.tfidf_corrector()))


def stage_emb(jobs):
    for key in EMB_KEYS:
        if not text.emb_path(key).exists():
            print(f"skip {key}: no cached embedding", flush=True)
            continue
        run_resid(f"{key} embedding + Ridge residual", lambda l, k=key: text.embedding_corrector(k, log=l), "embedding", jobs)
        run_feat(f"{key} embedding PCA(32) as HGB features", "embedding", jobs, "PCA fitted per fold",
                 feature_fn=text.embedding_pca_features(key))


def best_embedding() -> str:
    cv = pd.read_csv(CV_CSV)
    rows = cv[cv["variant"].str.endswith("embedding + Ridge residual")]
    return rows.sort_values("cv_mae_M").iloc[0]["variant"].split(" ")[0]


def stage_combo(jobs):
    key = best_embedding()
    flags = flag_table()
    run_resid(f"TF-IDF word+char + {key} embedding, one Ridge residual",
              lambda l: text.combined_corrector(key, log=l), "combined", jobs)
    run_resid(f"TF-IDF word+char + {key} embedding, one Ridge residual, price-weighted",
              lambda l: text.combined_corrector(key, weighted=True, log=l), "combined", jobs)
    run_resid(f"TF-IDF word+char + {key} embedding + regex flags, one Ridge residual",
              lambda l: text.combined_corrector(key, flags, log=l), "combined", jobs)
    run_resid("TF-IDF word+char + all 3 embeddings + regex flags, one Ridge residual",
              lambda l: text.combined_corrector(tuple(EMB_KEYS), flags, log=l), "combined", jobs)


def stage_emb_resid(jobs):
    """Ridge residual of every embedding only (re-run after widening the alpha grid)."""
    for key in EMB_KEYS:
        run_resid(f"{key} embedding + Ridge residual", lambda l, k=key: text.embedding_corrector(k, log=l), "embedding", jobs)


def flag_table() -> pd.DataFrame:
    return text.extract_flags(core.load_clean())


def stage_flags(jobs):
    flags = flag_table()
    oof = pd.read_csv(core.OUTPUT_DIR / "core" / "base_oof.csv")
    m = oof.merge(flags, on="id", how="left")
    rows = []
    for c in [c for c in flags.columns if c.startswith("flag_")]:
        on, off = m.loc[m[c] == 1, "log_resid"], m.loc[m[c] == 0, "log_resid"]
        boot = np.random.default_rng(core.SEED).choice(on.to_numpy(), (2000, len(on))) if len(on) else np.zeros((1, 1))
        lo, hi = np.percentile(np.median(boot, axis=1), [2.5, 97.5])
        rows.append({"flag": c.removeprefix("flag_"), "n_train_ads": int(len(on)), "share_pct": round(100 * len(on) / len(m), 2),
                     "median_resid_pct_flagged": round(100 * np.expm1(on.median()), 2),
                     "ci95_lo_pct": round(100 * np.expm1(lo), 2), "ci95_hi_pct": round(100 * np.expm1(hi), 2),
                     "median_resid_pct_rest": round(100 * np.expm1(off.median()), 2),
                     "mean_abs_resid_pct_flagged": round(100 * np.expm1(on.abs().mean()), 2),
                     "mannwhitney_p": float(mannwhitneyu(on, off).pvalue) if len(on) else np.nan,
                     "n_all_clean_ads": int(flags[c].sum())})
    stats = pd.DataFrame(rows).sort_values("median_resid_pct_flagged")
    # embedding-similarity agreement (best embedding): AUC of the prototype similarity for the regex flag
    try:
        from sklearn.metrics import roc_auc_score
        key = best_embedding()
        sims = text.flag_similarity(key)
        sims.to_csv(core.CACHE_DIR / f"text_flag_sim_{key}.csv", index=False)
        ms = flags.merge(sims, on="id")
        stats["sim_auc_vs_regex"] = [round(roc_auc_score(ms[f"flag_{f}"], ms[f"sim_{f}"]), 3) if ms[f"flag_{f}"].nunique() > 1
                                     else np.nan for f in stats["flag"]]
        stats["sim_model"] = key
    except Exception as exc:  # documented, not fatal
        print("similarity skipped:", exc, flush=True)
        sims = None
    stats.to_csv(OUT / "flags_stats.csv", index=False)
    print(stats.to_string(index=False), flush=True)
    fig_flags(stats)

    feat = flags.copy()
    run_feat("regex flags (12) as HGB features", "flags", jobs, extra=feat)
    run_resid("regex flags (12) + Ridge residual", lambda l: text.flag_corrector(feat, log=l), "flags", jobs)
    if sims is not None:
        run_feat(f"flag similarity scores (12, {best_embedding()}) as HGB features", "flags", jobs, extra=sims)


def stage_terms(_jobs):
    tr = core.load_train()
    oof = pd.read_csv(core.OUTPUT_DIR / "core" / "base_oof.csv").set_index("id").reindex(tr.ids)
    terms = text.top_terms(tr.df["description"], oof["log_resid"].to_numpy(), n=25)
    terms.to_csv(OUT / "top_terms.csv", index=False)
    print(terms.to_string(index=False), flush=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    t = terms.sort_values("coef")
    ax.barh(t["term"], t["coef"], color=np.where(t["coef"] > 0, "#2a7ab9", "#d0573a"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Ridge coefficient on the base model's OOF log residual\n(TF-IDF units; > 0 = asking price above the base prediction)")
    ax.set_ylabel("term (sanitized description, word 1-2 gram, in >= 10 ads)")
    ax.set_title("Description terms vs the base model residual")
    fig.tight_layout(); fig.savefig(OUT / "top_terms.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def stage_finetune(_jobs):
    log: list = []
    t0 = time.time()
    res = core.cv_evaluate_residual(text.finetune_corrector(log=log), "rubert-tiny2 fine-tuned on fold residuals (3 ep)", n_jobs=1)
    (OUT / "finetune_log.json").write_text(json.dumps(log, indent=1))
    record(res, "finetune", "lr 5e-5 (head x20), 3 epochs, max_len 256, Smooth-L1, OneCycle", residual=True, seconds=time.time() - t0)


# --------------------------------------------------------------------------- report
def fig_flags(stats: pd.DataFrame):
    s = stats.sort_values("median_resid_pct_flagged")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(s))
    ax.errorbar(s["median_resid_pct_flagged"], y, xerr=[s["median_resid_pct_flagged"] - s["ci95_lo_pct"],
                s["ci95_hi_pct"] - s["median_resid_pct_flagged"]], fmt="o", color="#2a7ab9", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, [f"{f} (n={n})" for f, n in zip(s["flag"], s["n_train_ads"])])
    ax.set_xlabel("median out-of-fold residual of the base model, %\n(actual / predicted - 1), 95 % bootstrap CI")
    ax.set_title("Description flags vs base model error (3,852 training ads)")
    fig.tight_layout(); fig.savefig(OUT / "flags_residuals.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def stage_report(_jobs):
    cv = pd.read_csv(CV_CSV)
    base = core.base_cv()
    thr = base.mean - base.sd
    c = cv.sort_values("cv_mae_M", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(c) + 1.8))
    y = np.arange(len(c))
    ax.errorbar(c["cv_mae_M"], y, xerr=c["cv_mae_sd_M"], fmt="o", color="#2a7ab9", capsize=3)
    ax.axvline(base.mean, color="grey", ls="--", label=f"Endterm base {base.mean:.3f} M")
    ax.axvline(thr, color="#d0573a", ls=":", label=f"1-SD rule threshold {thr:.3f} M")
    ax.set_yticks(y, c["variant"])
    ax.set_xlabel("5-fold CV MAE, M KZT\n(mean +- fold SD, 3,852 training ads)")
    ax.set_title("Text track: CV MAE vs the Endterm base")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09 * 22 / (len(c) + 4)), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "cv_text.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # features_text.csv: all clean ads; OOF residual corrections only for training ads (test ads NaN)
    clean = core.load_clean()[[core.ID]]
    feats = clean.merge(flag_table(), on="id", how="left")
    tr = core.load_train()
    oof_cols = {}
    res_rows = cv[cv["kind"].isin(["tfidf", "embedding", "combined", "finetune", "flags"])]
    for v in res_rows["variant"]:
        f = OOF_DIR / f"{slug(v)}.npy"
        if f.exists():
            oof_cols[f"oof_delta_{slug(v)}"[:80]] = pd.Series(np.load(f), index=tr.ids)
    oof_df = pd.DataFrame(oof_cols)
    oof_df.index.name = "id"
    feats = feats.merge(oof_df.reset_index(), on="id", how="left")
    feats.to_csv(OUT / "features_text.csv", index=False)
    print(f"features_text.csv: {feats.shape}", flush=True)
    print(cv[["variant", "cv_mae_M", "cv_mae_sd_M", "cv_mdape", "delta_vs_base_M", "folds_better", "passes_1sd_rule"]]
          .to_string(index=False), flush=True)


def stage_dupcheck(_jobs):
    """Is the best text gain driven by near-duplicate descriptions (relisted flats) across folds?
    Validation ads whose sanitized description has char-4gram TF-IDF cosine > 0.9 with any training-fold
    ad are 'near duplicates'; MAE of base vs best text correction is reported on both groups."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from krisha_ml import raw
    cv = pd.read_csv(CV_CSV)
    best = cv[cv["kind"] == "combined"].sort_values("cv_mae_M").iloc[0]["variant"]
    tr, base, folds = core.load_train(), core.base_cv(), core.cv_folds()
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=2).fit_transform(
        tr.df["description"].fillna("").map(raw.sanitize_text))
    dup = np.zeros(len(tr), bool)
    for trn, va in folds:
        dup[va] = (X[va] @ X[trn].T).max(axis=1).toarray().ravel() > 0.9
    pred = np.exp(np.log(base.oof_pred) + np.load(OOF_DIR / f"{slug(best)}.npy"))
    err0, err1 = np.abs(base.oof_pred - tr.y) / 1e6, np.abs(pred - tr.y) / 1e6
    out = {"variant": best, "n_near_duplicate_val_ads": int(dup.sum()), "share": round(float(dup.mean()), 4)}
    for m, k in [(dup, "near_dup"), (~dup, "rest")]:
        out[k] = {"n": int(m.sum()), "base_mae_M": round(float(err0[m].mean()), 4), "text_mae_M": round(float(err1[m].mean()), 4),
                  "delta_M": round(float(err1[m].mean() - err0[m].mean()), 4)}
    fd = [float(err1[va].mean() - err0[va].mean()) for _, va in folds]
    out["paired_fold_delta_M"] = [round(v, 4) for v in fd]
    out["paired_fold_delta_sd_M"] = round(float(np.std(fd)), 4)
    (OUT / "duplicate_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


STAGES = {"tfidf": stage_tfidf, "emb": stage_emb, "emb_resid": stage_emb_resid, "combo": stage_combo, "flags": stage_flags, "terms": stage_terms,
          "finetune": stage_finetune, "report": stage_report,
          "dupcheck": stage_dupcheck}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=["tfidf", "emb", "combo", "flags", "terms", "report"])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    for s in a.stages:
        t0 = time.time()
        print(f"== stage {s}", flush=True)
        STAGES[s](a.jobs)
        print(f"== stage {s} done in {time.time() - t0:.0f} s", flush=True)
