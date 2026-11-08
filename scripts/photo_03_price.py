"""Photo track step 3: do photos add price information? (5-fold CV on training ads, 1-SD rule)

Every fitted piece is refitted inside the outer fold (core.cv_evaluate feature_fn / residual corrector):
  * condition probabilities: CLIP linear probe (best probe of photo_02), nested: inner 5-fold OOF
    probabilities for the training fold, probe fitted on the whole training fold for the validation fold
  * PCA(16) of ad-mean CLIP / legacy MobileNetV3 embeddings: fitted on the training fold
  * CLIP zero-shot aggregates need no fitting (static ``extra``)
Residual correctors get honest inner-OOF residuals from core.cv_evaluate_residual.

Usage: python scripts/photo_03_price.py [--n-jobs 2]
Outputs: outputs/photos/price_cv.csv, price_cv.json, price_cv_delta.png
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from krisha_ml import core  # noqa: E402
from krisha_ml import photos as ph  # noqa: E402


def best_probe() -> dict:
    t = pd.read_csv(ph.OUT / "classifier_comparison.csv").dropna(subset=["macro_f1_cv"])
    r = t.sort_values("macro_f1_cv").iloc[-1]
    name = r["model"]
    enc = "clip" if "CLIP" in name else name.split("frozen ")[1].split(" ")[0]
    return {"model": name, "encoder": enc, "pool": "interior" if "interior" in name else None, "C": float(r["C"])}


PROBE = best_probe()
EMB_PROBE = ph.ad_embedding_table(PROBE["encoder"], PROBE["pool"])
EMB_CLIP = ph.ad_embedding_table("clip")
EMB_LEGACY = ph.ad_embedding_table("legacy_mnv3")
ZS = pd.read_csv(ph.OUT / "clip_zero_shot_ad.csv")
PC_COLS = [f"photo_p_{c}" for c in ph.CLASSES]


def cond_proba(df_tr, df_va):
    """Nested condition probabilities: inner OOF on the training fold, full-fold probe on validation."""
    X_tr, X_va = ph.lookup(EMB_PROBE, df_tr["id"]), ph.lookup(EMB_PROBE, df_va["id"])
    y_tr = ph.condition_labels(df_tr).values
    inner = list(KFold(5, shuffle=True, random_state=core.SEED).split(X_tr))
    P_tr = ph.probe_oof(X_tr, y_tr, inner, PROBE["C"])
    P_va = ph.probe_fit_predict(X_tr, y_tr, X_va, PROBE["C"])
    return pd.DataFrame(P_tr, columns=PC_COLS), pd.DataFrame(P_va, columns=PC_COLS)


def fn_cond(df_tr, y_tr, df_va):
    return cond_proba(df_tr, df_va)


def fn_cond_imputed(df_tr, y_tr, df_va):
    """Probabilities + stated condition filled from photos where the ad states none (categorical)."""
    out = []
    for d, P in zip((df_tr, df_va), cond_proba(df_tr, df_va)):
        stated = d["condition"].map(ph.CONDITION_TO_CLASS)
        photo = pd.Series(np.array(ph.CLASSES, dtype=object)[np.nan_to_num(P.values, nan=-1).argmax(1)])
        photo[P.isna().any(axis=1).values] = "no_photo"
        P["condition_photo_filled"] = stated.where(stated.notna(), "photo_" + photo).astype(str).values
        out.append(P)
    return tuple(out)


def fn_cond_gated(df_tr, y_tr, df_va):
    """Probabilities only where the ad states no condition (NaN otherwise): photos replace the missing
    field instead of competing with the stated one (motivated by price_subgroups.csv, training CV only)."""
    out = []
    for d, P in zip((df_tr, df_va), cond_proba(df_tr, df_va)):
        P.loc[(ph.condition_labels(d) >= 0).values, :] = np.nan
        out.append(P.add_suffix("_if_unknown"))
    return tuple(out)


def pca_fn(table, prefix, n=16):
    def fn(df_tr, y_tr, df_va):
        X_tr, X_va = ph.lookup(table, df_tr["id"]), ph.lookup(table, df_va["id"])
        ok = ~np.isnan(X_tr).any(1)
        sc = StandardScaler().fit(X_tr[ok])
        pca = PCA(n, random_state=core.SEED).fit(sc.transform(X_tr[ok]))
        cols = [f"{prefix}_pc{i}" for i in range(n)]

        def tr(X):
            E = np.full((len(X), n), np.nan)
            m = ~np.isnan(X).any(1)
            E[m] = pca.transform(sc.transform(X[m]))
            return pd.DataFrame(E, columns=cols)
        return tr(X_tr), tr(X_va)
    return fn


def combine(*fns):
    def fn(df_tr, y_tr, df_va):
        parts = [f(df_tr, y_tr, df_va) for f in fns]
        return (pd.concat([p[0] for p in parts], axis=1), pd.concat([p[1] for p in parts], axis=1))
    return fn


# ---------------------------------------------------------------- residual correctors (log scale)
def ridge_emb(table, scale=False):
    def corr(df_tr, resid_tr, df_va):
        X_tr, X_va = ph.lookup(table, df_tr["id"]), ph.lookup(table, df_va["id"])
        ok_tr, ok_va = ~np.isnan(X_tr).any(1), ~np.isnan(X_va).any(1)
        sc = StandardScaler().fit(X_tr[ok_tr]) if scale else None
        f = (lambda X: sc.transform(X)) if scale else (lambda X: X)
        m = RidgeCV(alphas=np.logspace(-1, 4, 16)).fit(f(X_tr[ok_tr]), resid_tr[ok_tr])
        delta = np.zeros(len(df_va))
        delta[ok_va] = m.predict(f(X_va[ok_va]))
        return delta
    return corr


def hgb_compact(df_tr, resid_tr, df_va):
    """Small, strongly regularised HGB on the compact photo block (condition probabilities + zero-shot)."""
    P_tr, P_va = cond_proba(df_tr, df_va)
    zs_tr = core.join_features(df_tr["id"].values, ZS).drop(columns="id", errors="ignore")
    zs_va = core.join_features(df_va["id"].values, ZS).drop(columns="id", errors="ignore")
    X_tr = pd.concat([P_tr, zs_tr], axis=1).assign(base_log_pred=df_tr["base_log_pred"].values)
    X_va = pd.concat([P_va, zs_va], axis=1).assign(base_log_pred=df_va["base_log_pred"].values)
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=40,
                                      l2_regularization=1.0, random_state=core.SEED).fit(X_tr, resid_tr)
    return m.predict(X_va)


RESUME_DIR = core.CACHE_DIR / "photo_price_cv"


def cached(name, job):
    """Run one CV variant once; results are kept in cache/photo_price_cv/ so a killed run resumes."""
    import hashlib
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    path = RESUME_DIR / (hashlib.sha1(f"{name}|{PROBE}".encode()).hexdigest()[:12] + ".npz")
    train = core.load_train()
    if path.exists():
        z = np.load(path)
        return core._result_from_oof(name, train.y, z["oof"], core.cv_folds(len(train)))
    r = job()
    np.savez(path, oof=r.oof_pred)
    return r


def plot(table, base, path):
    t = table.iloc[1:].sort_values("cv_mae_M", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(t) + 1.6))
    d = t["delta_vs_base_M"].values
    ax.barh(t["variant"], d, color=["#2a9d8f" if x < 0 else "#e76f51" for x in d])
    ax.axvline(0, color="black", lw=0.8)
    span = max(d.max(), 0) - min(d.min(), -base.sd)
    ax.set_xlim(min(d.min(), -base.sd) - 0.25 * span, max(d.max(), 0) + 0.25 * span)   # room for bar labels
    ax.axvline(-base.sd, color="#264653", ls="--", label=f"1-SD rule: -{base.sd:.3f} M")
    for i, (x, s) in enumerate(zip(d, t["folds_better"])):
        ax.text(x, i, f" {x:+.3f} ({int(s)}/5)", va="center", ha="left" if x >= 0 else "right", fontsize=8)
    ax.set_xlabel("change in 5-fold CV MAE vs Endterm base (M KZT); label = delta (folds better)")
    ax.set_title(f"Photo features and the 1-SD rule (base {base.mean:.3f} ± {base.sd:.3f} M KZT)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=2)
    args = ap.parse_args()
    base = core.base_cv()
    print("probe:", PROBE, flush=True)
    zs_extra = ZS.copy()
    zs_rooms = ZS[["id"] + [c for c in ZS if c.startswith("clip_share_") or c in ("clip_interior_share", "n_photos_ok")]]
    runs = [
        ("photo condition probabilities (nested OOF probe)", dict(feature_fn=fn_cond)),
        ("condition probs + stated condition filled from photos", dict(feature_fn=fn_cond_imputed)),
        ("condition probs only where condition is not stated", dict(feature_fn=fn_cond_gated)),
        ("CLIP zero-shot quality + room shares", dict(extra=zs_extra)),
        ("CLIP zero-shot room shares only", dict(extra=zs_rooms)),
        ("CLIP embedding PCA(16)", dict(feature_fn=pca_fn(EMB_CLIP, "clip"))),
        ("legacy MobileNetV3 embedding PCA(16)", dict(feature_fn=pca_fn(EMB_LEGACY, "mnv3"))),
        ("condition probs + zero-shot", dict(extra=zs_extra, feature_fn=fn_cond)),
        ("condition probs + zero-shot + CLIP PCA(16)", dict(extra=zs_extra, feature_fn=combine(fn_cond, pca_fn(EMB_CLIP, "clip")))),
    ]
    residuals = [
        ("residual: RidgeCV on CLIP ad embedding", ridge_emb(EMB_CLIP)),
        ("residual: RidgeCV on CLIP interior-weighted embedding", ridge_emb(ph.ad_embedding_table("clip", "interior"))),
        ("residual: RidgeCV on legacy MobileNetV3 embedding", ridge_emb(EMB_LEGACY, scale=True)),
        ("residual: HGB on condition probs + zero-shot", hgb_compact),
    ]
    rows, res_all = [base.row()], {}
    jobs = [(n, lambda kw=kw, n=n: core.cv_evaluate(n, n_jobs=args.n_jobs, **kw)) for n, kw in runs] + \
           [(n, lambda c=c, n=n: core.cv_evaluate_residual(c, n, n_jobs=args.n_jobs)) for n, c in residuals]
    for name, job in jobs:
        r = cached(name, job)
        rows.append(r.row(base)); res_all[name] = r
        print(json.dumps(rows[-1], default=float), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(ph.OUT / "price_cv.csv", index=False)
    json.dump({"probe": PROBE, "base": base.row(), "rows": rows}, open(ph.OUT / "price_cv.json", "w"), indent=1, default=float)
    oof = pd.DataFrame({"id": core.load_train().ids, **{k: v.oof_pred for k, v in res_all.items()}})
    oof.to_csv(ph.OUT / "price_cv_oof.csv", index=False)
    plot(table, base, ph.OUT / "price_cv_delta.png")
    print(table.to_string())


if __name__ == "__main__":
    main()
