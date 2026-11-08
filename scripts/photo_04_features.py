"""Photo track step 4: id-keyed photo feature table for the integration stage + photo/stated mismatch flag.

outputs/photos/features_photo.csv, one row per clean ad (4,815), columns:
  id, split ("train"/"test"), n_photos_ok, has_photo
  photo_p_<class>      condition probabilities of the best linear probe (CLIP ViT-B-32, interior-weighted
                       ad embedding). TRAIN rows: out-of-fold over core.cv_folds() (each training ad is
                       predicted by a probe that never saw it). TEST rows: probe fitted on all labelled
                       training ads. Test labels are never used.
  photo_class, photo_conf   argmax class and its probability
  clip_*               CLIP zero-shot room shares / quality grades (no fitting, same for train and test)
  clip_pc0..15         PCA(16) of the ad-mean CLIP embedding, PCA fitted on TRAINING ads only.
                       For CV inside the training set refit it per fold (photo_03_price.pca_fn) instead.
  stated_class, mismatch, mismatch_conf
mismatch = "photos_poor_vs_stated_good" when the ad states fresh/good renovation but
p(needs_repair)+p(shell) >= 0.70, "photos_good_vs_stated_poor" when it states needs_repair/shell but
p(fresh)+p(good) >= 0.80 (thresholds fixed a priori, not tuned).

Outputs: features_photo.csv, mismatch_examples.csv, mismatch_summary.json, examples_mismatch.png
(photos: local only, gitignored).
"""
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
from sklearn.preprocessing import StandardScaler  # noqa: E402

from krisha_ml import core  # noqa: E402
from krisha_ml import photos as ph  # noqa: E402

POOR_T, GOOD_T = 0.70, 0.80


def main():
    probe = json.load(open(ph.OUT / "price_cv.json"))["probe"] if (ph.OUT / "price_cv.json").exists() else \
        {"encoder": "clip", "pool": "interior", "C": 0.003}
    clean = core.load_clean()
    train = core.load_train()
    test_ids = np.array(sorted(core.load_test_ids()), dtype=np.int64)
    emb = ph.ad_embedding_table(probe["encoder"], probe["pool"])
    X_tr, X_te = ph.lookup(emb, train.ids), ph.lookup(emb, test_ids)
    y_tr = ph.condition_labels(train.df).values
    P_tr = ph.probe_oof(X_tr, y_tr, core.cv_folds(len(train)), probe["C"])
    P_te = ph.probe_fit_predict(X_tr, y_tr, X_te, probe["C"])

    # OOF quality on labelled training ads (same folds as the price CV)
    lab = (y_tr >= 0) & ~np.isnan(P_tr).any(1)
    oof_metrics = ph.class_metrics(y_tr[lab], P_tr[lab])

    ids = np.concatenate([train.ids, test_ids])
    P = np.vstack([P_tr, P_te])
    F = pd.DataFrame({"id": ids, "split": ["train"] * len(train) + ["test"] * len(test_ids)})
    for j, c in enumerate(ph.CLASSES):
        F[f"photo_p_{c}"] = P[:, j]
    has = ~np.isnan(P).any(1)
    F["has_photo"] = has.astype(int)
    F["photo_class"] = np.where(has, np.array(ph.CLASSES, dtype=object)[np.nan_to_num(P, nan=-1).argmax(1)], None)
    F["photo_conf"] = np.where(has, np.nanmax(np.nan_to_num(P, nan=-1), axis=1), np.nan)
    zs = pd.read_csv(ph.OUT / "clip_zero_shot_ad.csv")
    F = F.merge(zs, on="id", how="left")
    F["n_photos_ok"] = F["n_photos_ok"].fillna(0).astype(int)

    clip = ph.ad_embedding_table("clip")
    C_tr, C_all = ph.lookup(clip, train.ids), ph.lookup(clip, ids)
    ok = ~np.isnan(C_tr).any(1)
    sc = StandardScaler().fit(C_tr[ok])
    pca = PCA(16, random_state=core.SEED).fit(sc.transform(C_tr[ok]))
    E = np.full((len(ids), 16), np.nan)
    m = ~np.isnan(C_all).any(1)
    E[m] = pca.transform(sc.transform(C_all[m]))
    F[[f"clip_pc{i}" for i in range(16)]] = E

    stated = clean.set_index("id")["condition"].map(ph.CONDITION_TO_CLASS).reindex(ids).values
    F["stated_class"] = stated
    p_poor = F["photo_p_needs_repair"] + F["photo_p_shell"]
    p_good = F["photo_p_fresh"] + F["photo_p_good"]
    s_good = F["stated_class"].isin(["fresh", "good"])
    s_poor = F["stated_class"].isin(["needs_repair", "shell"])
    F["mismatch"] = np.select([s_good & (p_poor >= POOR_T), s_poor & (p_good >= GOOD_T)],
                              ["photos_poor_vs_stated_good", "photos_good_vs_stated_poor"], "")
    F["mismatch_conf"] = np.select([F["mismatch"] == "photos_poor_vs_stated_good", F["mismatch"] == "photos_good_vs_stated_poor"],
                                   [p_poor, p_good], np.nan)
    # ads whose photos are mostly plans/renders/exteriors (off-plan sales) look like "shell" to the probe:
    # a flag is "reliable" only if at least 40 % of the ad's photos are interiors (CLIP zero-shot)
    F["mismatch_reliable"] = ((F["mismatch"] != "") & (F["clip_interior_share"] >= 0.4)).astype(int)
    F.round(5).to_csv(ph.OUT / "features_photo.csv", index=False)

    mm = F[F["mismatch"] != ""].sort_values("mismatch_conf", ascending=False)
    mm = mm.assign(url=["https://krisha.kz/a/show/%d" % i for i in mm["id"]])
    cols = ["id", "split", "stated_class", "photo_class", "mismatch", "mismatch_conf", "mismatch_reliable"] + [f"photo_p_{c}" for c in ph.CLASSES] + \
           ["clip_mean_shell", "clip_mean_modern", "clip_interior_share", "n_photos_ok", "url"]
    mm[cols].round(3).to_csv(ph.OUT / "mismatch_examples.csv", index=False)
    labelled = F["stated_class"].notna() & (F["has_photo"] == 1)
    summary = {
        "probe": probe, "oof_train_metrics": {k: v for k, v in oof_metrics.items()},
        "thresholds": {"poor_vs_stated_good": POOR_T, "good_vs_stated_poor": GOOD_T},
        "labelled_ads_with_photos": {s: int((labelled & (F["split"] == s)).sum()) for s in ("train", "test")},
        "counts": {s: F.loc[F["split"] == s, "mismatch"].replace("", "none").value_counts().to_dict() for s in ("train", "test")},
        "by_stated_class_train": mm[mm["split"] == "train"].groupby(["mismatch", "stated_class"]).size().rename("n").reset_index().to_dict("records"),
        "reliable_counts_train": mm[(mm["split"] == "train") & (mm["mismatch_reliable"] == 1)]["mismatch"].value_counts().to_dict(),
        "top_ids": {k: g["id"].head(10).tolist() for k, g in mm[mm["split"] == "train"].groupby("mismatch")},
        "base_residual_check_train": residual_check(F),
    }
    json.dump(summary, open(ph.OUT / "mismatch_summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))
    plot_examples(mm[mm["split"] == "train"], ph.OUT / "examples_mismatch.png")


def residual_check(F):
    """Do flagged training ads have unusual Endterm OOF log residuals (log price - log prediction)?
    Compared with unflagged ads of the same stated group; Mann-Whitney U. Training rows only."""
    from scipy.stats import mannwhitneyu
    b = pd.read_csv(core.OUTPUT_DIR / "core" / "base_oof.csv")[["id", "log_resid"]].merge(F, on="id")
    out = {}
    for kind, stated in (("photos_poor_vs_stated_good", ["fresh", "good"]), ("photos_good_vs_stated_poor", ["needs_repair", "shell"])):
        a = b.loc[b["mismatch"] == kind, "log_resid"]
        c = b.loc[(b["mismatch"].fillna("") == "") & b["stated_class"].isin(stated), "log_resid"]
        out[kind] = {"n": int(len(a)), "mean_log_resid": float(a.mean()), "median_log_resid": float(a.median()),
                     "control_n": int(len(c)), "control_mean_log_resid": float(c.mean()),
                     "mannwhitney_p": float(mannwhitneyu(a, c).pvalue) if len(a) else None}
    return out


def plot_examples(mm, path, per_kind=3):
    """Two most interior photos of the top mismatches of each kind (local figure, photos not redistributed)."""
    imgs, idx = ph.load_image_cache()
    zs = pd.read_csv(ph.OUT / "clip_zero_shot_photo.csv")
    p_int = zs[[f"room_{r}" for r in ph.INTERIOR_ROOMS]].sum(axis=1).values
    ex = pd.concat([g.head(per_kind) for _, g in mm.groupby("mismatch")])
    if ex.empty:
        return
    fig, axes = plt.subplots(len(ex), 2, figsize=(5.2, 2.7 * len(ex)))
    axes = np.atleast_2d(axes)
    for r, (_, e) in enumerate(ex.iterrows()):
        rows = idx.index[idx["ad_id"] == e["id"]].values
        rows = rows[np.argsort(-p_int[rows])][:2]
        for c in range(2):
            ax = axes[r, c]; ax.axis("off")
            if c < len(rows):
                ax.imshow(imgs[idx.loc[rows[c], "row"]])
        axes[r, 0].set_title(f"ad {e['id']}: stated {e['stated_class']}, photos {e['photo_class']} "
                             f"(conf {e['mismatch_conf']:.2f})", fontsize=7, loc="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.suptitle("Photo vs stated condition mismatches (training ads, OOF probe)", fontsize=8, y=1 - 0.08 / fig.get_figheight())
    fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
