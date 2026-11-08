"""Photo track step 2: condition classifier comparison on TRAINING ads only.

Label = stated renovation (4 classes, krisha_ml.photos.CLASSES). Only labelled training ads with at
least one usable photo are used; test ads are never touched. Validation = StratifiedGroupKFold(5,
shuffle, seed 42) over those ads with groups = duplicate-photo clusters (shared byte-identical photo
or exact pHash), so relisted flats never sit on both sides.

  (a) frozen MobileNetV3-small / EfficientNet-B0 ad-mean embedding + logistic regression (5-fold)
  (c) CLIP ViT-B-32 ad-mean embedding + logistic regression (linear probe, 5-fold)
      (+ interior-weighted pooling with CLIP zero-shot p(interior) as photo weights)
  (b) fine-tuned EfficientNet-B0: fold 0 of the same split only (cost), training curves logged.

Usage: python scripts/photo_02_classify.py [--skip-finetune] [--epochs 6]
Outputs: outputs/photos/classifier_comparison.csv/.json, confusion_<best>.png, training_curves.png,
         cache/effnet_ft_fold0.pt
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
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from krisha_ml import core  # noqa: E402
from krisha_ml import photos as ph  # noqa: E402

C_GRID = (0.003, 0.01, 0.03, 0.1)


def labelled_train_ads():
    tr = core.load_train()
    y = ph.condition_labels(tr.df).values
    _, idx = ph.load_image_cache()
    has = np.isin(tr.ids, idx["ad_id"].unique())
    keep = (y >= 0) & has
    return tr.ids[keep], y[keep]


def splits(ids, y):
    groups = ph.duplicate_groups(ids)
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=core.SEED).split(ids, y, groups))


def probe_cv(X, y, folds, C):
    oof = np.zeros((len(y), len(ph.CLASSES)))
    per_fold = []
    for tr, va in folds:
        oof[va] = ph.fit_probe(X[tr], y[tr], C).predict_proba(X[va])
        per_fold.append(ph.class_metrics(y[va], oof[va])["macro_f1"])
    return oof, np.array(per_fold)


def plot_confusion(cm, title, path):
    cm = np.asarray(cm, float)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.55 else "black", fontsize=9)
    ax.set_xticks(range(len(cm)), ph.CLASSES, rotation=20)
    ax.set_yticks(range(len(cm)), ph.CLASSES)
    ax.set_xlabel("predicted condition (from photos)")
    ax.set_ylabel("stated condition (ad field)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row share")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_curves(history, path):
    h = pd.DataFrame(history)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(h["epoch"], h["train_loss"], "o-", label="train (augmented, smoothed CE)")
    a1.plot(h["epoch"], h["val_photo_loss"], "s-", label="validation (photo CE)")
    a1.set_xlabel("epoch"); a1.set_ylabel("cross-entropy loss"); a1.legend(); a1.set_title("Loss")
    a2.plot(h["epoch"], h["val_macro_f1"], "o-", label="val macro-F1 (ad level)")
    a2.plot(h["epoch"], h["val_accuracy"], "s-", label="val accuracy (ad level)")
    a2.set_xlabel("epoch"); a2.set_ylabel("score"); a2.set_ylim(0, 1); a2.legend(); a2.set_title("Validation")
    fig.suptitle("Fine-tuned EfficientNet-B0, condition from photos (fold 0 of training ads)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--interior-only", action="store_true",
                    help="train on photos with CLIP p(interior) > 0.5, interior-weighted ad aggregation")
    ap.add_argument("--only-finetune", action="store_true", help="append a fine-tune run to the existing table")
    ap.add_argument("--tag", default="", help="suffix for fine-tune outputs, e.g. _v2")
    args = ap.parse_args()
    ph.OUT.mkdir(parents=True, exist_ok=True)
    ids, y = labelled_train_ads()
    folds = splits(ids, y)
    print(f"labelled training ads with photos: {len(ids)}, classes {np.bincount(y).tolist()}", flush=True)

    zs = pd.read_csv(ph.OUT / "clip_zero_shot_photo.csv")
    p_int = zs[[f"room_{r}" for r in ph.INTERIOR_ROOMS]].sum(axis=1).values
    rows, results, oofs = [], {}, {}
    if args.only_finetune:
        rows = pd.read_csv(ph.OUT / "classifier_comparison.csv").to_dict("records")
        results = json.load(open(ph.OUT / "classifier_comparison.json"))["results"]
    variants = [] if args.only_finetune else [("mnv3", None), ("effnet_b0", None), ("clip", None), ("clip", "interior"), ("effnet_b0", "interior")]
    for enc, pool in variants:
        z = ph.embed_images(enc)
        X = ph.ad_mean_embedding(z, ids, weights=p_int if pool else None)
        best = None
        for C in C_GRID:   # C chosen by mean fold macro-F1 (reported as such: mild optimism, same for all)
            oof, f = probe_cv(X, y, folds, C)
            if best is None or f.mean() > best[2].mean():
                best = (C, oof, f)
        C, oof, f = best
        name = f"{'(c) CLIP linear probe' if enc == 'clip' else '(a) frozen ' + enc + ' + LR'}{' [interior-weighted]' if pool else ''}"
        m = ph.class_metrics(y, oof)
        f0 = ph.class_metrics(y[folds[0][1]], oof[folds[0][1]])
        rows.append({"model": name, "C": C, "macro_f1_cv": f.mean(), "macro_f1_sd": f.std(), "accuracy_cv": m["accuracy"],
                     "balanced_acc": m["balanced_acc"], "log_loss": m["log_loss"], "macro_f1_fold0": f0["macro_f1"],
                     "accuracy_fold0": f0["accuracy"], **{f"f1_{k}": v for k, v in m["per_class_f1"].items()}})
        results[name] = m; oofs[name] = oof
        print(json.dumps(rows[-1], default=float), flush=True)

    if not args.skip_finetune:
        import torch
        tr, va = folds[0]
        cfg = ph.FTConfig(epochs=args.epochs, lr=args.lr)
        rw = p_int if args.interior_only else None
        res = ph.finetune_effnet(ids[tr], y[tr], ids[va], y[va], cfg=cfg, log=lambda s: print(s, flush=True), row_weight=rw)
        tag = args.tag
        torch.save(res.model.state_dict(), core.CACHE_DIR / f"effnet_ft_fold0{tag}.pt")
        pd.DataFrame(res.history).to_csv(ph.OUT / f"finetune_history_fold0{tag}.csv", index=False)
        plot_curves(res.history, ph.OUT / f"training_curves{tag}.png")
        imgs, idx = ph.load_image_cache()
        vi = idx[idx["ad_id"].isin(ids[va])]
        pa = ph.ad_proba(ph.predict_photos(res.model, imgs, vi["row"].values), vi["ad_id"].values, ids[va],
                         None if rw is None else rw[vi["row"].values])
        m = ph.class_metrics(y[va], pa)
        name = f"(b) fine-tuned EfficientNet-B0 (lr {args.lr:g}{', interior photos only' if args.interior_only else ''})"
        rows.append({"model": name, "C": np.nan, "macro_f1_cv": np.nan, "macro_f1_sd": np.nan, "accuracy_cv": np.nan,
                     "balanced_acc": m["balanced_acc"], "log_loss": m["log_loss"], "macro_f1_fold0": m["macro_f1"],
                     "accuracy_fold0": m["accuracy"], **{f"f1_{k}": v for k, v in m["per_class_f1"].items()}})
        results[name + " (fold 0)"] = m
        np.save(core.CACHE_DIR / f"effnet_ft_fold0{tag}_valproba.npy", pa)
        plot_confusion(m["confusion"], f"{name}, fold 0 ({len(va)} ads)\nmacro-F1 {m['macro_f1']:.3f}, acc {m['accuracy']:.3f}",
                       ph.OUT / f"confusion_finetuned_effnet{tag}.png")
        print(json.dumps(rows[-1], default=float), flush=True)

    table = pd.DataFrame(rows)
    table.round(4).to_csv(ph.OUT / "classifier_comparison.csv", index=False)
    best_probe = table.dropna(subset=["macro_f1_cv"]).sort_values("macro_f1_cv").iloc[-1]["model"]
    plot_confusion(results[best_probe]["confusion"],
                   f"{best_probe}\n5-fold grouped CV on {len(ids)} training ads, macro-F1 {results[best_probe]['macro_f1']:.3f}",
                   ph.OUT / "confusion_best_probe.png")
    json.dump({"n_ads": int(len(ids)), "class_counts": dict(zip(ph.CLASSES, map(int, np.bincount(y)))),
               "split": "StratifiedGroupKFold(5, shuffle, 42), groups = duplicate-photo clusters",
               "fold0_val_ads": int(len(folds[0][1])), "best_probe": best_probe, "results": results},
              open(ph.OUT / "classifier_comparison.json", "w"), indent=1, default=float)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
