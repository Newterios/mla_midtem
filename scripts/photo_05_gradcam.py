"""Photo track step 5: Grad-CAM of the fine-tuned EfficientNet-B0 (cache/effnet_ft_fold0_v3.pt by default; argv[1] = run tag) on
validation ads of fold 0 (training ads only): confident correct and confident wrong ad-level predictions.

For each chosen ad the photo with the highest probability for the predicted class is explained.
Outputs (local only, contain krisha photos, gitignored): outputs/photos/gradcam_correct.png,
outputs/photos/gradcam_incorrect.png; outputs/photos/gradcam_examples.csv (ids, classes, probabilities).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402

from krisha_ml import core  # noqa: E402
from krisha_ml import photos as ph  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from photo_02_classify import labelled_train_ads, splits  # noqa: E402


TAG = sys.argv[1] if len(sys.argv) > 1 else "_v3"   # best fine-tuned run (zero-init head, interior photos)


def pick(ids, y, pa, correct: bool, per_class=2):
    pred, conf = pa.argmax(1), pa.max(1)
    ok = (pred == y) if correct else (pred != y)
    out = []
    for c in range(len(ph.CLASSES)):
        m = np.where(ok & ((y if correct else pred) == c))[0]
        out += list(m[np.argsort(-conf[m])][:per_class])
    return out


def figure(model, imgs, idx, ids, y, pa, chosen, title, path):
    dev = ph.device()
    fig, axes = plt.subplots(len(chosen), 2, figsize=(5.4, 2.75 * len(chosen)))
    axes = np.atleast_2d(axes)
    recs = []
    for r, i in enumerate(chosen):
        rows = idx.loc[idx["ad_id"] == ids[i], "row"].values
        pp = ph.predict_photos(model, imgs, rows)
        c = int(pa[i].argmax())
        row = rows[int(pp[:, c].argmax())]
        x = ph.to_tensor(imgs[row:row + 1], ph.IMAGENET_MEAN, ph.IMAGENET_STD, dev)
        cam, _, prob = ph.gradcam(model, x, target=c)
        axes[r, 0].imshow(imgs[row]); axes[r, 1].imshow(imgs[row]); axes[r, 1].imshow(cam, cmap="jet", alpha=0.45)
        for a in axes[r]:
            a.axis("off")
        axes[r, 0].set_title(f"ad {ids[i]}: stated {ph.CLASSES[y[i]]}, predicted {ph.CLASSES[c]} "
                             f"(ad p={pa[i, c]:.2f}, photo p={prob[c]:.2f})", fontsize=7, loc="left")
        recs.append({"id": int(ids[i]), "stated": ph.CLASSES[y[i]], "predicted": ph.CLASSES[c], "ad_p": float(pa[i, c]),
                     "photo_k": int(idx.loc[idx["row"] == row, "k"].iloc[0]), "photo_p": float(prob[c]), "figure": path.name})
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.suptitle(title, fontsize=8, y=1 - 0.08 / fig.get_figheight())
    fig.savefig(path, dpi=130); plt.close(fig)
    return recs


def main():
    ids, y = labelled_train_ads()
    _, va = splits(ids, y)[0]
    ids, y = ids[va], y[va]
    pa = np.load(core.CACHE_DIR / f"effnet_ft_fold0{TAG}_valproba.npy")
    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=False, num_classes=len(ph.CLASSES))
    model.load_state_dict(torch.load(core.CACHE_DIR / f"effnet_ft_fold0{TAG}.pt", map_location="cpu"))
    model = model.to(ph.device()).eval()
    imgs, idx = ph.load_image_cache()
    recs = figure(model, imgs, idx, ids, y, pa, pick(ids, y, pa, True),
                  "Grad-CAM, fine-tuned EfficientNet-B0: confident CORRECT predictions (fold-0 validation ads)",
                  ph.OUT / "gradcam_correct.png")
    recs += figure(model, imgs, idx, ids, y, pa, pick(ids, y, pa, False),
                   "Grad-CAM, fine-tuned EfficientNet-B0: confident WRONG predictions (fold-0 validation ads)",
                   ph.OUT / "gradcam_incorrect.png")
    pd.DataFrame(recs).round(3).to_csv(ph.OUT / "gradcam_examples.csv", index=False)
    print(pd.DataFrame(recs).round(3).to_string())


if __name__ == "__main__":
    main()
