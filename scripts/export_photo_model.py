"""Export the app's photo-condition model as a committed artifact.

    .venv/bin/python scripts/export_photo_model.py

Fits the logistic probe on the per-photo CLIP ViT-B/32 embeddings of the training ads with a stated
condition (cache/clip_emb.npz, written by scripts/photo_01_embed.py; the photos themselves stay local)
and writes models/photo_condition.joblib, which app/predict.py loads instead of refitting from the
cache. The file holds only the probe (a few hundred KB), no photos and no embeddings.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from krisha_ml import core  # noqa: E402
from app.predict import PHOTO_CLASSES, Predictor, fit_clip_probe  # noqa: E402

OUT = ROOT / "models" / "photo_condition.joblib"


def main() -> int:
    cache = ROOT / "cache" / "clip_emb.npz"
    if not cache.exists():
        print(f"missing {cache}: run scripts/cache_photos.py and scripts/photo_01_embed.py first", file=sys.stderr)
        return 1
    train = core.load_train()
    probe = fit_clip_probe(train.ids, train.df, ROOT)
    z = np.load(cache)
    bundle = {"encoder": "clip", "model": probe, "classes": list(PHOTO_CLASSES),
              "source": "CLIP ViT-B/32 (laion2b_s34b_b79k) image embeddings + StandardScaler + LogisticRegression(C=0.5, balanced)",
              "n_training_photos": int(np.isin(z["ad_id"], train.ids).sum()), "seed": core.SEED,
              "created": dt.datetime.now().isoformat(timespec="seconds")}
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, OUT)
    status = Predictor(root=ROOT).status()["photos"]
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} KB, {bundle['n_training_photos']} training photos); app status: {status}")
    return 0 if "artifact" in status else 2


if __name__ == "__main__":
    raise SystemExit(main())
