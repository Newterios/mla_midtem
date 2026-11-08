"""Photo track step 1: image cache, frozen embeddings (CLIP, EfficientNet-B0, MobileNetV3-small)
and CLIP zero-shot room/quality scores.

Run from the repo root:  .venv/bin/python scripts/photo_01_embed.py
Outputs: cache/photo_u8_224.npy + cache/photo_index.csv, cache/{clip,effnet_b0,mnv3}_emb.npz,
         outputs/photos/clip_zero_shot_photo.csv (per photo), outputs/photos/clip_zero_shot_ad.csv (per ad).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from krisha_ml import core  # noqa: E402
from krisha_ml import photos as ph  # noqa: E402


def main():
    t = time.time()
    clean = core.load_clean()
    idx = ph.build_image_cache(clean["id"].values)
    print(f"image cache: {len(idx)} photos of {idx['ad_id'].nunique()} ads ({time.time() - t:.0f}s)", flush=True)
    for name in ("clip", "effnet_b0", "mnv3"):
        t = time.time()
        z = ph.embed_images(name)
        print(f"{name}: {z['emb'].shape} ({time.time() - t:.0f}s)", flush=True)
    z = ph.embed_images("clip")
    zs = ph.clip_zero_shot(z["emb"])
    ph.OUT.mkdir(parents=True, exist_ok=True)
    zs.insert(0, "k", z["k"]); zs.insert(0, "ad_id", z["ad_id"])
    zs.round(5).to_csv(ph.OUT / "clip_zero_shot_photo.csv", index=False)
    ad = ph.aggregate_zero_shot(zs.drop(columns=["ad_id", "k"]), z["ad_id"])
    ad.round(5).to_csv(ph.OUT / "clip_zero_shot_ad.csv", index=False)
    rooms = [c for c in zs if c.startswith("room_")]
    print("photo room argmax shares:", zs[rooms].idxmax(axis=1).value_counts(normalize=True).round(3).to_dict())
    print("per-ad zero-shot table:", ad.shape)


if __name__ == "__main__":
    main()
