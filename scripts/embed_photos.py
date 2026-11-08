"""Per-image embeddings of the cached listing photos (near-duplicate search in krisha_ml.dedup).

Reads data/photos/manifest.csv, embeds every usable image (status ok/cached) on MPS if available
and writes cache/<model>_image_emb.npz with
    ad_id (int64, n)    k (int8, n)    emb (float16, n x d, L2-normalised)
Models: mnv3  torchvision MobileNetV3-small ImageNet weights, pooled 576-d features (default,
              the weights are in the local torch cache; krisha_ml.dedup uses this file)
        clip  open_clip ViT-B-32 laion2b_s34b_b79k, 512-d image embedding
The embeddings are static per image (nothing is fitted on the data).

Usage:  python scripts/embed_photos.py [--model mnv3|clip] [--batch 128]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "data" / "photos"


def load_model(name: str):
    """(model returning image features, preprocessing transform)."""
    if name == "clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        return _ClipImage(model), preprocess
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    weights = MobileNet_V3_Small_Weights.DEFAULT
    net = mobilenet_v3_small(weights=weights)
    return torch.nn.Sequential(net.features, net.avgpool), weights.transforms()


class _ClipImage(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, x):
        return self.clip.encode_image(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["mnv3", "clip"], default="mnv3")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    out_path = ROOT / "cache" / f"{args.model}_image_emb.npz"

    torch.manual_seed(42)
    man = pd.read_csv(PHOTOS / "manifest.csv", dtype={"ad_id": str})
    man = man[man["status"].isin(["ok", "cached"])].reset_index(drop=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, preprocess = load_model(args.model)
    model = model.eval().to(device)

    out = None
    t0 = time.time()
    for s in range(0, len(man), args.batch):
        rows = man.iloc[s:s + args.batch]
        imgs = []
        for a, k in zip(rows["ad_id"], rows["k"]):
            with Image.open(PHOTOS / f"{a}_{k}.jpg") as im:
                imgs.append(preprocess(im.convert("RGB")))
        with torch.inference_mode():
            e = model(torch.stack(imgs).to(device)).flatten(1)
            e = torch.nn.functional.normalize(e.float(), dim=1)
        if out is None:
            out = np.zeros((len(man), e.shape[1]), dtype=np.float16)
        out[s:s + len(rows)] = e.cpu().numpy().astype(np.float16)
        if (s // args.batch) % 20 == 0:
            print(f"{s + len(rows)}/{len(man)} images, {time.time() - t0:.0f} s", flush=True)

    out_path.parent.mkdir(exist_ok=True)
    tmp = out_path.with_name(f"{out_path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, ad_id=man["ad_id"].astype("int64").to_numpy(), k=man["k"].astype("int8").to_numpy(), emb=out)
    os.replace(tmp, out_path)
    print(f"done: {len(man)} images in {time.time() - t0:.0f} s -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
