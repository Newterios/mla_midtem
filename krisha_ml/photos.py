"""Photo track (lecture 9): CLIP zero-shot, frozen CNN features, fine-tuned EfficientNet-B0,
Grad-CAM, and per-ad photo features for the price model.

Pipeline (scripts/photo_*.py drive it, outputs go to outputs/photos/, caches to cache/):

1. ``build_image_cache``: every usable cached photo (data/photos/{id}_{k}.jpg, shorter side 256 px)
   is decoded once, resized so the shorter side is 224 px (bicubic) and centre-cropped to 224 x 224.
   The uint8 array lives in a memory map ``cache/photo_u8_224.npy`` (N, 224, 224, 3) with an index
   ``cache/photo_index.csv`` (row, ad_id, k). Every model below reads from it, so JPEG decoding
   happens once.
2. ``embed_images``: frozen image embeddings (CLIP ViT-B-32 laion2b, EfficientNet-B0, MobileNetV3-small)
   on MPS, cached in ``cache/<name>_emb.npz`` (``cache/clip_emb.npz`` for CLIP).
3. ``clip_zero_shot``: per-photo room type (8 classes) and renovation quality (6 Russian-market
   classes) from prompt ensembles, aggregated per ad by ``aggregate_zero_shot``.
4. Condition classifier on the stated renovation field (4 classes, ``condition_labels``): logistic
   regression on frozen ad-level embeddings, or ``finetune_effnet`` (photo-level training, ad-level
   prediction = mean of photo probabilities).
5. ``gradcam``: class-activation maps of the fine-tuned network.

Ad ids are int64 everywhere. Photos are never written anywhere outside data/photos/ and cache/.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .core import CACHE_DIR, OUTPUT_DIR, ROOT, SEED

PHOTO_DIR = ROOT / "data" / "photos"
MANIFEST = PHOTO_DIR / "manifest.csv"
HASHES = CACHE_DIR / "photo_hashes.csv"
IMG_CACHE = CACHE_DIR / "photo_u8_224.npy"
IMG_INDEX = CACHE_DIR / "photo_index.csv"
OUT = OUTPUT_DIR / "photos"
SIZE = 224

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

# ----------------------------------------------------------------------------- condition labels
# Stated renovation (clean column ``condition``, = raw info.flat.renovation) collapsed to 4 classes.
# "free_layout" (свободная планировка) is sold without finishing in new buildings -> shell.
CLASSES = ["fresh", "good", "needs_repair", "shell"]
CONDITION_TO_CLASS = {"fresh_renovation": "fresh", "good": "good", "needs_repair": "needs_repair",
                      "rough": "shell", "free_layout": "shell"}
CLASS_RU = {"fresh": "свежий ремонт", "good": "не новый, но аккуратный", "needs_repair": "требует ремонта",
            "shell": "черновая / свободная планировка"}


def condition_labels(df: pd.DataFrame) -> pd.Series:
    """4-class label index (0..3) per row of a clean frame; -1 where the ad states no condition."""
    cls = df["condition"].map(CONDITION_TO_CLASS)
    return cls.map({c: i for i, c in enumerate(CLASSES)}).fillna(-1).astype(int)


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- image cache
def photo_index(ad_ids=None) -> pd.DataFrame:
    """Usable cached photos (status ok/cached, file present): columns ad_id (int64), k, path."""
    m = pd.read_csv(MANIFEST, dtype={"ad_id": str})
    m = m[m["status"].isin(["ok", "cached"])].copy()
    m["ad_id"] = m["ad_id"].astype("int64")
    if ad_ids is not None:
        m = m[m["ad_id"].isin(set(map(int, ad_ids)))]
    m["path"] = [str(PHOTO_DIR / f"{a}_{k}.jpg") for a, k in zip(m["ad_id"], m["k"])]
    m = m[[Path(p).exists() for p in m["path"]]]
    return m.sort_values(["ad_id", "k"]).reset_index(drop=True)[["ad_id", "k", "path"]]


def _load_square(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = SIZE / min(w, h)
        im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - SIZE) // 2, (h - SIZE) // 2
        return np.asarray(im.crop((l, t, l + SIZE, t + SIZE)), dtype=np.uint8)


def build_image_cache(ad_ids, workers: int = 6, force: bool = False) -> pd.DataFrame:
    """Decode every usable photo of ``ad_ids`` once into the uint8 memmap; returns the index."""
    idx = photo_index(ad_ids)
    if not force and IMG_CACHE.exists() and IMG_INDEX.exists():
        old = pd.read_csv(IMG_INDEX)
        if len(old) == len(idx) and (old[["ad_id", "k"]].values == idx[["ad_id", "k"]].values).all():
            return old
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = IMG_CACHE.with_name(f"photo_u8_224.{os.getpid()}.tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(idx), SIZE, SIZE, 3))
    with ThreadPoolExecutor(workers) as ex:
        for i, img in enumerate(ex.map(_load_square, idx["path"], chunksize=64)):
            arr[i] = img
    arr.flush(); del arr
    os.replace(tmp, IMG_CACHE)
    out = idx.drop(columns="path").assign(row=np.arange(len(idx)))[["row", "ad_id", "k"]]
    out.to_csv(IMG_INDEX, index=False)
    return out


def load_image_cache() -> tuple[np.ndarray, pd.DataFrame]:
    """(memmap uint8 (N,224,224,3), index frame row/ad_id/k)."""
    return np.load(IMG_CACHE, mmap_mode="r"), pd.read_csv(IMG_INDEX)


def to_tensor(batch_u8: np.ndarray, mean, std, dev):
    """uint8 (B,H,W,3) -> normalised float tensor (B,3,H,W) on ``dev``."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(batch_u8)).to(dev).permute(0, 3, 1, 2).float().div_(255)
    m = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    return ((x - m) / s).contiguous()


# ----------------------------------------------------------------------------- frozen encoders
def _encoder(name: str):
    """(callable image tensor -> embedding, mean, std) for a frozen encoder, in eval mode on MPS."""
    import torch
    dev = device()
    if name == "clip":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        model = model.eval().to(dev)
        return (lambda x: model.encode_image(x)), CLIP_MEAN, CLIP_STD
    if name == "effnet_b0":
        import timm
        model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=0).eval().to(dev)
        return model, IMAGENET_MEAN, IMAGENET_STD
    if name == "mnv3":
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model = torch.nn.Sequential(net.features, net.avgpool, torch.nn.Flatten()).eval().to(dev)
        return model, IMAGENET_MEAN, IMAGENET_STD
    raise ValueError(name)


def embed_images(name: str, batch: int = 128, force: bool = False) -> dict[str, np.ndarray]:
    """Frozen per-photo embeddings (float16, L2-normalised) for every row of the image cache.

    Cached in cache/<name>_emb.npz with arrays ``emb`` (N, d), ``ad_id``, ``k``."""
    import torch
    path = CACHE_DIR / f"{name}_emb.npz"
    imgs, idx = load_image_cache()
    if path.exists() and not force:
        z = dict(np.load(path))
        if len(z["ad_id"]) == len(idx) and (z["ad_id"] == idx["ad_id"].values).all():
            return z
    fn, mean, std = _encoder(name)
    dev = device()
    out = []
    with torch.inference_mode():
        for s in range(0, len(idx), batch):
            e = fn(to_tensor(imgs[s:s + batch], mean, std, dev)).float()
            out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy().astype(np.float16))
    z = {"emb": np.concatenate(out), "ad_id": idx["ad_id"].values.astype(np.int64), "k": idx["k"].values}
    _atomic_save_npz(path, **z)
    return z


# ----------------------------------------------------------------------------- CLIP zero-shot
ROOM_PROMPTS = {
    "kitchen": ["a photo of a kitchen in an apartment", "a kitchen with a stove, sink and cabinets"],
    "bathroom": ["a photo of a bathroom", "a bathroom with a toilet, sink and bathtub or shower"],
    "bedroom": ["a photo of a bedroom with a bed", "a bedroom in an apartment"],
    "living_room": ["a photo of a living room with a sofa", "a photo of an empty room in an apartment"],
    "exterior": ["a photo of an apartment building exterior", "a photo of a residential complex and its courtyard"],
    "floor_plan": ["a floor plan of an apartment", "an architectural floor plan drawing with room dimensions"],
    "window_view": ["a view of the city from an apartment window", "a view from a high balcony over the city"],
    "corridor": ["a photo of an apartment entrance hallway", "a corridor in an apartment with a front door"],
}
INTERIOR_ROOMS = ["kitchen", "bathroom", "bedroom", "living_room", "corridor"]
# Russian-market renovation grades (евроремонт / современный / советский / требует ремонта / черновая / предчистовая)
QUALITY_PROMPTS = {
    "designer": ["a photo of a luxurious apartment interior with designer euro renovation",
                 "a stylish expensive apartment interior with premium designer finishing"],
    "modern": ["a photo of an apartment with fresh modern renovation",
               "a clean bright apartment interior with new renovation and new furniture"],
    "soviet": ["a photo of an old soviet-style apartment interior with outdated wallpaper and old furniture",
               "an old-fashioned worn apartment interior from the 1980s with a carpet on the wall"],
    "needs_repair": ["a photo of a shabby apartment that needs repair with damaged walls and floors",
                     "a run-down apartment interior in very poor condition with peeling wallpaper"],
    "shell": ["a photo of an unfinished apartment with bare grey concrete walls and floor",
              "a bare concrete shell apartment in a new building without any finishing"],
    "prefinish": ["a photo of an empty apartment with plastered white walls and a concrete screed floor, ready for finishing",
                  "an empty new apartment with smooth plastered walls and no flooring, no furniture"],
}
QUALITY_SCORE = {"designer": 2.0, "modern": 1.0, "soviet": -1.0, "needs_repair": -2.0, "shell": -1.5, "prefinish": -1.0}


def _text_classes(model, tokenizer, prompts: dict[str, list[str]], dev) -> np.ndarray:
    import torch
    rows = []
    with torch.inference_mode():
        for texts in prompts.values():
            t = model.encode_text(tokenizer(texts).to(dev)).float()
            t = torch.nn.functional.normalize(t, dim=1).mean(0)
            rows.append(torch.nn.functional.normalize(t, dim=0).cpu().numpy())
    return np.stack(rows)


def clip_zero_shot(emb: np.ndarray) -> pd.DataFrame:
    """Per-photo zero-shot probabilities: room_<r> (softmax over 8 rooms) and q_<g> (softmax over 6 grades).

    Uses CLIP's logit scale (100) on cosine similarities; prompt ensembles are averaged in text space."""
    import open_clip
    import torch
    dev = device()
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.eval().to(dev)
    tok = open_clip.get_tokenizer("ViT-B-32")
    e = emb.astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    scale = float(model.logit_scale.exp().item())
    out = {}
    for prefix, prompts in (("room", ROOM_PROMPTS), ("q", QUALITY_PROMPTS)):
        T = _text_classes(model, tok, prompts, dev)
        logits = torch.from_numpy(scale * e @ T.T)
        p = torch.softmax(logits, dim=1).numpy()
        for j, name in enumerate(prompts):
            out[f"{prefix}_{name}"] = p[:, j]
    return pd.DataFrame(out)


def aggregate_zero_shot(per_photo: pd.DataFrame, ad_id: np.ndarray) -> pd.DataFrame:
    """Per-ad CLIP features: mean room shares, share of interior photos, interior-weighted mean and max
    of each quality grade, a scalar quality index, and the number of usable photos."""
    rooms = [f"room_{r}" for r in ROOM_PROMPTS]
    qual = [f"q_{g}" for g in QUALITY_PROMPTS]
    d = per_photo.copy()
    d["ad_id"] = ad_id
    d["p_interior"] = d[[f"room_{r}" for r in INTERIOR_ROOMS]].sum(axis=1)
    d["is_interior"] = (d["p_interior"] > 0.5).astype(float)
    w = d["p_interior"].values[:, None]
    wq = pd.DataFrame(d[qual].values * w, columns=qual).assign(ad_id=ad_id, w=w[:, 0]).groupby("ad_id").sum()
    q_mean = wq[qual].div(wq["w"].clip(lower=1e-6), axis=0)
    g = d.groupby("ad_id")
    out = pd.concat([
        g[rooms].mean().add_prefix("clip_share_"),
        g["is_interior"].mean().rename("clip_interior_share"),
        q_mean.add_prefix("clip_mean_"),
        d.loc[d["is_interior"] > 0].groupby("ad_id")[qual].max().add_prefix("clip_max_"),
        g.size().rename("n_photos_ok"),
    ], axis=1)
    out.columns = [c.replace("clip_share_room_", "clip_share_").replace("_q_", "_") for c in out.columns]
    out["clip_quality_index"] = sum(out[f"clip_mean_{g_}"] * s for g_, s in QUALITY_SCORE.items())
    out.index.name = "id"
    return out.reset_index()


def ad_mean_embedding(z: dict[str, np.ndarray], ids, weights: np.ndarray | None = None) -> np.ndarray:
    """Ad-level embedding = (optionally weighted) mean of its photo embeddings, L2-normalised;
    rows of ads without photos are NaN. Order follows ``ids``."""
    emb = z["emb"].astype(np.float32)
    w = np.ones(len(emb), np.float32) if weights is None else np.asarray(weights, np.float32) + 1e-3
    df = pd.DataFrame(emb * w[:, None]).assign(ad_id=z["ad_id"], _w=w)
    s = df.groupby("ad_id").sum()
    m = s.drop(columns="_w").div(s["_w"], axis=0)
    m = m.reindex(np.asarray(ids, dtype=np.int64)).to_numpy(np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n > 0, n, 1)


# ----------------------------------------------------------------------------- duplicate groups
def duplicate_groups(ad_ids) -> np.ndarray:
    """Group label per ad: ads sharing a byte-identical photo or an exact pHash are one group
    (relisted flats / agency stock photos), so classifier holdouts do not leak duplicates."""
    ad_ids = np.asarray(ad_ids, dtype=np.int64)
    parent = {a: a for a in ad_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    h = pd.read_csv(HASHES, dtype={"ad_id": str, "phash": str})
    h["ad_id"] = h["ad_id"].astype("int64")
    m = pd.read_csv(MANIFEST, dtype={"ad_id": str})
    m["ad_id"] = m["ad_id"].astype("int64")
    keys = pd.concat([h[["ad_id", "phash"]].rename(columns={"phash": "key"}),
                      m.loc[m["sha1"].notna(), ["ad_id", "sha1"]].rename(columns={"sha1": "key"})])
    keys = keys[keys["ad_id"].isin(parent)]
    for _, grp in keys.groupby("key")["ad_id"]:
        u = grp.unique()
        for b in u[1:]:
            ra, rb = find(u[0]), find(b)
            if ra != rb:
                parent[rb] = ra
    return np.array([find(a) for a in ad_ids])


# ----------------------------------------------------------------------------- linear probe
def fit_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    """Standardise + multinomial logistic regression with balanced class weights."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                                              random_state=SEED)).fit(X, y)


def class_metrics(y_true, proba: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss
    proba = np.asarray(proba, float) / np.asarray(proba, float).sum(axis=1, keepdims=True)
    pred = proba.argmax(1)
    return {"macro_f1": float(f1_score(y_true, pred, average="macro")), "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
            "log_loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1), labels=range(len(CLASSES)))),
            "per_class_f1": dict(zip(CLASSES, map(float, f1_score(y_true, pred, average=None, labels=range(len(CLASSES)))))),
            "confusion": confusion_matrix(y_true, pred, labels=range(len(CLASSES))).tolist()}


# ----------------------------------------------------------------------------- fine-tuning
@dataclass
class FTConfig:
    epochs: int = 6
    batch: int = 48
    lr: float = 4e-4
    weight_decay: float = 0.02
    label_smoothing: float = 0.1
    drop_rate: float = 0.3
    max_photos: int = 5
    seed: int = SEED


@dataclass
class FTResult:
    model: object
    history: list = field(default_factory=list)


def _augment(x, gen):
    """GPU augmentation on a normalised batch: random resized crop (scale 0.55-1), h-flip,
    brightness/contrast jitter. Per-sample crop, one interpolate per sample (batch is small)."""
    import torch
    import torch.nn.functional as F
    B, _, H, W = x.shape
    out = torch.empty_like(x)
    r = torch.rand(B, 6, generator=gen).numpy()
    for i in range(B):
        scale = 0.55 + 0.45 * r[i, 0]
        h = w = int(H * np.sqrt(scale))
        t, l = int((H - h) * r[i, 1]), int((W - w) * r[i, 2])
        crop = x[i:i + 1, :, t:t + h, l:l + w]
        crop = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
        if r[i, 3] < 0.5:
            crop = crop.flip(3)
        out[i] = crop[0] * (0.8 + 0.4 * r[i, 4]) + (r[i, 5] - 0.5) * 0.4
    return out


def predict_photos(model, imgs: np.ndarray, rows: np.ndarray, batch: int = 128) -> np.ndarray:
    """Softmax probabilities (n_rows, n_classes) of the fine-tuned network for cache rows ``rows``."""
    import torch
    dev = device()
    model.eval()
    out = []
    order = np.argsort(rows)            # sequential memmap reads
    with torch.inference_mode():
        for s in range(0, len(rows), batch):
            r = rows[order[s:s + batch]]
            out.append(torch.softmax(model(to_tensor(imgs[r], IMAGENET_MEAN, IMAGENET_STD, dev)), 1).cpu().numpy())
    p = np.concatenate(out) if out else np.zeros((0, len(CLASSES)))
    res = np.empty_like(p)
    res[order] = p
    return res


def ad_proba(photo_proba: np.ndarray, photo_ad: np.ndarray, ids, weights: np.ndarray | None = None) -> np.ndarray:
    """Ad-level probability = (optionally weighted) mean of its photos' probabilities; NaN rows for ads
    without photos."""
    w = np.ones(len(photo_proba)) if weights is None else np.asarray(weights, float) + 1e-3
    s = pd.DataFrame(photo_proba * w[:, None]).assign(ad_id=photo_ad, _w=w).groupby("ad_id").sum()
    df = s.drop(columns="_w").div(s["_w"], axis=0)
    return df.reindex(np.asarray(ids, dtype=np.int64)).to_numpy()


def finetune_effnet(train_ads, train_y, val_ads=None, val_y=None, cfg: FTConfig = FTConfig(), log=print,
                    row_weight: np.ndarray | None = None) -> FTResult:
    """Fine-tune timm EfficientNet-B0 (ImageNet weights) on photos labelled with their ad's condition.

    Photo-level cross-entropy (balanced class weights, label smoothing), AdamW + cosine schedule with
    1 warm-up epoch, GPU augmentation. If ``val_ads`` is given, every epoch logs train loss, val
    photo loss and ad-level macro-F1 / accuracy (ad prediction = mean photo probability).

    ``row_weight`` (per image-cache row, e.g. CLIP p(interior)): if given, only photos with weight > 0.5
    are trained on and the ad prediction is the weight-weighted mean of photo probabilities."""
    import timm
    import torch
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = device()
    imgs, idx = load_image_cache()
    idx = idx[idx["k"] <= cfg.max_photos]
    lab = dict(zip(np.asarray(train_ads, np.int64), np.asarray(train_y)))
    tr = idx[idx["ad_id"].isin(lab)]
    if row_weight is not None:
        tr = tr[row_weight[tr["row"].values] > 0.5]
    tr_rows, tr_lab = tr["row"].values, tr["ad_id"].map(lab).values.astype(np.int64)
    counts = np.bincount(tr_lab, minlength=len(CLASSES)).astype(float)
    w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=len(CLASSES),
                              drop_rate=cfg.drop_rate)
    # timm's Google-style init of a 4-way head (uniform +-1/sqrt(4)) gives initial CE ~4.3 and large
    # gradients that damage the pretrained backbone in the first epoch; a zero head starts at ln(4).
    torch.nn.init.zeros_(model.classifier.weight); torch.nn.init.zeros_(model.classifier.bias)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = int(np.ceil(len(tr_rows) / cfg.batch))
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / steps_per_epoch) *
                                              0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    lossf = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if val_ads is not None:
        vlab = dict(zip(np.asarray(val_ads, np.int64), np.asarray(val_y)))
        va = idx[idx["ad_id"].isin(vlab)]
    history = []
    for ep in range(cfg.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        perm = rng.permutation(len(tr_rows))
        for s in range(0, len(perm), cfg.batch):
            b = perm[s:s + cfg.batch]
            rows = tr_rows[b]
            o = np.argsort(rows)
            x = _augment(to_tensor(imgs[rows[o]], IMAGENET_MEAN, IMAGENET_STD, dev), gen)
            yb = torch.from_numpy(tr_lab[b][o]).to(dev)
            loss = lossf(model(x), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += float(loss.item()) * len(b); n += len(b)
        rec = {"epoch": ep + 1, "train_loss": tot / n, "sec": round(time.time() - t0, 1)}
        if val_ads is not None:
            pp = predict_photos(model, imgs, va["row"].values)
            yv = va["ad_id"].map(vlab).values
            rec["val_photo_loss"] = float(-np.log(np.clip(pp[np.arange(len(yv)), yv], 1e-7, 1)).mean())
            ids_v = np.array(sorted(vlab))
            pa = ad_proba(pp, va["ad_id"].values, ids_v, None if row_weight is None else row_weight[va["row"].values])
            ok = ~np.isnan(pa).any(1)
            m = class_metrics(np.array([vlab[a] for a in ids_v])[ok], pa[ok])
            rec.update(val_macro_f1=m["macro_f1"], val_accuracy=m["accuracy"])
        history.append(rec)
        log(json.dumps(rec))
    return FTResult(model, history)


# ----------------------------------------------------------------------------- Grad-CAM
def gradcam(model, x, target: int | None = None, layer=None) -> tuple[np.ndarray, int, np.ndarray]:
    """Grad-CAM (Selvaraju et al. 2017) for one normalised image tensor (1,3,H,W).

    Returns (heat map in [0,1] at input resolution, explained class, softmax probabilities).
    ``layer`` defaults to the last conv block (``conv_head``) of a timm EfficientNet."""
    import torch
    import torch.nn.functional as F
    layer = layer or model.conv_head
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.eval()
        x = x.clone().requires_grad_(True)
        logits = model(x)
        prob = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        c = int(prob.argmax()) if target is None else int(target)
        model.zero_grad(set_to_none=True)
        logits[0, c].backward()
        a, g = acts["a"].detach(), grads["g"].detach()
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, c, prob
    finally:
        h1.remove(); h2.remove()


# ----------------------------------------------------------------------------- per-ad tables
def interior_weights() -> np.ndarray:
    """CLIP zero-shot p(interior) per cached photo (row order of the image cache)."""
    zs = pd.read_csv(OUT / "clip_zero_shot_photo.csv")
    _, idx = load_image_cache()
    if not (zs["ad_id"].values == idx["ad_id"].values).all():
        raise ValueError("clip_zero_shot_photo.csv is out of sync with the image cache; rerun photo_01_embed.py")
    return zs[[f"room_{r}" for r in INTERIOR_ROOMS]].sum(axis=1).values


def ad_embedding_table(name: str, pool: str | None = None) -> pd.DataFrame:
    """Ad-level embedding (rows = ads with photos, index = id): mean of photo embeddings, or
    interior-weighted mean (``pool='interior'``). ``name='legacy_mnv3'`` reads final/photo_embeddings.npz."""
    if name == "legacy_mnv3":
        z = np.load(ROOT / "final" / "photo_embeddings.npz")
        ok = z["valid_photos"] > 0
        return pd.DataFrame(z["features"][ok], index=pd.Index(z["ids"][ok].astype(np.int64), name="id"))
    z = embed_images(name)
    ids = np.unique(z["ad_id"])
    X = ad_mean_embedding(z, ids, weights=interior_weights() if pool == "interior" else None)
    return pd.DataFrame(X, index=pd.Index(ids, name="id"))


def lookup(table: pd.DataFrame, ids) -> np.ndarray:
    """Rows of an id-indexed table in the order of ``ids`` (NaN for missing ads)."""
    return table.reindex(np.asarray(ids, dtype=np.int64)).to_numpy(np.float32)


def probe_oof(X: np.ndarray, y: np.ndarray, folds, C: float) -> np.ndarray:
    """Out-of-fold class probabilities of the linear probe. Rows with y < 0 (no stated condition) or
    no photos (NaN X) are never trained on; every row with photos gets a prediction from the model of
    the fold where it is in validation. Rows without photos stay NaN."""
    out = np.full((len(y), len(CLASSES)), np.nan)
    has = ~np.isnan(X).any(1)
    for tr, va in folds:
        t = tr[(y[tr] >= 0) & has[tr]]
        v = va[has[va]]
        if len(v):
            out[v] = fit_probe(X[t], y[t], C).predict_proba(X[v])
    return out


def probe_fit_predict(X_tr, y_tr, X_new, C: float) -> np.ndarray:
    """Fit the probe on all labelled training rows with photos, predict ``X_new`` (NaN rows stay NaN)."""
    out = np.full((len(X_new), len(CLASSES)), np.nan)
    has_tr = (~np.isnan(X_tr).any(1)) & (y_tr >= 0)
    has_new = ~np.isnan(X_new).any(1)
    if has_new.any():
        out[has_new] = fit_probe(X_tr[has_tr], y_tr[has_tr], C).predict_proba(X_new[has_new])
    return out
