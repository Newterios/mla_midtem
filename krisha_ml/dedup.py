"""Duplicate / re-posted listing detection (Final, leakage audit).

The same flat is often on krisha.kz more than once: re-posted by the owner, or posted by several
agents. If two copies of one flat fall in different CV folds (or one in train, one in test), the
model is scored on a flat it has effectively seen, and the error estimate is optimistic.

Four independent signals give candidate ad pairs; their union is a graph whose connected
components are the duplicate groups:

1. photo hashes  - two images of different ads with pHash Hamming <= PHASH_MAX and dHash Hamming
                   <= DHASH_MAX (64-bit hashes of the cached photos, cache/photo_hashes.csv);
2. photo embeddings - per-image MobileNetV3-small ImageNet embeddings (cache/mnv3_image_emb.npz,
                   scripts/embed_photos.py) with cosine >= EMB_COS (catches crops, re-encodes and
                   colour edits that move the hashes);
3. structured match - same city, rooms, floor, total floors and year, area within 1 %, asking price
                   within 5 %, coordinates within 150 m (same street + house number when a side has
                   no coordinates);
4. description text - character 3-5-gram TF-IDF cosine >= TEXT_COS between descriptions of at least
                   TEXT_MIN_CHARS characters, with the same city and rooms, area within 2 % and the
                   same floor (when both known). TF-IDF is fitted on the descriptions only (no
                   prices), as a matching tool, never as a model feature.

"Stock" images (one picture that shows up in many ads: agency logo, complex render, entrance, a
floor plan used for a whole building) would chain unrelated flats together, so an image cluster
that spans more than HUB_ADS ads is ignored. A photo link also needs the two ads to be plausibly
the same flat (same city; same rooms and area within 5 % if they share only one image, area
within 10 % if they share two or more).

Nothing here uses prices for fitting, and nothing is learned from data: the thresholds are fixed
by inspection of the distance histograms (see outputs/leakage/).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .core import CACHE_DIR, ID, ROOT

HASH_CSV = CACHE_DIR / "photo_hashes.csv"
EMB_NPZ = CACHE_DIR / "mnv3_image_emb.npz"
PHOTO_DIR = ROOT / "data" / "photos"

PHASH_MAX = 8          # Hamming distance, 64-bit pHash
DHASH_MAX = 10         # Hamming distance, 64-bit dHash
EMB_COS = 0.97         # embedding cosine for a near-identical image: 91 % of pairs above it are also
                       # pHash pairs; at 0.93-0.97 most extra pairs are look-alike floor plans of different flats
HUB_ADS = 5            # an image cluster spanning more ads than this is stock, ignored
PHOTO_AREA_TOL_1 = 0.05  # area tolerance (relative) when the ads share exactly one image
PHOTO_AREA_TOL_2 = 0.10  # ... when they share two or more images
STRUCT_AREA_TOL = 0.01
STRUCT_PRICE_TOL = 0.05
STRUCT_DIST_M = 150.0
SAME_FLAT_AREA_TOL = 0.02
TEXT_COS = 0.90
TEXT_MIN_CHARS = 80


# --------------------------------------------------------------------------- image level
def load_hashes(path: Path | str = HASH_CSV) -> pd.DataFrame:
    """One row per cached image: ad_id (int64), k, phash/dhash as uint64."""
    h = pd.read_csv(path, dtype={"phash": str, "dhash": str})
    h["ad_id"] = h["ad_id"].astype("int64")
    for c in ("phash", "dhash"):
        h[c] = np.array([int(x, 16) for x in h[c]], dtype=np.uint64)
    return h


def load_embeddings(path: Path | str = EMB_NPZ) -> pd.DataFrame | None:
    """Image embeddings as a frame (ad_id, k) + a float32 matrix in ``attrs['emb']``; None if absent."""
    path = Path(path)
    if not path.exists():
        return None
    z = np.load(path)
    out = pd.DataFrame({"ad_id": z["ad_id"].astype("int64"), "k": z["k"].astype(int)})
    out.attrs["emb"] = z["emb"].astype(np.float32)
    return out


def hash_image_pairs(h: pd.DataFrame, phash_max: int = PHASH_MAX, dhash_max: int = DHASH_MAX,
                     chunk: int = 2000) -> pd.DataFrame:
    """Cross-ad image pairs (i < j, positions in ``h``) with both Hamming distances under the limits."""
    ph, dh, ad = h["phash"].to_numpy(), h["dhash"].to_numpy(), h["ad_id"].to_numpy()
    parts = []
    for s in range(0, len(h), chunk):
        dp = np.bitwise_count(ph[s:s + chunk, None] ^ ph[None, :])
        dd = np.bitwise_count(dh[s:s + chunk, None] ^ dh[None, :])
        i, j = np.nonzero((dp <= phash_max) & (dd <= dhash_max))
        keep = (j > i + s) & (ad[i + s] != ad[j])
        i, j = i[keep], j[keep]
        parts.append(pd.DataFrame({"i": i + s, "j": j, "phash_d": dp[i, j], "dhash_d": dd[i, j]}))
    return pd.concat(parts, ignore_index=True)


def emb_image_pairs(emb: np.ndarray, ad: np.ndarray, cos_min: float = EMB_COS, chunk: int = 2000) -> pd.DataFrame:
    """Cross-ad image pairs (i < j) with cosine similarity >= cos_min (rows of ``emb`` are unit vectors)."""
    parts = []
    for s in range(0, len(emb), chunk):
        sim = emb[s:s + chunk] @ emb.T
        i, j = np.nonzero(sim >= cos_min)
        keep = (j > i + s) & (ad[i + s] != ad[j])
        i, j = i[keep], j[keep]
        parts.append(pd.DataFrame({"i": i + s, "j": j, "cos": sim[i, j]}))
    return pd.concat(parts, ignore_index=True)


def hub_images(pairs: pd.DataFrame, ad: np.ndarray, hub_ads: int = HUB_ADS) -> np.ndarray:
    """Boolean mask over images: True for images whose match cluster spans more than ``hub_ads`` ads."""
    n = len(ad)
    g = coo_matrix((np.ones(len(pairs)), (pairs["i"], pairs["j"])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    span = pd.Series(ad).groupby(comp).nunique()
    return span.reindex(comp).to_numpy() > hub_ads


def ad_pairs_from_images(pairs: pd.DataFrame, ad: np.ndarray, hub: np.ndarray, signal: str) -> pd.DataFrame:
    """Collapse image pairs to ad pairs (id_a < id_b): number of distinct matched images on the side
    with fewer matches (so one image matching three photos of the other ad counts once)."""
    p = pairs[~hub[pairs["i"]] & ~hub[pairs["j"]]]
    a, b = ad[p["i"]], ad[p["j"]]
    swap = a > b
    ia, ib = np.where(swap, p["j"], p["i"]), np.where(swap, p["i"], p["j"])
    t = pd.DataFrame({"id_a": np.minimum(a, b), "id_b": np.maximum(a, b), "ia": ia, "ib": ib})
    g = t.groupby(["id_a", "id_b"])
    out = pd.concat([g["ia"].nunique(), g["ib"].nunique()], axis=1).min(axis=1).rename(f"n_{signal}")
    return out.reset_index()


# --------------------------------------------------------------------------- structured match
def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _rel_diff(a, b):
    return np.abs(a - b) / np.maximum(np.minimum(a, b), 1e-9)


def structured_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Ad pairs that match on the listing card (see module doc). Missing floor/total_floors match
    only another missing value; missing coordinates fall back to the same street and house number."""
    key = ["city", "rooms", "floor", "total_floors", "year_built"]
    d = df[[ID, *key, "area", "price", "lat", "lon", "street", "house_num"]].copy()
    d[key] = d[key].astype(str)  # NaN -> "nan": missing equals missing
    m = d.merge(d, on=key, suffixes=("_a", "_b"))
    m = m[m[f"{ID}_a"] < m[f"{ID}_b"]]
    ok = (_rel_diff(m["area_a"], m["area_b"]) <= STRUCT_AREA_TOL) & (_rel_diff(m["price_a"], m["price_b"]) <= STRUCT_PRICE_TOL)
    m = m[ok].copy()
    dist = _haversine_m(m["lat_a"], m["lon_a"], m["lat_b"], m["lon_b"])
    both = m[["lat_a", "lat_b"]].notna().all(axis=1)
    same_addr = (m["street_a"].astype(str).str.lower() == m["street_b"].astype(str).str.lower()) & \
                (m["house_num_a"].astype(str).str.lower() == m["house_num_b"].astype(str).str.lower()) & m["street_a"].notna()
    m["dist_m"] = dist
    m = m[np.where(both, dist <= STRUCT_DIST_M, same_addr)]
    return pd.DataFrame({"id_a": m[f"{ID}_a"].to_numpy(), "id_b": m[f"{ID}_b"].to_numpy(),
                         "struct": True, "dist_m": m["dist_m"].round(1).to_numpy()})


# --------------------------------------------------------------------------- description text
def text_pairs(df: pd.DataFrame, cos_min: float = TEXT_COS, min_chars: int = TEXT_MIN_CHARS) -> pd.DataFrame:
    """Ad pairs with near-identical descriptions and matching cards (see module doc, signal 4)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    txt = df["description"].fillna("").str.lower().to_numpy()
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32).fit_transform(txt)
    S = (X @ X.T).tocoo()
    long_ = np.char.str_len(txt.astype(str)) >= min_chars
    m = (S.row < S.col) & (S.data >= cos_min) & long_[S.row] & long_[S.col]
    a = df.iloc[S.row[m]].reset_index(drop=True)
    b = df.iloc[S.col[m]].reset_index(drop=True)
    floor_ok = (a["floor"] == b["floor"]) | a["floor"].isna() | b["floor"].isna()
    ok = ((a["city"] == b["city"]) & (a["rooms"] == b["rooms"]) & floor_ok
          & (_rel_diff(a["area"], b["area"]) <= SAME_FLAT_AREA_TOL)).to_numpy()
    ia, ib = a[ID].to_numpy("int64")[ok], b[ID].to_numpy("int64")[ok]
    return pd.DataFrame({"id_a": np.minimum(ia, ib), "id_b": np.maximum(ia, ib), "text_cos": S.data[m][ok].round(4)})


# --------------------------------------------------------------------------- combine
@dataclass
class DuplicateReport:
    pairs: pd.DataFrame      # one row per linked ad pair with the evidence per signal
    groups: pd.DataFrame     # id, group, group_size  (every ad of ``df``; singletons are their own group)
    image_pairs: pd.DataFrame  # matched image pairs (ad_a, k_a, ad_b, k_b, phash_d, dhash_d, cos, hub)


def _photo_plausible(p: pd.DataFrame, card: pd.DataFrame, n_col: str) -> np.ndarray:
    a = card.reindex(p["id_a"].to_numpy()).reset_index(drop=True)
    b = card.reindex(p["id_b"].to_numpy()).reset_index(drop=True)
    same_city = (a["city"] == b["city"]).to_numpy()
    same_rooms = (a["rooms"] == b["rooms"]).to_numpy()
    area = _rel_diff(a["area"].to_numpy(), b["area"].to_numpy())
    n = p[n_col].to_numpy()
    return same_city & np.where(n >= 2, area <= PHOTO_AREA_TOL_2, same_rooms & (area <= PHOTO_AREA_TOL_1))


def find_duplicates(df: pd.DataFrame, hashes: pd.DataFrame | None = None, emb: pd.DataFrame | None = None,
                    use_emb: bool = True) -> DuplicateReport:
    """Duplicate pairs and connected-component groups among the ads of ``df`` (clean columns needed:
    id, city, rooms, floor, total_floors, year_built, area, price, lat, lon, street, house_num)."""
    ids = df[ID].astype("int64").to_numpy()
    card = df.set_index(df[ID].astype("int64"))[["city", "rooms", "area"]]
    hashes = load_hashes() if hashes is None else hashes
    hashes = hashes[hashes["ad_id"].isin(ids)].reset_index(drop=True)
    ad = hashes["ad_id"].to_numpy()

    hp = hash_image_pairs(hashes)
    img = hp.assign(cos=np.nan)
    ad_pairs = [ad_pairs_from_images(hp, ad, hub_images(hp, ad), "phash")]
    if use_emb:
        emb = load_embeddings() if emb is None else emb
    if use_emb and emb is not None:
        pos = pd.MultiIndex.from_frame(emb[["ad_id", "k"]]).get_indexer(pd.MultiIndex.from_frame(hashes[["ad_id", "k"]]))
        have = pos >= 0
        mat = emb.attrs["emb"]
        e = np.zeros((len(hashes), mat.shape[1]), np.float32)
        e[have] = mat[pos[have]]
        ep = emb_image_pairs(e, np.where(have, ad, -np.arange(len(ad)) - 1))   # images without embedding never match
        ad_pairs.append(ad_pairs_from_images(ep, ad, hub_images(ep, ad), "emb"))
        img = img.merge(ep, on=["i", "j"], how="outer", suffixes=("", "_e"))
        img["cos"] = img["cos_e"].combine_first(img["cos"]) if "cos_e" in img else img["cos"]
        img = img.drop(columns=[c for c in img.columns if c.endswith("_e")])
    hub_all = hub_images(img, ad)
    img_out = pd.DataFrame({"ad_a": ad[img["i"]], "k_a": hashes["k"].to_numpy()[img["i"]],
                            "ad_b": ad[img["j"]], "k_b": hashes["k"].to_numpy()[img["j"]],
                            "phash_d": img["phash_d"], "dhash_d": img["dhash_d"], "cos": img["cos"].round(4),
                            "hub": hub_all[img["i"]] | hub_all[img["j"]]})

    pairs = None
    for p in ad_pairs:
        col = [c for c in p.columns if c.startswith("n_")][0]
        p = p[_photo_plausible(p, card, col)]
        pairs = p if pairs is None else pairs.merge(p, on=["id_a", "id_b"], how="outer")
    pairs = pairs.merge(structured_pairs(df), on=["id_a", "id_b"], how="outer")
    pairs = pairs.merge(text_pairs(df), on=["id_a", "id_b"], how="outer")
    for c in [c for c in pairs.columns if c.startswith("n_")]:
        pairs[c] = pairs[c].fillna(0).astype(int)
    pairs["struct"] = pairs["struct"].astype("boolean").fillna(False).astype(bool)
    sig = pairs[[c for c in ("n_phash", "n_emb") if c in pairs]].gt(0)
    sig["struct"] = pairs["struct"]
    sig["text"] = pairs["text_cos"].notna()
    pairs["signals"] = sig.apply(lambda r: "+".join(k.replace("n_", "") for k, v in r.items() if v), axis=1)
    pairs["tier"] = np.where(_same_flat(pairs, df), "same_flat", "near")
    pairs = pairs.sort_values(["id_a", "id_b"]).reset_index(drop=True)
    return DuplicateReport(pairs=pairs, groups=group_ids(ids, pairs), image_pairs=img_out)


def _same_flat(pairs: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Strict tier: the structured match, or a photo link whose cards agree on rooms, floor (when both
    known) and area within SAME_FLAT_AREA_TOL. The rest ("near") are mostly different flats of one
    building that share developer renders / plans: not the same flat, but near-identical targets."""
    card = df.set_index(df[ID].astype("int64"))[["rooms", "floor", "area"]]
    a = card.reindex(pairs["id_a"].to_numpy()).reset_index(drop=True)
    b = card.reindex(pairs["id_b"].to_numpy()).reset_index(drop=True)
    floor_ok = (a["floor"] == b["floor"]) | a["floor"].isna() | b["floor"].isna()
    photo_same = (a["rooms"] == b["rooms"]) & floor_ok & (_rel_diff(a["area"], b["area"]) <= SAME_FLAT_AREA_TOL)
    return pairs["struct"].to_numpy() | pairs["text_cos"].notna().to_numpy() | photo_same.to_numpy()


def group_ids(ids: np.ndarray, pairs: pd.DataFrame, tier: str | None = None) -> pd.DataFrame:
    """Connected components of the pair graph over ``ids`` (only pairs of ``tier`` if given).
    Group label = smallest ad id in the group; singletons are their own group."""
    ids = np.asarray(ids, dtype="int64")
    pos = pd.Series(np.arange(len(ids)), index=ids)
    p = pairs[pairs["id_a"].isin(ids) & pairs["id_b"].isin(ids)]
    if tier is not None:
        p = p[p["tier"] == tier]
    g = coo_matrix((np.ones(len(p)), (pos[p["id_a"]].to_numpy(), pos[p["id_b"]].to_numpy())), shape=(len(ids),) * 2)
    _, comp = connected_components(g, directed=False)
    label = pd.Series(ids).groupby(comp).transform("min").to_numpy()
    size = pd.Series(comp).map(pd.Series(comp).value_counts()).to_numpy()
    return pd.DataFrame({ID: ids, "group": label, "group_size": size})
