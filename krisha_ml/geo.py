"""Location features from OpenStreetMap and the unused raw ad fields (Final, track "geo").

OpenStreetMap points of interest are downloaded once per city and layer with the Overpass API and
cached as raw JSON in ``cache/osm/<city>_<layer>.json``. Per-ad features are computed with a
haversine BallTree for each city:

* ``geo_dist_centre_km``   distance to the city centre (Astana: Baiterek, Almaty: Republic Square)
* ``geo_dist_<layer>_km``  distance to the nearest park, school, kindergarten, mall, supermarket,
                           hospital/clinic, bus stop, cafe/restaurant, river, metro (Almaty only, NaN in Astana)
* ``geo_n_<layer>_1km``    number of such objects within 1 km
* ``geo_bus_stops_500m``   number of bus stops within 500 m

Parks and rivers are polygons/lines: every vertex of their geometry is a point, so the distance is
to the nearest edge, not to a centroid. Other ways/relations use their Overpass ``center``.

All features are static (no prices are used), so they can go into ``core.cv_evaluate(extra=...)``.
Ads with no coordinates (23 after the Endterm fix) get NaN everywhere.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

from . import raw
from .core import CACHE_DIR, CITY_BOX, ID

OSM_DIR = CACHE_DIR / "osm"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "krisha-ml-course-project/1.0 (non-commercial university ML course, SE-2424)"
EARTH_KM = 6371.0088

CENTRES = {"Astana": (51.128207, 71.430411),   # Baiterek
           "Almaty": (43.238045, 76.945465)}   # Republic Square

# layer -> (Overpass statements with {bbox}, output mode, cities)
# "center": one point per object; "geom": every vertex (nearest-edge distance for parks and rivers)
LAYERS: dict[str, tuple[list[str], str, tuple[str, ...]]] = {
    "park": (['nwr["leisure"~"^(park|garden)$"]({bbox});'], "geom", ("Astana", "Almaty")),
    "school": (['nwr["amenity"="school"]({bbox});'], "center", ("Astana", "Almaty")),
    "kindergarten": (['nwr["amenity"="kindergarten"]({bbox});'], "center", ("Astana", "Almaty")),
    "mall": (['nwr["shop"="mall"]({bbox});'], "center", ("Astana", "Almaty")),
    "supermarket": (['nwr["shop"="supermarket"]({bbox});'], "center", ("Astana", "Almaty")),
    "health": (['nwr["amenity"~"^(hospital|clinic|doctors)$"]({bbox});'], "center", ("Astana", "Almaty")),
    "bus_stop": (['node["highway"="bus_stop"]({bbox});',
                  'node["public_transport"="platform"]["bus"="yes"]({bbox});'], "center", ("Astana", "Almaty")),
    "food": (['nwr["amenity"~"^(cafe|restaurant|fast_food)$"]({bbox});'], "center", ("Astana", "Almaty")),
    "river": (['way["waterway"="river"]({bbox});'], "geom", ("Astana", "Almaty")),
    "metro": (['node["station"="subway"]({bbox});', 'node["railway"="station"]["subway"="yes"]({bbox});'],
              "center", ("Almaty",)),
}
COUNT_LAYERS = ("park", "school", "kindergarten", "supermarket", "health", "food")


# --------------------------------------------------------------------------- download
def _bbox(city: str) -> str:
    la0, la1, lo0, lo1 = CITY_BOX[city]
    return f"{la0},{lo0},{la1},{lo1}"


def _query(layer: str, city: str) -> str:
    statements, mode, _ = LAYERS[layer]
    body = "".join(s.format(bbox=_bbox(city)) for s in statements)
    out = "out geom;" if mode == "geom" else "out center;"
    return f"[out:json][timeout:120];({body});{out}"


def fetch_layer(layer: str, city: str, force: bool = False, retries: int = 4) -> dict:
    """Raw Overpass JSON for one layer and city, cached in cache/osm/ (downloaded at most once)."""
    path = OSM_DIR / f"{city}_{layer}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        resp = requests.post(OVERPASS_URL, data={"data": _query(layer, city)},
                             headers={"User-Agent": USER_AGENT}, timeout=180)
        if resp.status_code == 200:
            data = resp.json()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, path)          # atomic: a half-written cache never exists
            return data
        if resp.status_code not in (429, 502, 503, 504) or attempt == retries:
            resp.raise_for_status()
        time.sleep(10 * (attempt + 1))     # Overpass rate limit: back off
    raise RuntimeError("unreachable")


def fetch_all(force: bool = False, pause_s: float = 2.0) -> None:
    """Download every layer for both cities (polite pause between live requests)."""
    for layer, (_, _, cities) in LAYERS.items():
        for city in cities:
            cached = (OSM_DIR / f"{city}_{layer}.json").exists()
            fetch_layer(layer, city, force=force)
            if force or not cached:
                time.sleep(pause_s)


def layer_points(layer: str, city: str) -> np.ndarray:
    """(n, 2) array of lat/lon points for a layer (vertices for 'geom' layers, centres otherwise)."""
    _, mode, _ = LAYERS[layer]
    pts = []
    for el in fetch_layer(layer, city)["elements"]:
        if mode == "geom":
            if "geometry" in el:
                pts.extend((g["lat"], g["lon"]) for g in el["geometry"] if g)
            elif el["type"] == "relation":
                for m in el.get("members", []):
                    pts.extend((g["lat"], g["lon"]) for g in m.get("geometry", []) if g)
            elif "lat" in el:
                pts.append((el["lat"], el["lon"]))
        elif "lat" in el:
            pts.append((el["lat"], el["lon"]))
        elif "center" in el:
            pts.append((el["center"]["lat"], el["center"]["lon"]))
    return np.unique(np.asarray(pts, dtype="float64").reshape(-1, 2), axis=0)


def poi_summary() -> pd.DataFrame:
    """Number of OSM objects and points per city and layer (for the report)."""
    rows = []
    for layer, (_, _, cities) in LAYERS.items():
        for city in cities:
            rows.append({"city": city, "layer": layer, "n_objects": len(fetch_layer(layer, city)["elements"]),
                         "n_points": len(layer_points(layer, city))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- features
def _rad(latlon: np.ndarray) -> np.ndarray:
    return np.radians(latlon)


def geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ad OSM features for a frame with id, city, lat, lon (coordinate-fixed). Keyed by ``id``."""
    out = pd.DataFrame({ID: df[ID].to_numpy(dtype="int64")}, index=df.index)
    for city, (clat, clon) in CENTRES.items():
        rows = (df["city"] == city) & df["lat"].notna() & df["lon"].notna()
        if not rows.any():
            continue
        q = _rad(df.loc[rows, ["lat", "lon"]].to_numpy(dtype="float64"))
        centre = _rad(np.array([[clat, clon]]))
        d = np.arccos(np.clip(np.sin(q[:, 0]) * np.sin(centre[0, 0])
                              + np.cos(q[:, 0]) * np.cos(centre[0, 0]) * np.cos(q[:, 1] - centre[0, 1]), -1, 1))
        out.loc[rows, "geo_dist_centre_km"] = d * EARTH_KM
        for layer, (_, _, cities) in LAYERS.items():
            if city not in cities:
                continue
            pts = layer_points(layer, city)
            if len(pts) == 0:
                continue
            tree = BallTree(_rad(pts), metric="haversine")
            dist, _ = tree.query(q, k=1)
            out.loc[rows, f"geo_dist_{layer}_km"] = dist[:, 0] * EARTH_KM
            if layer in COUNT_LAYERS:
                out.loc[rows, f"geo_n_{layer}_1km"] = tree.query_radius(q, r=1.0 / EARTH_KM, count_only=True)
            if layer == "bus_stop":
                out.loc[rows, "geo_bus_stops_500m"] = tree.query_radius(q, r=0.5 / EARTH_KM, count_only=True)
    cols = [ID, "geo_dist_centre_km"] + [c for c in out.columns if c.startswith("geo_") and c != "geo_dist_centre_km"]
    return out[cols].reset_index(drop=True).astype({c: "float64" for c in cols if c != ID})


# --------------------------------------------------------------------------- raw fields
CAT_RAW = ["flooring", "door", "furniture", "internet", "phone", "balcony_glazed"]
SECURITY_EN = {"видеодомофон": "video_intercom", "видеонаблюдение": "cctv", "домофон": "intercom",
               "кодовый_замок": "code_lock", "консьерж": "concierge", "охрана": "guard",
               "решетки_на_окнах": "window_bars", "сигнализация": "alarm"}


def rawfield_features(raw_tab: pd.DataFrame | None = None, min_count: int = 30) -> pd.DataFrame:
    """Unused raw ad fields as features, keyed by ``id``.

    Categorical fields keep their raw values (one-hot in the harness, NaN -> 'unknown'); ``exchange``
    becomes 0/1; ``security`` becomes multi-hot items seen in >= ``min_count`` ads plus their count,
    NaN when the ad lists no security field (missing is not the same as 'none').
    Skipped: renovation_raw and mortgaged_text (same information as base ``condition``/``mortgaged``),
    complex_id/name (1,157 levels; a fitted target encoding would belong to a feature_fn),
    estate_verified (constant 0).
    """
    r = raw.load_raw_details() if raw_tab is None else raw_tab
    out = pd.DataFrame({ID: r[ID].astype("int64")})
    for c in CAT_RAW:
        out[f"raw_{c}"] = r[c].astype(object).where(r[c].notna(), None)
    out["raw_exchange"] = r["exchange"].map({"Да": 1.0, "Нет": 0.0})
    sec = raw.multi_hot(r["security"], prefix="raw_sec_", min_count=min_count).astype("float64")
    sec.columns = ["raw_sec_" + SECURITY_EN.get(c.removeprefix("raw_sec_"), c.removeprefix("raw_sec_")) for c in sec.columns]
    missing = r["security"].isna().to_numpy()
    sec.loc[missing, :] = np.nan
    sec["raw_sec_n_items"] = raw.multi_hot(r["security"], prefix="s_").sum(axis=1).where(~missing)
    out = pd.concat([out, sec.reset_index(drop=True).set_index(out.index)], axis=1)
    return out.drop_duplicates(ID).reset_index(drop=True)
