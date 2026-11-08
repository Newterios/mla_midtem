"""Inference API of the demo app (no Gradio import, usable from scripts, tests and the notebook).

    from app.predict import Predictor, validate
    pred = Predictor()                                    # loads models once (~2 s)
    listing = validate({"city": "Astana", "district": "Esilyskiy", "area": 62, "rooms": 2,
                        "floor": 5, "total_floors": 12, "year_built": 2019,
                        "building_type": "monolithic", "condition": "fresh_renovation",
                        "asking_price_m": 45})
    result = pred.predict(listing, photos=["kitchen.jpg"])   # -> Prediction
    result.to_dict()

Every block is optional and degrades on its own, with a note in ``Prediction.notes`` /
``Predictor.status()``:

price model   ``models/final_model.joblib`` when present. A ``krisha_ml.fusion`` model (the Final choice,
              e.g. the network + base blend) gets a one-row frame built from the form: geo block from the
              coordinates, text flags + USER-base sentence embedding from the description, CLIP
              photo embedding from the uploads; blocks a form cannot give (raw ad-page fields, the
              photo track's tabular features) stay missing, exactly like ads without them in training.
              Otherwise a fitted estimator (``predict(X) -> KZT``,
              ``feature_names_in_`` gives its columns) or a dict ``{"model": est, "name": str,
              "intervals": ConformalIntervals, "comparables": Comparables, "photo_model": ...}``.
              Extra columns beyond the 24 Endterm features are filled from what the form can
              provide (``geo_*`` via ``krisha_ml.geo``, ``flag_*`` via ``krisha_ml.text``), the rest
              NaN. Otherwise: the submitted Endterm model ``models/endterm_best_model.joblib``.
intervals     a ``ConformalIntervals`` object from the final bundle or ``outputs/trust/*.joblib``;
              otherwise absolute-residual split conformal on the honest base OOF log residuals
              (``outputs/core/base_oof.csv``), Mondrian per city, centred on the estimate.
SHAP          ``krisha_ml.trust.shap_explain`` (exact TreeSHAP, log scale, grouped features).
comparables   ``krisha_ml.trust.Comparables`` fitted on the 3,852 training ads (never test ads).
text flags    ``krisha_ml.text.extract_flags`` (regex red flags) + checks against the form fields.
photos        a photo-track artifact ``models/photo_condition.joblib`` (dict with ``model`` and
              ``encoder="clip"``) or the fallback: logistic probe on frozen CLIP ViT-B/32 photo
              embeddings (``cache/clip_emb.npz``) of training ads, labelled with their stated
              condition (4 classes), cached in ``cache/app/``.
"""
from __future__ import annotations

import importlib
import io
import math
import os
import threading
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from krisha_ml import core

# --------------------------------------------------------------------------- vocabularies
CITIES = ("Astana", "Almaty")
DISTRICTS: dict[str, tuple[str, ...]] = {
    "Astana": ("Esilyskiy", "Nura", "Almaty", "Saryarka", "Baykonur", "Sarayshyk"),
    "Almaty": ("Bostandykskiy", "Medeuskiy", "Almalinskiy", "Auezovskiy", "Alatauskiy",
               "Nauryzbayskiy", "Turksibskiy", "Zhetysuskiy"),
}
DISTRICT_RU = {"Esilyskiy": "Есильский", "Nura": "Нура", "Almaty": "Алматы", "Saryarka": "Сарыарка",
               "Baykonur": "Байконур", "Sarayshyk": "Сарайшык", "Bostandykskiy": "Бостандыкский",
               "Medeuskiy": "Медеуский", "Almalinskiy": "Алмалинский", "Auezovskiy": "Ауэзовский",
               "Alatauskiy": "Алатауский", "Nauryzbayskiy": "Наурызбайский", "Turksibskiy": "Турксибский",
               "Zhetysuskiy": "Жетысуский"}
# field -> {code: Russian hint}; "unknown" = the ad does not state it (the model learned that level too)
CHOICES: dict[str, dict[str, str]] = {
    "building_type": {"monolithic": "монолитный", "brick": "кирпичный", "panel": "панельный",
                      "other": "иное", "unknown": "не указано"},
    "condition": {"fresh_renovation": "свежий ремонт", "good": "не новый, но аккуратный",
                  "needs_repair": "требует ремонта", "rough": "черновая отделка",
                  "free_layout": "свободная планировка", "unknown": "не указано"},
    "bathroom": {"combined": "совмещённый", "separate": "раздельный", "two_or_more": "2 и более",
                 "unknown": "не указано"},
    "furnished": {"fully": "полностью", "partly": "частично", "no": "без мебели", "unknown": "не указано"},
    "balcony": {"has": "есть", "unknown": "не указано"},
    "parking": {"parking": "паркинг", "street": "рядом / на улице", "garage": "гараж", "unknown": "не указано"},
    "former_dorm": {"no": "нет", "yes": "да", "unknown": "не указано"},
    "seller_type": {"owner": "хозяин", "specialist": "специалист / агент", "company": "застройщик / компания"},
}

# photo condition classes (mirror krisha_ml.photos.CLASSES / CONDITION_TO_CLASS)
PHOTO_CLASSES = ("fresh", "good", "needs_repair", "shell")
PHOTO_CLASS_LABEL = {"fresh": "fresh renovation (свежий ремонт)", "good": "lived-in, tidy (аккуратный)",
                     "needs_repair": "needs repair (требует ремонта)", "shell": "shell / rough (черновая)"}
CONDITION_TO_CLASS = {"fresh_renovation": "fresh", "good": "good", "needs_repair": "needs_repair",
                      "rough": "shell", "free_layout": "shell"}

# text flags of krisha_ml.text.FLAGS -> (severity, English label)
FLAG_INFO = {
    "assignment_of_rights": ("red", "Assignment of rights (переуступка): unfinished building, legal risk"),
    "cash_only": ("red", "Cash only / no mortgage (только наличные): often a document problem"),
    "encumbrance": ("red", "Encumbrance, pledge or arrest mentioned (обременение / залог)"),
    "former_dorm_text": ("red", "Former dormitory (бывшее общежитие)"),
    "needs_repair": ("amber", "Text says it needs repair (требует ремонта)"),
    "rough_finish": ("amber", "Rough / pre-finish, no renovation (черновая / предчистовая)"),
    "urgent": ("amber", "Urgent sale (срочно)"),
    "bargain": ("info", "Price negotiable (торг)"),
    "no_encumbrance": ("good", "States no encumbrance (без обременений)"),
    "designer_renovation": ("good", "Designer renovation (дизайнерский ремонт)"),
    "panoramic_view": ("good", "Panoramic windows / view (панорамные окна)"),
    "two_level": ("good", "Two-level apartment / penthouse (двухуровневая)"),
}

MAX_PHOTOS = 3
MAX_PHOTO_BYTES = 15_000_000
MAX_DESCRIPTION = 10_000
QUERY_ID = -1
LEVELS = (0.8, 0.9)


class ValidationError(ValueError):
    """Input problems, one human-readable message per field."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# --------------------------------------------------------------------------- input
@dataclass
class ListingInput:
    city: str
    district: str
    area: float
    rooms: int
    floor: int
    total_floors: int
    year_built: int
    building_type: str = "unknown"
    condition: str = "unknown"
    kitchen_area: float | None = None
    ceiling_height: float | None = None
    bathroom: str = "unknown"
    furnished: str = "unknown"
    balcony: str = "unknown"
    parking: str = "unknown"
    former_dorm: str = "unknown"
    seller_type: str = "owner"
    n_photos: int | None = None
    has_complex: bool = False
    mortgaged: bool = False
    is_urgent: bool = False
    lat: float | None = None
    lon: float | None = None
    description: str = ""
    asking_price_m: float | None = None

    def to_clean_frame(self, n_photos_default: float = 15.0) -> pd.DataFrame:
        """One row with the clean-CSV columns the models and trust tools read."""
        row = asdict(self)
        row.pop("asking_price_m")
        row.update(id=QUERY_ID, log_area=math.log(self.area),
                   n_photos=float(self.n_photos) if self.n_photos is not None else n_photos_default,
                   price=(self.asking_price_m or np.nan) * 1e6)
        for c in ("kitchen_area", "ceiling_height", "lat", "lon"):
            row[c] = np.nan if row[c] is None else float(row[c])
        return pd.DataFrame([row])


def _num(raw: Mapping[str, Any], key: str, errors: list[str], *, label: str, lo: float, hi: float,
         required: bool = True, integer: bool = False) -> float | None:
    v = raw.get(key)
    if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and math.isnan(v)):
        if required:
            errors.append(f"{label} is required")
        return None
    try:
        x = float(str(v).replace(",", ".").replace(" ", "")) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a number, got {v!r}")
        return None
    if not required and x == 0:
        # an optional number left empty in the UI arrives as 0 once its widget has been rendered
        # (Gradio Number inside an opened accordion); 0 is never a real kitchen, ceiling, price or coordinate
        return None
    if not math.isfinite(x) or not lo <= x <= hi:
        errors.append(f"{label} must be between {lo:g} and {hi:g}, got {x:g}")
        return None
    if integer and x != int(x):
        errors.append(f"{label} must be a whole number, got {x:g}")
        return None
    return int(x) if integer else x


def _choice(raw: Mapping[str, Any], key: str, errors: list[str], default: str) -> str:
    v = raw.get(key)
    v = default if v in (None, "") else str(v)
    if v not in CHOICES[key]:
        errors.append(f"{key.replace('_', ' ')} must be one of {', '.join(CHOICES[key])}, got {v!r}")
    return v


def _flag(raw: Mapping[str, Any], key: str) -> bool:
    v = raw.get(key, False)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "да"}
    return bool(v)


def validate(raw: Mapping[str, Any]) -> ListingInput:
    """Check and coerce form values. Raises ``ValidationError`` listing every problem at once."""
    errors: list[str] = []
    city = str(raw.get("city") or "")
    if city not in CITIES:
        errors.append(f"city must be Astana or Almaty, got {city!r}")
    district = str(raw.get("district") or "unknown")
    if city in CITIES and district != "unknown" and district not in DISTRICTS[city]:
        errors.append(f"district {district!r} is not in {city} (choose one of {', '.join(DISTRICTS[city])})")

    area = _num(raw, "area", errors, label="Total area (m²)", lo=10, hi=1000)
    rooms = _num(raw, "rooms", errors, label="Rooms", lo=1, hi=10, integer=True)
    floor = _num(raw, "floor", errors, label="Floor", lo=1, hi=100, integer=True)
    total = _num(raw, "total_floors", errors, label="Total floors", lo=1, hi=100, integer=True)
    year = _num(raw, "year_built", errors, label="Year built", lo=1900, hi=2030, integer=True)
    kitchen = _num(raw, "kitchen_area", errors, label="Kitchen area (m²)", lo=2, hi=200, required=False)
    ceiling = _num(raw, "ceiling_height", errors, label="Ceiling height (m)", lo=2.0, hi=6.0, required=False)
    n_photos = _num(raw, "n_photos", errors, label="Photos in the ad", lo=0, hi=200, required=False, integer=True)
    asking = _num(raw, "asking_price_m", errors, label="Asking price (M KZT)", lo=1, hi=5000, required=False)
    lat = _num(raw, "lat", errors, label="Latitude", lo=-90, hi=90, required=False)
    lon = _num(raw, "lon", errors, label="Longitude", lo=-180, hi=180, required=False)

    if floor is not None and total is not None and floor > total:
        errors.append(f"Floor ({floor}) cannot be above total floors ({total})")
    if kitchen is not None and area is not None and kitchen >= area:
        errors.append(f"Kitchen area ({kitchen:g}) must be smaller than total area ({area:g})")
    if (lat is None) != (lon is None):
        errors.append("Give both latitude and longitude, or leave both empty")
    elif lat is not None and city in CITIES:
        la0, la1, lo0, lo1 = core.CITY_BOX[city]
        if not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
            errors.append(f"Coordinates ({lat:.4f}, {lon:.4f}) are outside {city} "
                          f"(lat {la0}–{la1}, lon {lo0}–{lo1})")
    description = "" if raw.get("description") is None else str(raw.get("description"))
    if len(description) > MAX_DESCRIPTION:
        errors.append(f"Description is too long ({len(description):,} characters, max {MAX_DESCRIPTION:,})")

    chosen = {k: _choice(raw, k, errors, "owner" if k == "seller_type" else "unknown") for k in CHOICES}
    if errors:
        raise ValidationError(errors)
    return ListingInput(city=city, district=district, area=area, rooms=rooms, floor=floor, total_floors=total,
                        year_built=year, kitchen_area=kitchen, ceiling_height=ceiling, n_photos=n_photos,
                        has_complex=_flag(raw, "has_complex"), mortgaged=_flag(raw, "mortgaged"),
                        is_urgent=_flag(raw, "is_urgent"), lat=lat, lon=lon, description=description.strip(),
                        asking_price_m=asking, **chosen)


# --------------------------------------------------------------------------- photos
@dataclass
class LoadedPhoto:
    name: str
    image: Any          # PIL.Image.Image, RGB


def load_photos(photos: Iterable[Any] | None) -> tuple[list[LoadedPhoto], list[str]]:
    """Open up to ``MAX_PHOTOS`` images (paths, bytes, PIL images or uint8 arrays).

    Unreadable or oversized files are skipped with a warning; nothing here raises."""
    from PIL import Image, UnidentifiedImageError
    out, warns = [], []
    items = [p for p in (photos or []) if p is not None]
    if len(items) > MAX_PHOTOS:
        warns.append(f"{len(items)} photos given, only the first {MAX_PHOTOS} are used")
        items = items[:MAX_PHOTOS]
    for i, item in enumerate(items, 1):
        name = Path(str(getattr(item, "name", item))).name if isinstance(item, (str, os.PathLike)) or hasattr(item, "name") \
            else f"photo {i}"
        try:
            if isinstance(item, Image.Image):
                im = item.copy()
            elif isinstance(item, np.ndarray):
                im = Image.fromarray(np.asarray(item, dtype=np.uint8))
            else:
                if isinstance(item, (bytes, bytearray)):
                    data = bytes(item)
                else:
                    path = Path(item) if isinstance(item, (str, os.PathLike)) else Path(getattr(item, "name", item))
                    if path.stat().st_size > MAX_PHOTO_BYTES:
                        raise ValueError(f"file is larger than {MAX_PHOTO_BYTES // 1_000_000} MB")
                    data = path.read_bytes()
                if len(data) > MAX_PHOTO_BYTES:
                    raise ValueError(f"file is larger than {MAX_PHOTO_BYTES // 1_000_000} MB")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    im = Image.open(io.BytesIO(data))
                    im.load()
            im = im.convert("RGB")
            if min(im.size) < 32:
                raise ValueError(f"image is too small ({im.size[0]}×{im.size[1]} px)")
            out.append(LoadedPhoto(name, im))
        except (UnidentifiedImageError, OSError, ValueError, TypeError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as e:
            why = "unsupported or corrupt image file" if isinstance(e, UnidentifiedImageError) else \
                "file not found" if isinstance(e, FileNotFoundError) else str(e)
            warns.append(f"{name}: not a readable image, skipped ({why})")
    return out, warns


def square_224(im) -> np.ndarray:
    """Shorter side to 224 px (bicubic) and centre crop, as krisha_ml.photos does for the training photos."""
    from PIL import Image
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.uint8)


class ClipConditionModel:
    """Photo condition classifier: frozen CLIP ViT-B/32 (laion2b) image embedding -> sklearn probe.
    Ad-level probabilities = mean over the photos (as in krisha_ml.photos.ad_proba)."""

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, probe, classes: Sequence[str] = PHOTO_CLASSES, source: str = ""):
        self.probe, self.classes, self.source = probe, tuple(classes), source
        self._clip = None
        self._lock = threading.Lock()

    def _encoder(self):
        with self._lock:
            return self._load_encoder()

    def _load_encoder(self):
        if self._clip is None:
            import open_clip
            import torch
            old = os.environ.get("HF_HUB_OFFLINE")
            try:      # weights are in the local HF cache: no network round trip (falls back online if not)
                os.environ["HF_HUB_OFFLINE"] = "1"
                model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            except Exception:
                os.environ.pop("HF_HUB_OFFLINE", None)
                model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            finally:
                if old is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old
            self._clip = (model.eval(), torch)
        return self._clip

    def warmup(self) -> None:
        """Load the CLIP encoder now (the app calls this in a background thread at start-up)."""
        self._encoder()

    def embed(self, images: Sequence[Any]) -> np.ndarray:
        model, torch = self._encoder()
        x = np.stack([square_224(im) for im in images]).astype(np.float32) / 255.0
        x = (x - np.array(self.CLIP_MEAN, np.float32)) / np.array(self.CLIP_STD, np.float32)
        with torch.inference_mode():
            e = model.encode_image(torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()).float()
            e = torch.nn.functional.normalize(e, dim=1).numpy()
        return e

    def predict_proba(self, images: Sequence[Any]) -> np.ndarray:
        """(n_images, n_classes) class probabilities."""
        return np.asarray(self.probe.predict_proba(self.embed(images)), float)


def fit_clip_probe(train_ids: Iterable[int], train_df: pd.DataFrame, root: Path, C: float = 0.5):
    """Logistic probe on per-photo CLIP embeddings of training ads with a stated condition."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    z = np.load(root / "cache" / "clip_emb.npz")
    lab = train_df.set_index("id")["condition"].map(CONDITION_TO_CLASS).dropna()
    lab = lab[lab.index.isin(set(map(int, train_ids)))]
    keep = np.isin(z["ad_id"], lab.index.to_numpy())
    X = z["emb"][keep].astype(np.float32)
    y = lab.reindex(z["ad_id"][keep]).map({c: i for i, c in enumerate(PHOTO_CLASSES)}).to_numpy(int)
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                                              random_state=core.SEED)).fit(X, y)


# --------------------------------------------------------------------------- result
@dataclass
class Prediction:
    price_kzt: float
    model_name: str
    intervals: dict[str, tuple[float, float]] | None = None      # {"80": (lo, hi), "90": (lo, hi)} KZT
    interval_source: str | None = None
    verdict: dict[str, Any] | None = None
    shap: pd.DataFrame | None = None            # group, effect_pct, shap_log, value (sorted by |effect|)
    comparables: pd.DataFrame | None = None
    few_comparables: bool | None = None
    comparables_estimate_kzt: float | None = None
    base_price_kzt: float | None = None          # the Endterm base estimate, when the final model is a fusion model
    text_flags: list[dict[str, str]] = field(default_factory=list)
    photo: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def price_m(self) -> float:
        return self.price_kzt / 1e6

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("shap", "comparables"):
            v = getattr(self, k)
            d[k] = None if v is None else v.to_dict(orient="records")
        return d


def verdict_for(asking_kzt: float, estimate_kzt: float,
                intervals: Mapping[str, tuple[float, float]] | None) -> dict[str, Any]:
    """Over/under-pricing verdict of an asking price.

    With intervals: above the 90 % upper bound -> overpriced, above the 80 % bound -> slightly high,
    mirrored below, else fair. Without: the Endterm rule (more than 20 % from the estimate)."""
    gap = asking_kzt / estimate_kzt - 1
    if intervals:
        lo80, hi80 = intervals["80"]
        lo90, hi90 = intervals["90"]
        if asking_kzt > hi90:
            code, text = "over", "Overpriced: above the 90 % range"
        elif asking_kzt > hi80:
            code, text = "high", "Slightly high: above the 80 % range, inside the 90 % range"
        elif asking_kzt < lo90:
            code, text = "under", "Underpriced: below the 90 % range (check the ad carefully)"
        elif asking_kzt < lo80:
            code, text = "low", "Slightly low: below the 80 % range, inside the 90 % range"
        else:
            code, text = "fair", "Fair: inside the 80 % range of similar ads"
        rule = "conformal interval"
    else:
        code = "over" if gap > 0.20 else "under" if gap < -0.20 else "fair"
        text = {"over": "Overpriced: more than 20 % above the estimate",
                "under": "Underpriced: more than 20 % below the estimate",
                "fair": "Fair: within ±20 % of the estimate"}[code]
        rule = "Endterm ±20 % rule"
    return {"code": code, "text": text, "gap_pct": 100 * gap, "asking_kzt": asking_kzt, "rule": rule}


# --------------------------------------------------------------------------- predictor
def _import_optional(name: str):
    """Import a sibling track module; None (with the reason) when it is missing or broken."""
    try:
        return importlib.import_module(name), None
    except Exception as e:  # a track module being written right now may not import yet
        return None, f"{type(e).__name__}: {e}"


class _NeedsBlocks(Exception):
    """The final model exists but needs inputs the form cannot provide."""


class Predictor:
    """Loads the price model and every optional block once; ``predict`` is cheap (SHAP ~50 ms,
    CLIP ~0.3 s per photo after a one-time ~3 s model load)."""

    def __init__(self, root: Path | str | None = None, final_model_path: Path | str | None = None,
                 photo_model: Any = "auto", fit_photo_fallback: bool = True, text_encoder: bool = True):
        self.root = Path(root or core.ROOT)
        self.final_model_path = Path(final_model_path) if final_model_path else self.root / "models" / "final_model.joblib"
        self._status: dict[str, str] = {}
        self.use_text_encoder = text_encoder
        self.fusion_blocks: dict[str, list[str]] = {}       # block name -> model columns (fusion models only)
        self.train = core.load_train()                     # 3,852 training ads only; test ads never used here
        self.n_photos_default = float(np.nanmedian(self.train.df["n_photos"]))
        self.trust, err = _import_optional("krisha_ml.trust")
        if self.trust is None:
            self._status["trust"] = f"module not available yet ({err}): no SHAP, no comparables"
        self.text, err = _import_optional("krisha_ml.text")
        self._status["text"] = ("regex red flags (krisha_ml.text.extract_flags)" if self.text is not None and
                                hasattr(self.text, "extract_flags") else f"module not available yet ({err or 'no extract_flags'})")
        self._load_price_model()
        self._load_intervals()
        self._load_comparables()
        self._load_photo_model(photo_model, fit_photo_fallback)

    # ---------------------------------------------------------------- loading
    def _load_price_model(self) -> None:
        self.bundle: dict[str, Any] = {}
        self.model, self.model_name, self.is_final = None, "", False
        if self.final_model_path.exists():
            try:
                obj = joblib.load(self.final_model_path)
                self.bundle = obj if isinstance(obj, dict) else {"model": obj}
                model = self.bundle["model"]
                if not hasattr(model, "predict"):
                    raise TypeError("bundle['model'] has no predict()")
                blocks = list(getattr(model, "blocks", None) or [])
                if blocks:
                    cols = {getattr(b, "name", str(b)): list(getattr(b, "columns", None) or []) for b in blocks}
                    if not all(cols.values()) and any(n != "text" for n in cols):
                        raise _NeedsBlocks(f"{getattr(model, 'strategy', 'fusion')} model has blocks without column "
                                           f"names ({', '.join(cols)})")
                    self.fusion_blocks = cols
                    self.model_name = str(self.bundle.get("name") or _fusion_name(model, cols))
                    self._status["model"] = (f"{self.model_name} ({self.final_model_path.name}); form-built inputs: "
                                             f"{', '.join(n for n in cols if n in FORM_BLOCKS)}; always missing in "
                                             f"the app: {', '.join(n for n in cols if n not in FORM_BLOCKS) or 'none'}")
                else:
                    self.model_name = str(self.bundle.get("name", "Final model"))
                    self._status["model"] = f"{self.model_name} ({self.final_model_path.name})"
                self.model, self.is_final = model, True
            except _NeedsBlocks as e:
                self._status["model"] = f"final model found, not used in the app: {e}; using Endterm base"
            except Exception as e:
                self._status["model"] = f"final model could not be loaded ({type(e).__name__}: {e}); using Endterm base"
        else:
            self._status["model"] = "final model not available yet (models/final_model.joblib); using Endterm base"
        base_path = self.root / "models" / "endterm_best_model.joblib"
        self.base_model = joblib.load(base_path) if base_path.exists() else core.make_base_model().fit(self.train.X, self.train.y)
        if self.model is None:
            self.model, self.model_name = self.base_model, "Endterm base (tuned HGB, 24 features)"
            self._status["model"] = self._status["model"].replace("; using Endterm base", "") + \
                "; using the Endterm base model (test MAE 6.29 M KZT, MdAPE 9.1 %)"
        self._text_enc_lock = threading.Lock()
        if "text_emb" in self.fusion_blocks:
            self._status["text"] += ("; description embedding: USER-base (sentence-transformers, loaded on first use)"
                                     if self.use_text_encoder else "; description embedding disabled (left missing)")
        cols = getattr(self.model, "feature_names_in_", None)
        self.model_columns = list(cols) if cols is not None else list(self.bundle.get("features", core.FEATURES))

    def _find_trust_object(self, cls_name: str):
        cands = [self.bundle.get(k) for k in ("intervals", "comparables", "photo_model")]
        for p in sorted((self.root / "outputs" / "trust").glob("*.joblib")) + sorted((self.root / "models").glob("trust*.joblib")):
            try:
                obj = joblib.load(p)
            except Exception:
                continue
            cands.extend(obj.values() if isinstance(obj, dict) else [obj])
        return next((c for c in cands if type(c).__name__ == cls_name), None)

    def _load_intervals(self) -> None:
        self.ci = self._find_trust_object("ConformalIntervals") if self.trust is not None else None
        self.qhat, self.qhat_band = None, None
        chosen = self.root / "outputs" / "trust" / "chosen_settings.json"
        if self.ci is None and chosen.exists():
            # the trust track's chosen intervals: absolute log-residual conformal, Mondrian by the price
            # band of the estimate, cross-fitted on honest OOF residuals of the 3,852 training ads
            try:
                import json
                iv = json.loads(chosen.read_text())["intervals"]
                if iv.get("method") == "resid" and iv.get("mondrian") == "estimate_band":
                    self.qhat_band = {str(int(round(float(lv) * 100))): {k: float(v) for k, v in q.items()}
                                      for lv, q in iv["qhat_log"].items()}
                    self.interval_source = (f"trust track choice: {iv.get('chosen_variant', 'abs-residual conformal')} "
                                            "(80 % / 90 %, calibrated on base OOF residuals)")
            except Exception:
                self.qhat_band = None
        if self.qhat_band is not None:
            pass
        elif self.ci is not None:
            self.interval_source = f"trust ConformalIntervals ({self.ci.method}, Mondrian {self.ci.mondrian})"
        else:
            # split-conformal on honest OOF |log residuals| of the base model, per city (+ global)
            oof_path = self.root / "outputs" / "core" / "base_oof.csv"
            if oof_path.exists():
                oof = pd.read_csv(oof_path)
                score = np.abs(oof["log_resid"].to_numpy(float))
                city = core.join_features(oof["id"], self.train.df[["id", "city"]])["city"].to_numpy()
                self.qhat = {}
                for lv in LEVELS:
                    q = {"__all__": _conformal_q(score, lv)}
                    for c in CITIES:
                        q[c] = _conformal_q(score[city == c], lv)
                    self.qhat[str(int(lv * 100))] = q
                self.interval_source = "split conformal on base OOF log residuals, per city"
            else:
                self.qhat, self.interval_source = None, None
        self._status["intervals"] = self.interval_source or "not available (no outputs/core/base_oof.csv)"

    def _load_comparables(self) -> None:
        self.comps = None
        if self.trust is None:
            return
        try:
            self.comps = self._find_trust_object("Comparables") or \
                self.trust.Comparables().fit(self.trust.comparables_frame(self.train.df))
            self._status["comparables"] = f"{self.comps.k} nearest training ads, same city"
            self._status["shap"] = "exact TreeSHAP (krisha_ml.trust.shap_explain)"
        except Exception as e:
            self._status["comparables"] = f"not available ({type(e).__name__}: {e})"

    def _load_photo_model(self, photo_model, fit_fallback: bool) -> None:
        self.photo_model = None
        if photo_model not in ("auto", None):
            self.photo_model = photo_model
            self._status["photos"] = getattr(photo_model, "source", "") or "injected photo model"
            return
        if photo_model is None:
            self._status["photos"] = "photo model disabled"
            return
        art = self.root / "models" / "photo_condition.joblib"
        try:
            if art.exists():
                obj = joblib.load(art)
                if isinstance(obj, dict) and obj.get("encoder") == "clip":
                    self.photo_model = ClipConditionModel(obj["model"], obj.get("classes", PHOTO_CLASSES),
                                                          source=f"photo track artifact ({art.name})")
            if self.photo_model is None:
                self.photo_model = self._fallback_probe(fit_fallback)
        except Exception as e:
            self.photo_model = None
            self._status["photos"] = f"not available ({type(e).__name__}: {e})"
            return
        if self.photo_model is None:
            self._status["photos"] = "photo model not available yet (no models/photo_condition.joblib, no cache/clip_emb.npz)"
        else:
            self._status["photos"] = self.photo_model.source

    def _fallback_probe(self, fit: bool):
        cache = self.root / "cache" / "app" / "photo_probe_clip.joblib"
        src = "fallback: CLIP ViT-B/32 + logistic probe on training-ad photos (cache/app)"
        if cache.exists():
            return ClipConditionModel(joblib.load(cache), source=src)
        if not fit or not (self.root / "cache" / "clip_emb.npz").exists():
            return None
        probe = fit_clip_probe(self.train.ids, self.train.df, self.root)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp")
        joblib.dump(probe, tmp)
        os.replace(tmp, cache)
        return ClipConditionModel(probe, source=src)

    def status(self) -> dict[str, str]:
        """Which block runs on what, for the app's 'Model status' panel."""
        s = dict(self._status)
        s.setdefault("shap", "not available (krisha_ml.trust missing)")
        s.setdefault("comparables", "not available (krisha_ml.trust missing)")
        return s

    # ---------------------------------------------------------------- features
    def _feature_row(self, clean: pd.DataFrame, notes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(X for the price model, X with the 24 base features)."""
        X_base = core.encode_features(clean)
        extra_cols = [c for c in self.model_columns if c not in core.FEATURES]
        if not extra_cols:
            return X_base[self.model_columns], X_base
        extra = pd.DataFrame(index=[0], columns=extra_cols, dtype="float64")
        if any(c.startswith("geo_") for c in extra_cols) and clean["lat"].notna().all():
            geo, err = _import_optional("krisha_ml.geo")
            try:
                g = geo.geo_features(clean)
                for c in extra.columns.intersection(g.columns):
                    extra[c] = g[c].to_numpy()
            except Exception as e:
                notes.append(f"Geo features could not be computed ({err or type(e).__name__}); left empty")
        if self.text is not None and any(c.startswith("flag_") for c in extra_cols):
            f = self.text.extract_flags(clean)
            for c in extra.columns.intersection(f.columns):
                extra[c] = f[c].to_numpy(float)
        missing = [c for c in extra_cols if extra[c].isna().all()]
        if missing:
            notes.append(f"{len(missing)} of {len(extra_cols)} extra model inputs are not in the form and were left "
                         f"empty (imputed): {', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}")
        X = pd.concat([X_base, extra.reset_index(drop=True)], axis=1)
        return X[self.model_columns], X_base

    def _text_embedding(self, text: str, notes: list[str]) -> np.ndarray | None:
        if not self.use_text_encoder or self.text is None:
            return None
        try:
            with self._text_enc_lock:
                return self.text.embed_texts("user-base", [self.text.clean_for_embedding(text)])[0]
        except Exception as e:
            notes.append(f"Description embedding not available ({type(e).__name__}); the text branch sees 'no text'")
            return None

    def _fusion_frame(self, clean: pd.DataFrame, listing: ListingInput, loaded: list[LoadedPhoto],
                      notes: list[str]) -> pd.DataFrame:
        """One-row frame for a krisha_ml.fusion model: clean columns + every block column the form allows."""
        from krisha_ml import raw
        frame = clean.copy()
        frame["text_clean"] = raw.sanitize_text(listing.description)
        cols = {c: np.nan for b in self.fusion_blocks.values() for c in b}
        filled: list[str] = []
        if "geo" in self.fusion_blocks and listing.lat is not None:
            try:
                from krisha_ml import geo
                g = geo.geo_features(clean)
                for c in self.fusion_blocks["geo"]:
                    src = c.split("__", 1)[1]
                    if src in g.columns:
                        cols[c] = float(g[src].iloc[0])
                filled.append("geo")
            except Exception as e:
                notes.append(f"Geo features could not be computed ({type(e).__name__}); left missing")
        if "text_flags" in self.fusion_blocks and self.text is not None and listing.description:
            f = self.text.extract_flags(clean)
            for c in self.fusion_blocks["text_flags"]:
                src = c.split("__", 1)[1]
                if src in f.columns:
                    cols[c] = float(f[src].iloc[0])
            filled.append("text flags")
        if "text_emb" in self.fusion_blocks and listing.description:
            e = self._text_embedding(listing.description, notes)
            if e is not None and len(e) == len(self.fusion_blocks["text_emb"]):
                cols.update(zip(self.fusion_blocks["text_emb"], map(float, e)))
                filled.append("description embedding")
        self._photo_emb_cache = {}
        if "photo_emb" in self.fusion_blocks and loaded and self.photo_model is not None and hasattr(self.photo_model, "embed"):
            try:
                per = np.asarray(self.photo_model.embed([p.image for p in loaded]), float)
                self._photo_emb_cache[id(loaded)] = per
                m = per.mean(0)
                m /= max(np.linalg.norm(m), 1e-12)
                if len(m) == len(self.fusion_blocks["photo_emb"]):
                    cols.update(zip(self.fusion_blocks["photo_emb"], map(float, m)))
                    filled.append("photo embedding")
            except Exception as e:
                notes.append(f"Photo embedding failed ({type(e).__name__}); the photo branch sees 'no photos'")
        missing = [n for n in self.fusion_blocks if n != "text" and not any(
            isinstance(cols.get(c), float) and math.isfinite(cols[c]) for c in self.fusion_blocks[n])]
        if missing:
            notes.append("Final model inputs left missing (as for training ads without them): " + ", ".join(missing))
        extra = pd.DataFrame([cols])
        for c in extra.columns:   # categorical raw fields stay object (NaN -> "unknown" in the model)
            if c.startswith("rawfields__") and extra[c].isna().all():
                extra[c] = extra[c].astype(object)
        return pd.concat([frame.reset_index(drop=True), extra], axis=1)

    # ---------------------------------------------------------------- predict
    def predict(self, listing: ListingInput | Mapping[str, Any], photos: Iterable[Any] | None = None) -> Prediction:
        """Full analysis of one listing. Raises ``ValidationError`` for bad form values only;
        every optional block failing is reported in ``notes`` instead."""
        if not isinstance(listing, ListingInput):
            listing = validate(listing)
        notes: list[str] = []
        warns: list[str] = []
        clean = listing.to_clean_frame(self.n_photos_default)
        loaded, photo_warns = load_photos(photos)
        warns.extend(photo_warns)
        if self.fusion_blocks:
            X, X_base = self._fusion_frame(clean, listing, loaded, notes), core.encode_features(clean)
        else:
            X, X_base = self._feature_row(clean, notes)

        model, name = self.model, self.model_name
        try:
            price = float(np.asarray(model.predict(X), float)[0])
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"non-positive estimate {price}")
        except Exception as e:
            if model is self.base_model:
                raise
            notes.append(f"Final model failed on this input ({type(e).__name__}: {e}); used the Endterm base")
            model, name, X = self.base_model, "Endterm base (tuned HGB, 24 features)", X_base
            price = float(self.base_model.predict(X_base)[0])
        res = Prediction(price_kzt=price, model_name=name, warnings=warns, notes=notes)

        warns.extend(_range_warnings(listing))
        if listing.lat is None:
            notes.append("No coordinates given: location uses the district only (the model imputes lat/lon)")

        self._add_intervals(res, X_base, listing, used_final=model is not self.base_model)
        if listing.asking_price_m:
            res.verdict = verdict_for(listing.asking_price_m * 1e6, price, res.intervals)
        self._add_shap(res, model, X, X_base)
        self._add_comparables(res, clean, listing)
        res.text_flags = self._text_flags(clean, listing, notes)
        res.photo = self._photo_block(loaded, photo_warns, listing, warns, notes)
        if self.fusion_blocks and model is not self.base_model:
            res.base_price_kzt = float(self.base_model.predict(X_base)[0])
        return res

    def _add_intervals(self, res: Prediction, X_base: pd.DataFrame, listing: ListingInput, used_final: bool = True) -> None:
        try:
            if self.qhat_band is not None:
                lp = math.log(res.price_kzt)
                band = _price_band_label(res.price_kzt)
                res.intervals = {k: (math.exp(lp - q.get(band, q["__all__"])), math.exp(lp + q.get(band, q["__all__"])))
                                 for k, q in self.qhat_band.items() if k in ("80", "90")}
            elif self.ci is not None:
                res.intervals = {str(int(lv * 100)): tuple(float(v[0]) for v in self.ci.predict(X_base, lv)) for lv in self.ci.levels
                                 if int(lv * 100) in (80, 90)}
            elif self.qhat is not None:
                lp = math.log(res.price_kzt)
                res.intervals = {k: (math.exp(lp - q.get(listing.city, q["__all__"])),
                                     math.exp(lp + q.get(listing.city, q["__all__"]))) for k, q in self.qhat.items()}
            res.interval_source = self.interval_source
            if res.intervals and self.is_final and used_final and self.ci is None:
                res.notes.append("Intervals use the base model's OOF calibration, centred on the final estimate "
                                 "(conservative: the final model's CV error is lower)")
        except Exception as e:
            res.intervals = None
            res.notes.append(f"Intervals not available ({type(e).__name__}: {e})")

    def _add_shap(self, res: Prediction, model, X: pd.DataFrame, X_base: pd.DataFrame) -> None:
        if self.trust is None:
            res.notes.append("SHAP reasons: module not available yet")
            return
        try:
            s = self.trust.shap_explain(model, X)
        except Exception:
            try:      # e.g. a final model that is not Pipeline(pre, tree): explain the base model instead
                s = self.trust.shap_explain(self.base_model, X_base)
                res.notes.append("SHAP reasons are for the Endterm base model (the final model is not a tree pipeline)")
            except Exception as e:
                res.notes.append(f"SHAP reasons not available ({type(e).__name__}: {e})")
                return
        tab = s.explain_one(0)
        tab["label"] = tab["group"].map(_pretty_group)
        res.shap = tab[["label", "group", "value", "effect_pct", "shap_log"]]

    def _add_comparables(self, res: Prediction, clean: pd.DataFrame, listing: ListingInput) -> None:
        if self.comps is None:
            res.notes.append("Comparable listings: module not available yet")
            return
        try:
            q = self.comps.query(clean, exclude_self=False)
        except Exception as e:
            res.notes.append(f"Comparable listings not available ({type(e).__name__}: {e})")
            return
        res.few_comparables = bool(q["few_comparables"].iloc[0])
        res.comparables_estimate_kzt = float(q["comp_price_m2"].median() * listing.area)
        if res.few_comparables:
            res.warnings.append("Few comparables: this flat is unusual for the training data "
                                "(5th nearest ad is farther than for 95 % of ads), so the estimate is less reliable")
        res.comparables = pd.DataFrame({
            "rank": q["rank"], "ad id": q["comp_id"], "price, M": (q["comp_price"] / 1e6).round(1),
            "M per m²": (q["comp_price_m2"] / 1e6).round(3), "area": q["area"], "rooms": q["rooms"].astype(int),
            "year": q["year_built"].astype("Int64"), "floor": q["floor"].astype("Int64"),
            "district": q["district"].fillna("unknown"),
            "km away": _km(listing, q) if listing.lat is not None else np.nan,
            "link": "https://krisha.kz/a/show/" + q["comp_id"].astype(str)})

    def _text_flags(self, clean: pd.DataFrame, listing: ListingInput, notes: list[str]) -> list[dict[str, str]]:
        if not listing.description:
            return []
        out = []
        from krisha_ml import raw
        if raw.mask_contacts(listing.description) != listing.description:
            out.append({"key": "contacts", "severity": "info", "label": "Contains phone / e-mail / link (masked on krisha)"})
        if self.text is None or not hasattr(self.text, "extract_flags"):
            notes.append("Text red flags: module not available yet")
            return out
        try:
            f = self.text.extract_flags(clean).iloc[0]
        except Exception as e:
            notes.append(f"Text red flags not available ({type(e).__name__}: {e})")
            return out
        hits = {c.removeprefix("flag_") for c, v in f.items() if c.startswith("flag_") and v}
        for k in hits:
            sev, label = FLAG_INFO.get(k, ("info", k.replace("_", " ")))
            out.append({"key": k, "severity": sev, "label": label})
        cond = listing.condition
        checks = [
            ("rough_finish" in hits and cond in ("fresh_renovation", "good"),
             f"Text says rough / no finishing, but the form says '{cond}'"),
            ("needs_repair" in hits and cond == "fresh_renovation", "Text says it needs repair, but the form says fresh renovation"),
            ("encumbrance" in hits and not listing.mortgaged, "Text mentions a pledge / encumbrance, but 'mortgaged' is not ticked"),
            ("former_dorm_text" in hits and listing.former_dorm == "no", "Text mentions a dormitory, but the form says not a former dorm"),
        ]
        out += [{"key": "mismatch", "severity": "red", "label": msg} for bad, msg in checks if bad]
        order = {"red": 0, "amber": 1, "info": 2, "good": 3}
        return sorted(out, key=lambda d: order.get(d["severity"], 9))

    def _photo_block(self, loaded: list[LoadedPhoto], photo_warns: list[str], listing: ListingInput, warns: list[str],
                     notes: list[str]) -> dict[str, Any] | None:
        if not loaded:
            return {"status": "no photos" if not photo_warns else "no readable photos", "n_photos": 0}
        if self.photo_model is None:
            notes.append("Photo condition: module not available yet")
            return {"status": "photo model not available", "n_photos": len(loaded)}
        try:
            emb = self._photo_emb_cache.get(id(loaded)) if hasattr(self, "_photo_emb_cache") else None
            per = np.asarray(self.photo_model.probe.predict_proba(emb) if emb is not None and hasattr(self.photo_model, "probe")
                             else self.photo_model.predict_proba([p.image for p in loaded]), float)
        except Exception as e:
            notes.append(f"Photo condition failed ({type(e).__name__}: {e})")
            return {"status": "photo model failed", "n_photos": len(loaded)}
        classes = tuple(getattr(self.photo_model, "classes", PHOTO_CLASSES))
        proba = per.mean(0)
        top = classes[int(proba.argmax())]
        out = {"status": "ok", "n_photos": len(loaded), "names": [p.name for p in loaded],
               "proba": {c: float(p) for c, p in zip(classes, proba)}, "predicted": top,
               "predicted_label": PHOTO_CLASS_LABEL.get(top, top), "confidence": float(proba.max()),
               "stated": listing.condition, "mismatch": None}
        stated = CONDITION_TO_CLASS.get(listing.condition)
        if stated is None:
            out["mismatch"] = None
            out["comment"] = "Condition not stated in the form; photos suggest " + out["predicted_label"]
        elif stated != top and out["proba"].get(stated, 0) < 0.20 and proba.max() >= 0.45:
            out["mismatch"] = (f"Photos look like {PHOTO_CLASS_LABEL.get(top, top)} ({proba.max():.0%}), "
                               f"but the form says '{listing.condition}' ({out['proba'].get(stated, 0):.0%} on photos)")
            warns.append("Photo vs stated condition: " + out["mismatch"])
        else:
            out["comment"] = "Photos agree with the stated condition" if stated == top else \
                "Photos are not conclusive against the stated condition"
        return out


# --------------------------------------------------------------------------- helpers
FORM_BLOCKS = ("geo", "text_flags", "text_emb", "photo_emb")   # blocks the form (+ uploads) can supply
_BLOCK_LABEL = {"geo": "OSM geo", "rawfields": "raw ad fields", "photo": "photo features", "text_flags": "text flags",
                "text": "TF-IDF text", "text_emb": "text embedding", "photo_emb": "photo embedding"}


def _fusion_name(model, cols: Mapping[str, Any]) -> str:
    kind = {"Cb": "Network + Endterm base, 50/50 log-blend", "C": "Multimodal network", "A": "Residual stacking",
            "B": "HGB with block features"}.get(getattr(model, "strategy", ""), "Fusion model")
    return f"Final: {kind} ({', '.join(_BLOCK_LABEL.get(n, n) for n in cols)})"


def _price_band_label(price_kzt: float) -> str:
    m = price_kzt / 1e6
    return "<25 M" if m < 25 else "25-40 M" if m < 40 else "40-60 M" if m < 60 else "60-100 M" if m < 100 else ">=100 M"

def _conformal_q(scores: np.ndarray, level: float) -> float:
    """ceil((n+1)*level)-th smallest score (finite-sample split-conformal quantile)."""
    scores = np.sort(np.asarray(scores, float))
    k = math.ceil((len(scores) + 1) * level)
    return float(scores[min(k, len(scores)) - 1])


_GROUP_LABEL = {"location": "Location (district + coordinates)", "area": "Total area", "rooms": "Rooms",
                "kitchen_area": "Kitchen area", "floor": "Floor", "total_floors": "Floors in building",
                "year_built": "Year built", "ceiling_height": "Ceiling height", "n_photos": "Photos in ad",
                "city": "City", "building_type": "Building type", "condition": "Condition", "bathroom": "Bathroom",
                "furnished": "Furniture", "balcony": "Balcony", "parking": "Parking", "former_dorm": "Former dorm",
                "seller_type": "Seller type", "has_complex": "Residential complex", "mortgaged": "Mortgaged",
                "is_urgent": "Urgent sale"}


def _pretty_group(g: str) -> str:
    return _GROUP_LABEL.get(g, g.replace("_", " ").capitalize())


def _km(listing: ListingInput, q: pd.DataFrame) -> pd.Series:
    lat0, lon0 = math.radians(listing.lat), math.radians(listing.lon)
    lat, lon = np.radians(q["lat"].to_numpy(float)), np.radians(q["lon"].to_numpy(float))
    a = np.sin((lat - lat0) / 2) ** 2 + math.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    return pd.Series(np.round(2 * 6371.0 * np.arcsin(np.sqrt(a)), 1), index=q.index)


def _range_warnings(x: ListingInput) -> list[str]:
    """Inputs outside the bulk of the training data (1st–99th percentile, Oct 2026 ads)."""
    w = []
    if not 25 <= x.area <= 185:
        w.append(f"Area {x.area:g} m² is outside the usual training range (25–185 m²): extrapolation")
    if x.rooms > 5:
        w.append(f"{x.rooms} rooms is rare in the training data")
    if x.year_built < 1960:
        w.append(f"Year built {x.year_built} is older than almost all training ads")
    if x.total_floors > 30:
        w.append(f"{x.total_floors}-storey buildings are rare in the training data")
    if x.kitchen_area and x.kitchen_area > 0.6 * x.area:
        w.append("Kitchen area is more than 60 % of total area: check the numbers")
    return w
