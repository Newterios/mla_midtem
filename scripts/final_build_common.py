"""Shared helpers for the Final deliverable builders (notebook, report, slides).

    from final_build_common import Values, fill, OUT, ...

The three builders (`build_final_notebook.py`, `build_final_report.py`, `build_final_slides.py`)
never compute a model result themselves: they read what the track scripts wrote to `outputs/`
and degrade gracefully when a file is not there yet.

Placeholders
------------
Text may contain `{{key}}` or `{{key|fmt}}`; `fill(text, values)` replaces every placeholder whose
key resolves and leaves the others as they are (or writes "pending", see `missing=`).
Keys:

* `<dir>/<stem>:<dotted.path>`      a value inside `outputs/<dir>/<stem>.json`
                                    (`leakage/leakage_summary:all_links.groups`,
                                    `trust/chosen_settings:intervals.oof.0.9.coverage`; list items by index)
* `<dir>/<stem>@best:<col>`         the row of `outputs/<dir>/<stem>.csv` with the lowest `cv_mae_M`
                                    among the non-base rows (CV tables of every track)
* `<dir>/<stem>@n` / `@n_pass`      number of rows / rows with `passes_1sd_rule` True (non-base rows)
* `<dir>/<stem>[text]:<col>`        the first row whose first column contains `text` (case-insensitive)
* aliases in `ALIASES` (`base.cv_mae_M`, `base.threshold_M`, `endterm.test_mae_M`, ...)

`fmt` is a Python format spec (`.2f`, `.1%`, `,`) or `pct` (x100, one decimal). Default: floats
with 2 decimals (3 below 1), ints with a thousands separator, booleans as yes/no.

Integration-stage outputs (not written by any track; the builders read them when they exist)
--------------------------------------------------------------------------------------------
* `outputs/fusion/selection.csv`    one row per fused variant, the columns of every track `cv.csv`
                                    (`variant, cv_mae_M, cv_mae_sd_M, cv_mdape, cv_r2, folds_mae_M,
                                    delta_vs_base_M, folds_better, passes_1sd_rule`) + `blocks`
* `outputs/fusion/selection.json`   `{"chosen": str, "blocks": [...], "cv_mae_M", "cv_mae_sd_M",
                                    "group_cv_mae_M", "group_cv_sd_M", "reason": str}`
* `outputs/fusion/cv_blocks.png`    CV MAE of every variant with the 1-SD threshold
* `outputs/final_eval/test_metrics.json`  `{"chosen": {mae_M, rmse_M, mdape, r2, within10, within20,
                                    bootstrap95_mae: [lo, hi], bootstrap95_mdape: [lo, hi]},
                                    "base": {...same...}, "n_test": 963}`
* `outputs/final_eval/results_table.csv`  model, CV MAE, CV MdAPE, test MAE/RMSE/MdAPE/R2/within
* `outputs/final_eval/nov_holdout.json`   `{"n": int, "chosen": {...metrics}, "base": {...},
                                    "by_city": {...}, "features_missing": [...]}`
* `outputs/final_eval/error_by_segment.csv`, `worst_cases.csv`, `residuals.png`, `nov_holdout.png`
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
NOTEBOOK = ROOT / "Final_SE-2424_Balgabekov_Kaiyrbekuly.ipynb"
REPORT = ROOT / "FINAL_REPORT.md"
SLIDES = ROOT / "slides" / "Final_SE-2424_slides.pptx"
TEAM = "Rakhat Balgabekov, Aitbek Kaiyrbekuly"
GROUP = "SE-2424"

# who writes what (used in the "pending" notes)
PRODUCERS = {
    "core": "scripts/verify_endterm.py",
    "leakage": "scripts/leakage_audit.py",
    "geo": "scripts/run_geo.py",
    "photos": "scripts/photo_01_embed.py, photo_02_classify.py, photo_03_price.py",
    "text": "scripts/run_text.py",
    "trust": "scripts/run_trust.py",
    "nov_holdout": "scripts/build_nov_holdout.py",
    "fusion": "the integration stage (fusion of the kept blocks)",
    "final_eval": "the integration stage (one test evaluation + November hold-out)",
}

ALIASES = {
    "base.cv_mae_M": "core/base_cv:cv_mae_M",
    "base.cv_mae_sd_M": "core/base_cv:cv_mae_sd_M",
    "base.cv_mdape": "core/base_cv:cv_mdape",
    "base.cv_r2": "core/base_cv:cv_r2",
    "endterm.test_mae_M": "core/endterm_repro_test:test.mae_M",
    "endterm.test_rmse_M": "core/endterm_repro_test:test.rmse_M",
    "endterm.test_mdape": "core/endterm_repro_test:test.mdape",
    "endterm.test_r2": "core/endterm_repro_test:test.r2",
    "endterm.within10": "core/endterm_repro_test:test.within10",
    "endterm.within20": "core/endterm_repro_test:test.within20",
    "final.chosen": "fusion/selection:chosen",
    "final.cv_mae_M": "fusion/selection:cv_mae_M",
    "final.cv_mae_sd_M": "fusion/selection:cv_mae_sd_M",
    "final.test_mae_M": "final_eval/test_metrics:chosen.mae_M",
    "final.test_mdape": "final_eval/test_metrics:chosen.mdape",
    "final.test_r2": "final_eval/test_metrics:chosen.r2",
    "final.within20": "final_eval/test_metrics:chosen.within20",
    "nov.n": "final_eval/nov_holdout:n",
    "nov.mae_M": "final_eval/nov_holdout:chosen.mae_M",
    "nov.mdape": "final_eval/nov_holdout:chosen.mdape",
    "nov.base_mae_M": "final_eval/nov_holdout:base.mae_M",
    "nov.base_mdape": "final_eval/nov_holdout:base.mdape",
}

_PLACEHOLDER = re.compile(r"\{\{([^{}|]+?)(?:\|([^{}]+))?\}\}")
_TABLE_KEY = re.compile(r"^(?P<file>[\w./-]+?)(?:(?P<best>@best)|\[(?P<match>[^\]]+)\]):(?P<col>.+)$")
_COUNT_KEY = re.compile(r"^(?P<file>[\w./-]+?)@(?P<what>n|n_pass)$")
_JSON_KEY = re.compile(r"^(?P<file>[\w./-]+?):(?P<path>.+)$")
MISSING = object()


def _is_base(name: Any) -> bool:
    return str(name).lower().startswith("base") and "+" not in str(name)


@dataclass
class Values:
    """Lazy, cached resolver of placeholder keys against an outputs directory."""
    out: Path = OUT
    extra: dict[str, Any] = field(default_factory=dict)
    _json: dict[str, Any] = field(default_factory=dict, repr=False)
    _csv: dict[str, pd.DataFrame | None] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.extra.update(self._derived())

    # ---------------------------------------------------------------- files
    def json(self, rel: str) -> Any:
        if rel not in self._json:
            p = self.out / f"{rel}.json"
            try:
                self._json[rel] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
            except (OSError, ValueError):
                self._json[rel] = None
        return self._json[rel]

    def csv(self, rel: str) -> pd.DataFrame | None:
        if rel not in self._csv:
            p = self.out / f"{rel}.csv"
            try:
                self._csv[rel] = pd.read_csv(p) if p.exists() else None
            except (OSError, ValueError, pd.errors.ParserError):
                self._csv[rel] = None
        return self._csv[rel]

    def exists(self, rel_path: str) -> bool:
        return (self.out / rel_path).exists()

    # ---------------------------------------------------------------- derived numbers
    def _derived(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        b = self.json("core/base_cv")
        if b:
            d["base.threshold_M"] = b["cv_mae_M"] - b["cv_mae_sd_M"]
        for track in ("geo/cv", "text/cv", "photos/price_cv", "fusion/selection"):
            t = self.csv(track)
            if t is not None and "cv_mae_M" in t:
                rest = t[~t.iloc[:, 0].map(_is_base)]
                d[f"{track}@n"] = len(rest)
                if "passes_1sd_rule" in rest:
                    d[f"{track}@n_pass"] = int(rest["passes_1sd_rule"].astype(str).str.lower().eq("true").sum())
        return d

    # ---------------------------------------------------------------- resolution
    def get(self, key: str) -> Any:
        key = key.strip()
        if key in self.extra:
            return self.extra[key]
        key = ALIASES.get(key, key)
        if key in self.extra:
            return self.extra[key]
        m = _TABLE_KEY.match(key)
        if m:
            return self._table_value(m["file"], m["best"], m["match"], m["col"])
        m = _COUNT_KEY.match(key)
        if m:
            return self.extra.get(key, MISSING)
        m = _JSON_KEY.match(key)
        if m:
            return _dig(self.json(m["file"]), m["path"])
        return MISSING

    def _table_value(self, rel: str, best: str | None, match: str | None, col: str) -> Any:
        t = self.csv(rel)
        if t is None or t.empty or col not in t:
            return MISSING
        if best:
            if "cv_mae_M" not in t:
                return MISSING
            rest = t[~t.iloc[:, 0].map(_is_base)]
            if rest.empty:
                return MISSING
            row = rest.loc[rest["cv_mae_M"].idxmin()]
        else:
            hit = t[t.iloc[:, 0].astype(str).str.lower().str.contains(match.lower(), regex=False)]
            if hit.empty:
                return MISSING
            row = hit.iloc[0]
        v = row[col]
        return MISSING if isinstance(v, float) and math.isnan(v) else _py(v)

    def has(self, key: str) -> bool:
        return self.get(key) is not MISSING


def _py(v: Any) -> Any:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def _dig(obj: Any, path: str) -> Any:
    """Follow a dotted path; keys may themselves contain dots (e.g. '0.9'), so try the longest match first."""
    if obj is None:
        return MISSING
    parts = path.split(".")
    i = 0
    while i < len(parts):
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if isinstance(obj, dict) and k in obj:
                obj, i = obj[k], j
                break
            if isinstance(obj, list) and j == i + 1 and k.lstrip("-").isdigit() and -len(obj) <= int(k) < len(obj):
                obj, i = obj[int(k)], j
                break
        else:
            return MISSING
    return MISSING if obj is None else obj


def fmt_value(v: Any, spec: str | None = None) -> str:
    if spec == "pct":
        return f"{100 * float(v):.1f}"
    if spec:
        try:
            return format(v, spec)
        except (TypeError, ValueError):
            return str(v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
    if isinstance(v, list):
        return ", ".join(fmt_value(x) for x in v)
    return str(v)


def fill(text: str, values: Values, missing: str = "keep") -> str:
    """Replace placeholders. missing='keep' leaves `{{key}}`, 'pending' writes `pending`."""
    def sub(m: re.Match) -> str:
        v = values.get(m[1])
        if v is MISSING:
            return m[0] if missing == "keep" else "pending"
        return fmt_value(v, m[2].strip() if m[2] else None)
    return _PLACEHOLDER.sub(sub, text)


def placeholders(text: str) -> list[str]:
    return [m[0] for m in _PLACEHOLDER.finditer(text)]


def pending_note(paths: list[str], values: Values) -> str | None:
    """Markdown note naming the outputs that are not there yet, or None when all exist."""
    gone = [p for p in paths if not values.exists(p)]
    if not gone:
        return None
    who = sorted({PRODUCERS.get(p.split("/")[0], "the track scripts") for p in gone})
    files = ", ".join(f"`outputs/{p}`" for p in gone)
    return f"> **Pending.** Not produced yet: {files} (written by {'; '.join(who)}). The cells below show a pending note until then."


def cv_view(t: pd.DataFrame | None) -> pd.DataFrame | None:
    """The common CV-table view: variant, CV MAE mean ± SD, MdAPE, delta, folds better, 1-SD rule."""
    if t is None or "cv_mae_M" not in t:
        return None
    v = pd.DataFrame({"variant": t.iloc[:, 0]})
    v["CV MAE (M KZT)"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(t["cv_mae_M"], t.get("cv_mae_sd_M", 0 * t["cv_mae_M"]))]
    if "cv_mdape" in t:
        v["CV MdAPE %"] = t["cv_mdape"].round(2)
    if "delta_vs_base_M" in t:
        v["Δ vs base (M)"] = t["delta_vs_base_M"].round(3)
    if "folds_better" in t:
        v["folds better"] = t["folds_better"].map(lambda x: f"{int(x)}/5" if pd.notna(x) else "")
    if "passes_1sd_rule" in t:
        v["kept (1-SD rule)"] = t["passes_1sd_rule"].astype(str).str.lower().map({"true": "yes", "false": "no"}).fillna("")
    return v.set_index("variant")


def track_status(values: Values) -> pd.DataFrame:
    """One row per track: which of its key outputs exist."""
    rows = []
    for track, files in KEY_OUTPUTS.items():
        have = [f for f in files if values.exists(f)]
        rows.append({"track": track, "outputs present": f"{len(have)}/{len(files)}",
                     "missing": ", ".join(f for f in files if f not in have) or "-"})
    return pd.DataFrame(rows).set_index("track")


KEY_OUTPUTS = {
    "core (Endterm reproduction)": ["core/base_cv.json", "core/endterm_repro_test.json"],
    "leakage audit": ["leakage/leakage_summary.json", "leakage/folds_group.csv", "leakage/cv_schemes_summary.csv"],
    "location": ["geo/cv.csv", "geo/features_geo.csv", "geo/residual_map.png"],
    "photos": ["photos/clip_zero_shot_ad.csv", "photos/classifier_comparison.csv", "photos/price_cv.csv"],
    "text": ["text/cv.csv", "text/flags_stats.csv", "text/features_text.csv"],
    "trust": ["trust/chosen_settings.json", "trust/interval_methods.csv", "trust/shap_global_importance.csv"],
    "fusion": ["fusion/selection.csv", "fusion/selection.json"],
    "final evaluation": ["final_eval/test_metrics.json", "final_eval/nov_holdout.json"],
}


def first_existing(values: Values, *paths: str) -> str | None:
    return next((p for p in paths if values.exists(p)), None)


BLOCK_TABLES = {
    "location (OSM + raw fields)": "geo/cv",
    "photos": "photos/price_cv",
    "text": "text/cv",
    "fusion of kept blocks": "fusion/selection",
}


def blocks_summary(values: Values) -> pd.DataFrame:
    """Best variant of every block by CV MAE, against the Endterm base and the 1-SD rule."""
    rows = []
    for block, rel in BLOCK_TABLES.items():
        t = values.csv(rel)
        if t is None or "cv_mae_M" not in t:
            rows.append({"block": block, "best variant": "pending", "CV MAE (M KZT)": "", "Δ vs base (M)": "",
                         "folds better": "", "variants kept": ""})
            continue
        rest = t[~t.iloc[:, 0].map(_is_base)]
        if rest.empty:
            continue
        r = rest.loc[rest["cv_mae_M"].idxmin()]
        n_pass = values.get(f"{rel}@n_pass")
        rows.append({"block": block, "best variant": str(r.iloc[0]),
                     "CV MAE (M KZT)": f"{r['cv_mae_M']:.3f} ± {r.get('cv_mae_sd_M', float('nan')):.3f}",
                     "Δ vs base (M)": f"{r['delta_vs_base_M']:+.3f}" if "delta_vs_base_M" in r else "",
                     "folds better": f"{int(r['folds_better'])}/5" if pd.notna(r.get("folds_better")) else "",
                     "variants kept": f"{n_pass if n_pass is not MISSING else '?'}/{len(rest)}"})
    return pd.DataFrame(rows).set_index("block")
