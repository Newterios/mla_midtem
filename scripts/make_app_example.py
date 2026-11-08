"""Render one app prediction as a static card for the slides: outputs/final/app_example.png.

    .venv/bin/python scripts/make_app_example.py

Runs app.predict.Predictor (final model + trust layer + text flags + photo condition) on one
hand-made listing (not a test ad) with two training-ad photos, and draws the result with matplotlib.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.predict import Predictor  # noqa: E402

BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
LISTING = {"city": "Astana", "district": "Esilyskiy", "area": 62, "rooms": 2, "floor": 5, "total_floors": 12,
           "year_built": 2019, "building_type": "monolithic", "condition": "fresh_renovation", "has_complex": True,
           "asking_price_m": 52, "lat": 51.128, "lon": 71.43,
           "description": "Продается светлая двухкомнатная квартира в новом ЖК, свежий ремонт, панорамные окна, "
                          "без обременений. Срочно, торг."}
PHOTOS = [ROOT / "data" / "photos" / "1000184327_1.jpg", ROOT / "data" / "photos" / "1000184327_2.jpg"]


def main() -> None:
    pr = Predictor()
    r = pr.predict(LISTING, photos=[p for p in PHOTOS if p.exists()])
    fig = plt.figure(figsize=(11, 5.2), facecolor="white")
    fig.text(0.02, 0.94, "Krisha price check: one listing through the Final app", fontsize=14, weight="bold", color=INK)
    fig.text(0.02, 0.895, "Astana, Esil district, 62 m², 2 rooms, 5/12 floor, built 2019, monolithic, fresh renovation; "
             "asking 52 M KZT; 2 photos; description with red-flag words", fontsize=9, color=MUTED)

    # price + intervals
    ax = fig.add_axes([0.04, 0.50, 0.42, 0.22])
    lo90, hi90 = (v / 1e6 for v in r.intervals["90"])
    lo80, hi80 = (v / 1e6 for v in r.intervals["80"])
    ax.barh(0, hi90 - lo90, left=lo90, height=0.5, color=BLUE, alpha=0.25, label="90 % interval")
    ax.barh(0, hi80 - lo80, left=lo80, height=0.5, color=BLUE, alpha=0.5, label="80 % interval")
    ax.plot([r.price_m], [0], "o", color=INK, ms=8, label=f"estimate {r.price_m:.1f} M")
    ax.axvline(LISTING["asking_price_m"], color=ORANGE, lw=2, label=f"asking {LISTING['asking_price_m']} M")
    if r.base_price_kzt:
        ax.plot([r.base_price_kzt / 1e6], [0], "|", color=MUTED, ms=18, mew=2, label=f"Endterm base {r.base_price_kzt / 1e6:.1f} M")
    ax.set_yticks([]); ax.set_xlabel("M KZT"); ax.grid(axis="y", visible=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.02))
    v = r.verdict
    fig.text(0.04, 0.34, f"Verdict: {v['text']} (asking {v['gap_pct']:+.0f} % vs estimate)", fontsize=10.5, weight="bold",
             color=ORANGE if v["code"] in ("over", "high") else AQUA if v["code"] == "fair" else BLUE)

    # SHAP
    ax2 = fig.add_axes([0.62, 0.43, 0.35, 0.40])
    sh = r.shap.head(7).iloc[::-1]
    ax2.barh(sh["label"], sh["effect_pct"], color=[AQUA if x > 0 else ORANGE for x in sh["effect_pct"]])
    ax2.axvline(0, color=INK, lw=0.8)
    ax2.set_xlabel("effect on the estimate, % (SHAP, Endterm base part)", fontsize=8)
    ax2.tick_params(labelsize=8)
    ax2.set_title("Why this price", loc="left", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    # flags, photo, comparables
    lines = ["Text flags: " + "; ".join(f["label"].split(" (")[0] for f in r.text_flags[:5])]
    p = r.photo or {}
    if p.get("status") == "ok":
        lines.append(f"Photo condition: {p['predicted_label']} ({p['confidence']:.0%}); stated: {p['stated']}"
                     + (f" -> MISMATCH: {p['mismatch']}" if p.get("mismatch") else f" -> {p.get('comment', '')}"))
    if r.comparables is not None:
        c = r.comparables
        lines.append(f"5 nearest training ads: median {np.median(c['M per m²']) * 1000:.0f} k KZT/m² -> "
                     f"{r.comparables_estimate_kzt / 1e6:.1f} M for 62 m²; nearest {c['km away'].min():.1f} km away"
                     + ("; few comparables warning" if r.few_comparables else ""))
    lines += [f"Warning: {w}" for w in r.warnings[:1]]
    fig.text(0.04, 0.05, "\n".join(lines), fontsize=9, color=INK, va="bottom", linespacing=1.6)
    out = ROOT / "outputs" / "final" / "app_example.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"{out}: estimate {r.price_m:.2f} M, verdict {v['code']}")


if __name__ == "__main__":
    main()
