"""Gradio demo of the krisha.kz asking-price model (Final, SE-2424).

    .venv/bin/python app/gradio_app.py            # http://127.0.0.1:7860
    .venv/bin/python app/gradio_app.py --port 7870 --share

All model logic lives in app/predict.py; this file only builds the form and renders the result.
"""
from __future__ import annotations

import argparse
import html
import logging
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app.predict import (CHOICES, CITIES, DISTRICTS, MAX_PHOTOS, Prediction, Predictor,  # noqa: E402
                         ValidationError, validate)

log = logging.getLogger("krisha_app")

# --------------------------------------------------------------------------- look
# Direction: a printed property sheet, like the classifieds page of a newspaper. Warm paper, black ink,
# square corners, hairline rules, a serif for headings and the price, one brick-red accent for
# "your asking price". The range ruler places the asking price against the 80 % / 90 % intervals.
INK, MUTED, LINE, BG, SURFACE = "#1b1a17", "#6b665c", "#cfc8b8", "#f4efe4", "#fbf8f1"
PRIMARY, PRIMARY_SOFT, GOLD = "#2f4a3a", "#dfe5d6", "#b5432a"
GOOD, WARN, BAD = "#2f4a3a", "#8a5a12", "#9c2f1f"
SERIF = "'Iowan Old Style', 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif"
SANS = "'Gill Sans', 'Gill Sans MT', 'Trebuchet MS', Verdana, sans-serif"
MONO = "'Courier Prime', 'Courier New', Courier, monospace"

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(c50="#f1f4ec", c100="#dfe5d6", c200="#c3cfb5", c300="#9fb08f", c400="#6f8a68",
                                c500="#2f4a3a", c600="#294233", c700="#22382b", c800="#1b2d22", c900="#14221a",
                                c950="#0d1711"),
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_none,
    font=["Gill Sans", "Gill Sans MT", "Trebuchet MS", "Verdana", "sans-serif"],
    font_mono=["Courier Prime", "Courier New", "Courier", "monospace"],
).set(
    body_background_fill=BG, body_background_fill_dark="#171512",
    block_background_fill=SURFACE, block_background_fill_dark="#1f1c18",
    block_border_color=LINE, block_border_color_dark="#3a352d",
    block_shadow="none", block_border_width="1px",
    block_label_text_weight="600", block_title_text_weight="600",
    input_background_fill="#ffffff", input_border_color=LINE, input_shadow="none", input_shadow_focus="none",
    button_primary_background_fill=INK, button_primary_background_fill_hover=GOLD,
    button_primary_text_color="#fbf8f1", button_primary_shadow="none",
)

CSS = f"""
:root {{ --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --surface:{SURFACE}; --primary:{PRIMARY};
        --primary-soft:{PRIMARY_SOFT}; --gold:{GOLD}; --good:{GOOD}; --warn:{WARN}; --bad:{BAD};
        --good-soft:#e6ebdc; --warn-soft:#f3e7cc; --bad-soft:#f1dcd4; }}
.dark {{ --ink:#efe9dc; --muted:#a9a293; --line:#3a352d; --surface:#1f1c18; --primary:#9fb08f;
         --primary-soft:#2a3327; --gold:#e07a5f; --good:#9fb08f; --warn:#d9a94e; --bad:#e58b7b;
         --good-soft:#232b20; --warn-soft:#332a17; --bad-soft:#3a211c; }}
.gradio-container {{ max-width: 1240px !important; margin: 0 auto; font-family: {SANS}; }}
.gradio-container button.primary {{ text-transform: uppercase; letter-spacing: .12em; font-weight: 600; border-radius: 0; }}
.kp-num {{ font-family: {MONO}; font-variant-numeric: tabular-nums; }}
.kp-head {{ padding:10px 2px 12px; border-top:3px solid var(--ink); border-bottom:1px solid var(--ink); margin-bottom:10px; }}
.kp-head .kp-over {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.22em; color:var(--gold); font-weight:700; }}
.kp-head h1 {{ font-family:{SERIF}; font-size:2.3rem; font-weight:700; letter-spacing:-0.01em; color:var(--ink);
              margin:2px 0 4px; line-height:1.05; }}
.kp-head p {{ color:var(--muted); margin:0; font-size:.95rem; max-width:62ch; }}
.kp-card {{ color:var(--ink); }}
.kp-label {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.2em; color:var(--muted); font-weight:700;
            border-bottom:1px solid var(--line); padding-bottom:4px; margin-bottom:6px; }}
.kp-price {{ font-family:{SERIF}; font-size:3.4rem; font-weight:700; line-height:1.05; margin:2px 0 4px; }}
.kp-price small {{ font-size:1rem; color:var(--muted); font-weight:500; margin-left:6px; }}
.kp-sub {{ color:var(--muted); font-size:.9rem; }}
.kp-ruler {{ position:relative; height:46px; margin:22px 0 26px; }}
.kp-ruler .track {{ position:absolute; top:22px; left:0; right:0; height:1px; background:var(--ink); }}
.kp-ruler .b90 {{ position:absolute; top:16px; height:14px; background:var(--primary-soft); border:1px solid var(--primary); }}
.kp-ruler .b80 {{ position:absolute; top:16px; height:14px; background:var(--primary); opacity:.75; }}
.kp-ruler .est {{ position:absolute; top:6px; width:2px; height:34px; background:var(--ink); margin-left:-1px; }}
.kp-ruler .ask {{ position:absolute; top:4px; width:14px; height:14px; margin-left:-7px; transform:rotate(45deg);
                  background:var(--gold); border:2px solid var(--surface); box-shadow:0 0 0 1px var(--gold); }}
.kp-ruler .tick {{ position:absolute; top:34px; font-size:.72rem; color:var(--muted); transform:translateX(-50%); white-space:nowrap; }}
.kp-ruler .tick.l {{ transform:none; }} .kp-ruler .tick.r {{ transform:translateX(-100%); }}
.kp-legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:.8rem; color:var(--muted); }}
.kp-legend i {{ display:inline-block; width:12px; height:10px; margin-right:5px; vertical-align:-1px; }}
.kp-verdict {{ margin-top:14px; padding:10px 14px; border:1px solid; border-left-width:6px; font-family:{SERIF};
              font-size:1.1rem; font-weight:700; }}
.kp-verdict span {{ font-family:{SANS}; }}
.kp-verdict span {{ display:block; font-weight:400; font-size:.88rem; color:var(--muted); margin-top:2px; }}
.v-fair {{ background:var(--good-soft); border-color:var(--good); }}
.v-high, .v-low {{ background:var(--warn-soft); border-color:var(--warn); }}
.v-over, .v-under {{ background:var(--bad-soft); border-color:var(--bad); }}
.kp-list {{ list-style:none; padding:0; margin:6px 0 0; }}
.kp-list li {{ padding:7px 10px 7px 12px; margin:0; border-bottom:1px solid var(--line); border-left:4px solid var(--line); font-size:.9rem; }}
.kp-list li.red {{ border-color:var(--bad); background:var(--bad-soft); }}
.kp-list li.amber {{ border-color:var(--warn); background:var(--warn-soft); }}
.kp-list li.good {{ border-color:var(--good); background:var(--good-soft); }}
.kp-list li.info {{ border-color:var(--primary); background:var(--primary-soft); }}
.kp-list li.note {{ border-color:var(--line); color:var(--muted); }}
.kp-bars {{ margin-top:8px; }}
.kp-bars div {{ display:grid; grid-template-columns: minmax(120px, 42%) 1fr 48px; gap:8px; align-items:center;
               font-size:.85rem; margin:4px 0; }}
.kp-bars b {{ display:block; height:10px; background:var(--primary); }}
.kp-bars .hit b {{ background:var(--gold); }}
.kp-error {{ padding:12px 14px; background:var(--bad-soft); border:1px solid var(--bad); border-left-width:6px; color:var(--ink); }}
.kp-error ul {{ margin:6px 0 0 18px; }}
.kp-empty {{ color:var(--muted); padding:18px 4px; font-family:{SERIF}; font-style:italic; font-size:1.05rem; }}
.kp-status td {{ padding:3px 10px 3px 0; vertical-align:top; font-size:.85rem; }}
.kp-status td:first-child {{ font-weight:700; white-space:nowrap; }}
"""

E = html.escape


def _m(kzt: float, digits: int = 1) -> str:
    return f"{kzt / 1e6:,.{digits}f}"


def _choices(field: str) -> list[tuple[str, str]]:
    return [(code.replace("_", " ").capitalize(), code) for code in CHOICES[field]]


def _districts(city: str) -> list[tuple[str, str]]:
    return [(d, d) for d in DISTRICTS.get(city, ())] + [("Not stated", "unknown")]


# --------------------------------------------------------------------------- renderers
def render_price(r: Prediction) -> str:
    parts = [f"<div class='kp-card' aria-live='polite'><div class='kp-label'>Estimated asking price</div>"
             f"<div class='kp-price kp-num'>{_m(r.price_kzt)}<small>M KZT</small></div>"
             f"<div class='kp-sub'>≈ <span class='kp-num'>{round(r.price_kzt, -5):,.0f}</span> ₸ · {E(r.model_name)}</div>"]
    if r.base_price_kzt:
        parts.append(f"<div class='kp-sub'>Endterm base model alone: {r.base_price_kzt / 1e6:.1f} M ₸</div>")
    asking = r.verdict["asking_kzt"] if r.verdict else None
    if r.intervals:
        lo90, hi90 = r.intervals["90"]
        lo80, hi80 = r.intervals["80"]
        vals = [lo90, hi90] + ([asking] if asking else [])
        a, b = min(vals) * 0.97, max(vals) * 1.03
        pos = lambda v: 100 * (v - a) / (b - a)
        parts.append(
            "<div class='kp-ruler' role='img' aria-label='"
            f"80 % range {_m(lo80)} to {_m(hi80)} M, 90 % range {_m(lo90)} to {_m(hi90)} M"
            + (f", asking {_m(asking)} M" if asking else "") + "'>"
            "<div class='track'></div>"
            f"<div class='b90' style='left:{pos(lo90):.2f}%;width:{pos(hi90) - pos(lo90):.2f}%'></div>"
            f"<div class='b80' style='left:{pos(lo80):.2f}%;width:{pos(hi80) - pos(lo80):.2f}%'></div>"
            f"<div class='est' style='left:{pos(r.price_kzt):.2f}%'></div>"
            + (f"<div class='ask' style='left:{pos(asking):.2f}%'></div>" if asking else "")
            + f"<div class='tick kp-num l' style='left:{pos(lo90):.2f}%'>{_m(lo90)}</div>"
            f"<div class='tick kp-num r' style='left:{pos(hi90):.2f}%'>{_m(hi90)}</div></div>"
            "<div class='kp-legend'>"
            f"<span><i style='background:var(--primary);opacity:.55'></i>80 %: <b class='kp-num'>{_m(lo80)}–{_m(hi80)}</b> M</span>"
            f"<span><i style='background:var(--primary-soft)'></i>90 %: <b class='kp-num'>{_m(lo90)}–{_m(hi90)}</b> M</span>"
            + ("<span><i style='background:var(--gold);transform:rotate(45deg);width:9px;height:9px'></i>your asking price</span>"
               if asking else "") + "</div>"
            f"<div class='kp-sub' style='margin-top:6px'>Intervals: {E(r.interval_source or '')}</div>")
    if r.comparables_estimate_kzt:
        parts.append(f"<div class='kp-sub'>5 comparables (median price per m² × area): "
                     f"<span class='kp-num'>{_m(r.comparables_estimate_kzt)}</span> M</div>")
    if r.verdict:
        v = r.verdict
        parts.append(f"<div class='kp-verdict v-{v['code']}'>{E(v['text'])}"
                     f"<span>Asking <b class='kp-num'>{_m(v['asking_kzt'])}</b> M is "
                     f"<b class='kp-num'>{v['gap_pct']:+.1f} %</b> vs the estimate ({E(v['rule'])}).</span></div>")
    else:
        parts.append("<div class='kp-sub' style='margin-top:10px'>Enter your asking price to get an over/under-pricing verdict.</div>")
    return "".join(parts) + "</div>"


def render_alerts(r: Prediction) -> str:
    items = [("red" if "Photo vs stated" in w else "amber", w) for w in r.warnings] + [("note", n) for n in r.notes]
    if not items:
        return "<div class='kp-card kp-sub'>No warnings for this input.</div>"
    lis = "".join(f"<li class='{c}'>{E(t)}</li>" for c, t in items)
    return f"<div class='kp-card'><div class='kp-label'>Warnings and notes</div><ul class='kp-list'>{lis}</ul></div>"


def render_flags(r: Prediction, has_text: bool) -> str:
    if not has_text:
        return "<div class='kp-card kp-empty'>Paste the listing description to scan it for red flags.</div>"
    if not r.text_flags:
        return "<div class='kp-card'><div class='kp-label'>Description scan</div><ul class='kp-list'>" \
               "<li class='good'>No red flags found in the description.</li></ul></div>"
    lis = "".join(f"<li class='{E(f['severity'])}'>{E(f['label'])}</li>" for f in r.text_flags)
    return f"<div class='kp-card'><div class='kp-label'>Description scan</div><ul class='kp-list'>{lis}</ul></div>"


def render_photo(r: Prediction) -> str:
    p = r.photo or {}
    status = p.get("status")
    if status != "ok":
        msg = {"no photos": f"Upload up to {MAX_PHOTOS} interior photos to check the stated condition.",
               "no readable photos": "None of the uploaded files could be read as an image."}.get(
            status, f"Photo check unavailable: {status}.")
        return f"<div class='kp-card kp-empty'>{E(msg)}</div>"
    labels = {"fresh": "Fresh renovation", "good": "Lived-in, tidy", "needs_repair": "Needs repair", "shell": "Shell / rough"}
    bars = "".join(
        f"<div class='{'hit' if c == p['predicted'] else ''}'><span>{E(labels.get(c, c))}</span>"
        f"<span><b style='width:{100 * v:.1f}%'></b></span><span class='kp-num'>{100 * v:.0f}%</span></div>"
        for c, v in p["proba"].items())
    if p.get("mismatch"):
        verdict = f"<div class='kp-verdict v-over'>Photo vs stated condition mismatch<span>{E(p['mismatch'])}</span></div>"
    else:
        verdict = f"<div class='kp-sub' style='margin-top:8px'>{E(p.get('comment', ''))}</div>"
    return (f"<div class='kp-card'><div class='kp-label'>Condition from {p['n_photos']} photo(s)</div>"
            f"<div style='font-weight:700;margin-top:4px'>{E(p['predicted_label'])} "
            f"<span class='kp-num kp-sub'>{p['confidence']:.0%}</span></div><div class='kp-bars'>{bars}</div>{verdict}</div>")


def shap_figure(r: Prediction, top: int = 8):
    if r.shap is None or r.shap.empty:
        return None
    t = r.shap.head(top).iloc[::-1]
    vals = t["effect_pct"].to_numpy()
    labels = [f"{lab}  ({_fmt_value(v)})" for lab, v in zip(t["label"], t["value"])]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(t) + 0.9), dpi=110)
    ax.barh(labels, vals, color=[PRIMARY if v >= 0 else BAD for v in vals], height=0.62)
    ax.axvline(0, color=INK, lw=0.8)
    for y, v in enumerate(vals):
        ax.text(v + (0.25 if v >= 0 else -0.25), y, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=8.5, color=INK)
    lim = max(abs(vals).max() * 1.35, 2)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("effect on the estimate, %  (green raises, red lowers)", fontsize=8.5, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="y", length=0, labelsize=8.5, colors=INK)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    fig.suptitle("Top reasons (exact TreeSHAP vs the average ad)", x=0.02, ha="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout()
    plt.close(fig)
    return fig


def _fmt_value(v: str) -> str:
    v = str(v)
    if v in ("nan", "", "None"):
        return "not given"
    if "nan" in v.split(", "):                       # "nan, nan, Esilyskiy" (no coordinates given)
        v = ", ".join(p for p in v.split(", ") if p != "nan") or "not given"
    if v.count(",") == 1 and v.split(",")[1].strip().replace(".", "").isdigit():    # "62, 4.127" (area, log area)
        return v.split(",")[0] + " m²"
    return v.replace("_", " ")[:28]


def comparables_table(r: Prediction) -> pd.DataFrame:
    if r.comparables is None:
        return pd.DataFrame({"comparables": ["module not available yet"]})
    t = r.comparables.copy()
    t["link"] = [f"[open ↗]({u})" for u in t["link"]]
    return t.drop(columns="rank")       # "km away" stays (empty without coordinates) so the column types line up


def render_status(status: dict[str, str]) -> str:
    rows = "".join(f"<tr><td>{E(k)}</td><td>{E(v)}</td></tr>" for k, v in status.items())
    return f"<table class='kp-status'>{rows}</table>"


def render_error(errors: list[str], title: str = "Please fix the form") -> str:
    lis = "".join(f"<li>{E(e)}</li>" for e in errors)
    return f"<div class='kp-error' role='alert'><b>{E(title)}</b><ul>{lis}</ul></div>"


# --------------------------------------------------------------------------- UI
FIELDS = ["city", "district", "area", "rooms", "floor", "total_floors", "year_built", "building_type", "condition",
          "kitchen_area", "ceiling_height", "bathroom", "furnished", "balcony", "parking", "former_dorm", "seller_type",
          "n_photos", "has_complex", "mortgaged", "is_urgent", "lat", "lon", "description", "asking_price_m"]

EXAMPLES = [
    ["Astana", "Esilyskiy", 62, 2, 7, 12, 2019, "monolithic", "fresh_renovation", 12, 3.0, "combined", "fully", "has",
     "parking", "no", "owner", 18, True, False, False, 51.1284, 71.4306,
     "Продается 2-комнатная квартира в ЖК на Левом берегу. Свежий ремонт, вся мебель и техника остаются. "
     "Без обременений, документы готовы.", 47],
    ["Almaty", "Bostandykskiy", 78, 3, 3, 5, 1978, "brick", "good", 8, 2.7, "separate", "partly", "has",
     "street", "no", "specialist", 12, False, False, True, None, None,
     "Срочно! Квартира по переуступке, только наличный расчет. Требует косметического ремонта. Торг.", 36],
]


def build_ui(predictor: Predictor) -> gr.Blocks:
    def run(*values):
        *form, photos = values
        raw = dict(zip(FIELDS, form))
        empty = (None, pd.DataFrame(), "", "", "")
        try:
            listing = validate(raw)
            res = predictor.predict(listing, photos=photos or [])
        except ValidationError as e:
            return (render_error(e.errors),) + empty
        except Exception as e:   # never show a traceback to the user
            log.exception("prediction failed")
            return (render_error([f"{type(e).__name__}: {e}"], "Something went wrong while predicting"),) + empty
        return (render_price(res), shap_figure(res), comparables_table(res), render_flags(res, bool(listing.description)),
                render_photo(res), render_alerts(res))

    with gr.Blocks(title="Is this flat over-priced?", analytics_enabled=False) as demo:
        gr.HTML("<header class='kp-head'><div class='kp-over'>Astana &middot; Almaty &middot; October 2026</div>"
                "<h1>Is this flat over-priced?</h1>"
                "<p>Describe an apartment the way it is listed on krisha.kz and compare its asking price with what "
                "similar flats ask. The model was trained on 3,852 listings.</p></header>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=340):
                with gr.Group():
                    with gr.Row():
                        city = gr.Dropdown(list(CITIES), value="Astana", label="City")
                        district = gr.Dropdown(_districts("Astana"), value="Esilyskiy", label="District",
                                               allow_custom_value=True)
                    with gr.Row():
                        area = gr.Number(62, label="Total area, m²")
                        rooms = gr.Number(2, label="Rooms", precision=0)
                    with gr.Row():
                        floor = gr.Number(5, label="Floor", precision=0)
                        total_floors = gr.Number(12, label="Floors in building", precision=0)
                        year_built = gr.Number(2018, label="Year built", precision=0)
                    with gr.Row():
                        building_type = gr.Dropdown(_choices("building_type"), value="monolithic", label="Building type")
                        condition = gr.Dropdown(_choices("condition"), value="fresh_renovation", label="Condition")
                    asking = gr.Number(None, label="Asking price, M KZT (optional)")
                with gr.Accordion("More details", open=False):
                    with gr.Row():
                        kitchen_area = gr.Number(None, label="Kitchen area, m²")
                        ceiling_height = gr.Number(None, label="Ceiling height, m")
                        n_photos = gr.Number(None, label="Photos in the listing", precision=0,
                                             info="Empty = typical listing (15)")
                    with gr.Row():
                        bathroom = gr.Dropdown(_choices("bathroom"), value="unknown", label="Bathroom")
                        furnished = gr.Dropdown(_choices("furnished"), value="unknown", label="Furniture")
                        balcony = gr.Dropdown(_choices("balcony"), value="unknown", label="Balcony")
                    with gr.Row():
                        parking = gr.Dropdown(_choices("parking"), value="unknown", label="Parking")
                        former_dorm = gr.Dropdown(_choices("former_dorm"), value="no", label="Former dormitory")
                        seller_type = gr.Dropdown(_choices("seller_type"), value="owner", label="Seller")
                    with gr.Row():
                        has_complex = gr.Checkbox(True, label="In a residential complex")
                        mortgaged = gr.Checkbox(False, label="Mortgaged / pledged")
                        is_urgent = gr.Checkbox(False, label="Urgent sale")
                    with gr.Row():
                        lat = gr.Number(None, label="Latitude", info="Optional, e.g. 51.1284 (right-click on 2GIS / Google Maps)")
                        lon = gr.Number(None, label="Longitude", info="Optional, e.g. 71.4306")
                description = gr.Textbox(label="Listing description", lines=4, max_length=10_000,
                                         placeholder="Paste the description as it appears in the listing (krisha.kz "
                                                     "listings are written in Russian or Kazakh)")
                photos = gr.File(label=f"Up to {MAX_PHOTOS} interior photos", file_count="multiple",
                                 file_types=["image"], type="filepath")
                go = gr.Button("Estimate price", variant="primary", size="lg")
            with gr.Column(scale=7, min_width=340):
                price_html = gr.HTML("<div class='kp-empty'>Fill in the sheet on the left and press Estimate price.</div>")
                alerts_html = gr.HTML()
                shap_plot = gr.Plot(label="Why this price", show_label=False)
                comps = gr.Dataframe(label="5 comparable training ads (same city)", interactive=False, wrap=True,
                                     datatype=["number", "number", "number", "number", "number", "number", "number", "str",
                                               "number", "markdown"])
                with gr.Row():
                    flags_html = gr.HTML()
                    photo_html = gr.HTML()
                with gr.Accordion("Model status: what runs on what", open=False):
                    gr.HTML(render_status(predictor.status()))
                    gr.Markdown("Prices are **asking** prices of October 2026 krisha.kz ads, not sale prices. "
                                "Intervals are conformal (about 80 % / 90 % of similar ads fall inside). "
                                "Team SE-2424: Rakhat Balgabekov, Aitbek Kaiyrbekuly.")

        inputs = [city, district, area, rooms, floor, total_floors, year_built, building_type, condition, kitchen_area,
                  ceiling_height, bathroom, furnished, balcony, parking, former_dorm, seller_type, n_photos, has_complex,
                  mortgaged, is_urgent, lat, lon, description, asking]
        def on_city(c, d):   # keep a district that belongs to the new city (e.g. when an example is loaded)
            return gr.Dropdown(choices=_districts(c), value=d if d in DISTRICTS.get(c, ()) else "unknown")

        city.change(on_city, [city, district], district)
        gr.Examples(EXAMPLES, inputs=inputs, label="Examples")
        go.click(run, inputs + [photos], [price_html, shap_plot, comps, flags_html, photo_html, alerts_html],
                 concurrency_limit=2, api_name="estimate")
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--no-photos", action="store_true", help="skip the CLIP photo model (faster start)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    predictor = Predictor(photo_model=None if args.no_photos else "auto")
    for k, v in predictor.status().items():
        log.info("%-12s %s", k, v)
    if predictor.photo_model is not None and hasattr(predictor.photo_model, "warmup"):
        threading.Thread(target=predictor.photo_model.warmup, daemon=True, name="clip-warmup").start()
    build_ui(predictor).launch(server_name=args.host, server_port=args.port, share=args.share, theme=THEME, css=CSS,
                               max_file_size="15mb", footer_links=[])


if __name__ == "__main__":
    main()
