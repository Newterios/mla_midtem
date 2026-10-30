"""Build the Endterm slide deck from the notebook's figures and tables.

    .venv/bin/python tools/build_slides.py

Reads endterm/results_table.csv, endterm/summary.json and the figures/e*.png
files written by the Endterm notebook, and writes slides/Endterm_SE-2424_slides.pptx.
Texts live in CONTENT below so that they can be edited without touching the layout code.
"""
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "slides" / "Endterm_SE-2424_slides.pptx"
results = pd.read_csv(ROOT / "endterm/results_table.csv", index_col="model")
summary = json.loads((ROOT / "endterm/summary.json").read_text())
CONTENT = json.loads((ROOT / "tools/slides_content.json").read_text(encoding="utf-8"))

DARK, GREY, ACCENT, LIGHT = RGBColor(0x1F, 0x29, 0x37), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xB9, 0x1C, 0x1C), RGBColor(0xF3, 0xF4, 0xF6)
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, s, size=14, bold=False, color=DARK, align=PP_ALIGN.LEFT, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(s if isinstance(s, list) else [s]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run(); run.text = line; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color; run.font.name = font
        p.alignment = align; p.space_after = Pt(4)
    return tb


def bullets(slide, x, y, w, h, items, size=14):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        head, _, rest = it.partition("::")
        r = p.add_run(); r.text = "• " + head.strip(); r.font.size = Pt(size); r.font.bold = bool(rest); r.font.color.rgb = DARK; r.font.name = "Calibri"
        if rest:
            r2 = p.add_run(); r2.text = " " + rest.strip(); r2.font.size = Pt(size); r2.font.color.rgb = DARK; r2.font.name = "Calibri"
        p.space_after = Pt(6)
    return tb


def header(slide, title, n):
    bar = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.12)); bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    text(slide, 0.5, 0.25, 12.3, 0.9, title, size=24, bold=True)
    text(slide, 0.5, 7.0, 10, 0.4, "SE-2424 · Balgabekov, Kaiyrbekuly · Endterm · Machine Learning Algorithms", size=10, color=GREY)
    text(slide, 12.3, 7.0, 0.6, 0.4, str(n), size=10, color=GREY, align=PP_ALIGN.RIGHT)


def picture(slide, name, x, y, w=None, h=None):
    p = FIG / name
    if not p.exists():
        text(slide, x, y, 5, 0.5, f"[missing figure {name}]", color=ACCENT); return
    kw = {"width": Inches(w)} if w else {"height": Inches(h)}
    slide.shapes.add_picture(str(p), Inches(x), Inches(y), **kw)


def table(slide, df, x, y, w, h, size=11, highlight=None, first_col=None):
    rows, cols = df.shape[0] + 1, df.shape[1] + 1
    t = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    if first_col:  # wide model column, the rest share the remaining width
        t.columns[0].width = Inches(first_col)
        for j in range(1, cols): t.columns[j].width = Inches((w - first_col) / (cols - 1))
    for r in t.rows: r.height = Inches(h / rows)
    for j, c in enumerate([df.index.name or ""] + list(df.columns)):
        cell = t.cell(0, j); cell.text = str(c); cell.text_frame.paragraphs[0].runs[0].font.size = Pt(size); cell.text_frame.paragraphs[0].runs[0].font.bold = True
    for i, (idx, row) in enumerate(df.iterrows(), 1):
        for j, v in enumerate([idx] + list(row)):
            cell = t.cell(i, j); cell.text = str(v); r = cell.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(size)
            if highlight and idx == highlight: r.font.bold = True; r.font.color.rgb = ACCENT
    return t


# ---------------------------------------------------------------- slides
n = 0
# 1 title
s = prs.slides.add_slide(BLANK); n += 1
text(s, 0.8, 1.6, 11.5, 1.6, CONTENT["title"], size=34, bold=True)
text(s, 0.8, 3.3, 11.5, 1.2, CONTENT["subtitle"], size=18, color=GREY)
text(s, 0.8, 5.2, 11.5, 1.2, ["Course project, Endterm (Week 9) · Machine Learning Algorithms · Direction 1 (Apartments for sale) · Regression",
                                "Rakhat Balgabekov, Aitbek Kaiyrbekuly · Group SE-2424 · November 2026"], size=14, color=GREY)

# 2 question + what changed since the midterm
s = prs.slides.add_slide(BLANK); n += 1; header(s, CONTENT["s2_title"], n)
text(s, 0.5, 1.2, 12.3, 1.0, CONTENT["s2_question"], size=16, bold=True, color=ACCENT)
bullets(s, 0.5, 2.2, 6.0, 4.5, CONTENT["s2_left"], size=13)
bullets(s, 6.8, 2.2, 6.0, 4.5, CONTENT["s2_right"], size=13)

# 3 ensembles: bias/variance + lr curve
s = prs.slides.add_slide(BLANK); n += 1; header(s, CONTENT["s3_title"], n)
picture(s, "e01_boosting_lr_trees.png", 0.4, 1.2, w=6.2)
bullets(s, 6.9, 1.2, 6.0, 5.5, CONTENT["s3_bullets"], size=13)

# 4 tuning + importance
s = prs.slides.add_slide(BLANK); n += 1; header(s, CONTENT["s4_title"], n)
picture(s, "e02_random_search.png", 0.4, 1.15, w=7.6)
picture(s, "e03_importance.png", 0.4, 4.1, w=7.6)
bullets(s, 8.3, 1.2, 4.7, 5.6, CONTENT["s4_bullets"], size=12)

# 5 clustering
s = prs.slides.add_slide(BLANK); n += 1; header(s, CONTENT["s5_title"], n)
picture(s, "e05_segments_map.png", 0.4, 1.15, w=7.4)
picture(s, "e06_dbscan.png", 0.4, 4.2, w=7.4)
bullets(s, 8.0, 1.2, 5.0, 5.6, CONTENT["s5_bullets"], size=12)

# 6 PCA + segment feature
s = prs.slides.add_slide(BLANK); n += 1; header(s, CONTENT["s6_title"], n)
picture(s, "e07_pca.png", 0.4, 1.2, w=8.2)
bullets(s, 8.8, 1.2, 4.3, 5.6, CONTENT["s6_bullets"], size=12)

# 7 neural network
s = prs.slides.add_slide(BLANK); n += 1; header(s, CONTENT["s7_title"], n)
picture(s, "e08_nn_curves.png", 0.4, 1.2, w=12.5)
bullets(s, 0.5, 4.1, 12.3, 3.0, CONTENT["s7_bullets"], size=13)

# 8 results table
s = prs.slides.add_slide(BLANK); n += 1; header(s, CONTENT["s8_title"], n)
show = results.loc[CONTENT["s8_rows"], ["CV MAE (M KZT)", "CV MdAPE %", "Test MAE (M KZT)", "Test MdAPE %", "Test R²", "within ±20 %"]]
show.index = [CONTENT["s8_names"].get(i, i) for i in show.index]; show.index.name = "model"
table(s, show, 0.4, 1.2, 8.5, 0.33 * (len(show) + 1), size=10, highlight=CONTENT["s8_names"].get(summary["best_model"], summary["best_model"]), first_col=2.5)
bullets(s, 9.1, 1.2, 4.0, 5.6, CONTENT["s8_bullets"], size=12)
text(s, 0.4, 6.72, 8.4, 0.3, "CV = 5-fold cross-validation on the 3,852 training rows (mean ± fold SD). Test = the 963 ads saved at the Midterm, used once. Model chosen by CV MAE.", size=9, color=GREY)

# 9 errors
s = prs.slides.add_slide(BLANK); n += 1; header(s, CONTENT["s9_title"], n)
picture(s, "e10_residuals.png", 0.4, 1.15, w=8.0)
picture(s, "e11_overpricing_flags.png", 0.4, 4.0, w=5.2)
bullets(s, 8.6, 1.2, 4.5, 5.6, CONTENT["s9_bullets"], size=12)

# 10 conclusions + plan
s = prs.slides.add_slide(BLANK); n += 1; header(s, CONTENT["s10_title"], n)
bullets(s, 0.5, 1.2, 6.1, 5.6, CONTENT["s10_left"], size=13)
bullets(s, 6.9, 1.2, 6.0, 5.6, CONTENT["s10_right"], size=13)

OUT.parent.mkdir(exist_ok=True); prs.save(OUT); print("saved", OUT, "with", n, "slides")
