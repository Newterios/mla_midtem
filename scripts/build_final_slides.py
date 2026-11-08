"""Build the Final slide deck (slides/Final_SE-2424_slides.pptx), same look as the Endterm deck.

    ~/Downloads/mla/mla_midtem/.venv/bin/python scripts/build_final_slides.py [--out PATH]

Texts live in scripts/final_slides_content.json (edit there, not in the layout code); their `{{key}}`
placeholders are filled from outputs/ (scripts/final_build_common.py), and a value that does not exist
yet is shown as "pending". Figures come from outputs/<track>/; a missing figure becomes a grey
"figure pending" box naming the file, so the deck can be built at any stage. Layout, colours and
fonts follow the Endterm builder (white 16:9, red top bar, Calibri, grey footer with the slide number).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_build_common import SLIDES, Values, blocks_summary, fill  # noqa: E402

CONTENT_JSON = Path(__file__).resolve().parent / "final_slides_content.json"
DARK, GREY, ACCENT, LIGHT = RGBColor(0x1F, 0x29, 0x37), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xB9, 0x1C, 0x1C), RGBColor(0xF3, 0xF4, 0xF6)
FONT = "Calibri"


class Deck:
    def __init__(self, values: Values, content: dict):
        self.v, self.c = values, content
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0
        self.missing_figures: list[str] = []

    # ------------------------------------------------------------------ primitives
    def t(self, key_or_text) -> list[str] | str:
        val = self.c.get(key_or_text, key_or_text) if isinstance(key_or_text, str) else key_or_text
        if isinstance(val, list):
            return [fill(x, self.v, missing="pending") for x in val]
        return fill(val, self.v, missing="pending")

    def text(self, slide, x, y, w, h, s, size=14, bold=False, color=DARK, align=PP_ALIGN.LEFT):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, line in enumerate(s if isinstance(s, list) else [s]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            run = p.add_run()
            run.text = line
            run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
            run.font.color.rgb = color
            p.alignment = align
            p.space_after = Pt(4)
        return tb

    def bullets(self, slide, x, y, w, h, items, size=13):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            head, _, rest = it.partition("::")
            r = p.add_run()
            r.text = "• " + head.strip()
            r.font.size, r.font.bold, r.font.name = Pt(size), bool(rest), FONT
            r.font.color.rgb = DARK
            if rest:
                r2 = p.add_run()
                r2.text = " " + rest.strip()
                r2.font.size, r2.font.name = Pt(size), FONT
                r2.font.color.rgb = DARK
            p.space_after = Pt(6)
        return tb

    def slide(self, title_key: str | None = None):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        if title_key:
            bar = s.shapes.add_shape(1, 0, 0, self.prs.slide_width, Inches(0.12))
            bar.fill.solid()
            bar.fill.fore_color.rgb = ACCENT
            bar.line.fill.background()
            self.text(s, 0.5, 0.25, 12.3, 0.9, self.t(title_key), size=24, bold=True)
            self.text(s, 0.5, 7.0, 10, 0.4, self.c["footer"], size=10, color=GREY)
            self.text(s, 12.3, 7.0, 0.6, 0.4, str(self.n), size=10, color=GREY, align=PP_ALIGN.RIGHT)
        return s

    def picture(self, slide, candidates: list[str], x, y, w, h):
        """First existing figure among candidates (paths under outputs/), fitted into the w×h box."""
        out = self.v.out
        rel = next((c for c in candidates if (out / c).exists()), None)
        if rel is None:
            self.missing_figures.append(candidates[0])
            box = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
            box.fill.solid()
            box.fill.fore_color.rgb = LIGHT
            box.line.color.rgb = LIGHT
            self.text(slide, x + 0.2, y + h / 2 - 0.3, w - 0.4, 0.6, f"figure pending: outputs/{candidates[0]}", size=12, color=GREY, align=PP_ALIGN.CENTER)
            return None
        from PIL import Image
        with Image.open(out / rel) as im:
            iw, ih = im.size
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        return slide.shapes.add_picture(str(out / rel), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))

    def table(self, slide, df: pd.DataFrame, x, y, w, h, size=11, first_col=None, highlight_col=None):
        rows, cols = df.shape[0] + 1, df.shape[1] + 1
        t = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
        if first_col:
            t.columns[0].width = Inches(first_col)
            for j in range(1, cols):
                t.columns[j].width = Inches((w - first_col) / (cols - 1))
        for r in t.rows:
            r.height = Inches(h / rows)
        for j, c in enumerate([df.index.name or ""] + list(df.columns)):
            cell = t.cell(0, j)
            cell.text = str(c)
            run = cell.text_frame.paragraphs[0].runs[0]
            run.font.size, run.font.bold = Pt(size), True
        for i, (idx, row) in enumerate(df.iterrows(), 1):
            for j, val in enumerate([idx] + list(row)):
                cell = t.cell(i, j)
                cell.text = "" if pd.isna(val) else str(val)
                if cell.text_frame.paragraphs[0].runs:
                    run = cell.text_frame.paragraphs[0].runs[0]
                    run.font.size = Pt(size)
                    if highlight_col and str(row.get(highlight_col, "")).startswith("yes"):
                        run.font.bold, run.font.color.rgb = True, ACCENT
        return t

    # ------------------------------------------------------------------ slides
    def figure_slide(self, title, figs_left: list[list[str]], bullets_key, left_w=7.6, size=12):
        s = self.slide(title)
        n = len(figs_left)
        h = 5.6 / n
        for i, cands in enumerate(figs_left):
            self.picture(s, cands, 0.4, 1.15 + i * h, left_w, h - 0.1)
        self.bullets(s, left_w + 0.7, 1.2, 12.9 - left_w - 0.7, 5.6, self.t(bullets_key), size=size)
        return s

    def build(self) -> Presentation:
        c = self.c
        # 1 title
        s = self.slide()
        self.text(s, 0.8, 1.6, 11.5, 1.6, self.t("title"), size=34, bold=True)
        self.text(s, 0.8, 3.75, 11.5, 1.2, self.t("subtitle"), size=18, color=GREY)
        self.text(s, 0.8, 5.2, 11.5, 1.2, c["title_meta"], size=14, color=GREY)
        # 2 question
        s = self.slide("s2_title")
        self.text(s, 0.5, 1.2, 12.3, 1.0, self.t("s2_question"), size=16, bold=True, color=ACCENT)
        self.bullets(s, 0.5, 2.2, 6.0, 4.5, self.t("s2_left"), size=13)
        self.bullets(s, 6.8, 2.2, 6.0, 4.5, self.t("s2_right"), size=13)
        # 3 leakage
        self.figure_slide("s3_title", [["leakage/cv_schemes.png"]], "s3_bullets")
        # 4 location
        self.figure_slide("s4_title", [["geo/residual_map.png"], ["geo/permutation_importance.png"]], "s4_bullets")
        # 5 photos
        self.figure_slide("s5_title", [["photos/confusion_best_probe.png", "photos/confusion_finetuned_effnet.png"],
                                       ["photos/gradcam_examples.png", "photos/training_curves.png"]], "s5_bullets")
        # 6 text
        self.figure_slide("s6_title", [["text/cv_text.png", "text/top_terms.png"], ["text/flags_residuals.png", "text/top_terms.png"]], "s6_bullets")
        # 7 fusion: block table + figure
        s = self.slide("s7_title")
        tab = blocks_summary(self.v)
        tab = tab[["best variant", "CV MAE (M KZT)", "Δ vs base (M)", "folds better", "variants kept"]]
        tab["best variant"] = tab["best variant"].map(lambda x: x if len(x) <= 42 else x[:40] + "…")
        tab.index.name = "block"
        self.table(s, tab, 0.4, 1.2, 8.6, 0.42 * (len(tab) + 1), size=10, first_col=2.0)
        self.picture(s, ["final/fig_model_comparison.png"], 0.4, 3.6, 8.6, 3.2)
        self.bullets(s, 9.3, 1.2, 3.8, 5.6, self.t("s7_bullets"), size=12)
        self.text(s, 0.4, 6.72, 8.6, 0.3, fill("CV = 5-fold CV on the 3,852 training ads (mean ± fold SD). Rule: kept only if CV MAE < base − 0.354 M ({{final/selection:rule_threshold_M|.3f}} M, repaired folds).",
                                               self.v, missing="pending"), size=9, color=GREY)
        # 8 intervals + flag
        self.figure_slide("s8_title", [["trust/coverage_chosen.png", "trust/interval_methods_width.png"], ["trust/flag_threshold_by_band.png"]], "s8_bullets")
        # 9 SHAP + comparables
        self.figure_slide("s9_title", [["trust/shap_beeswarm.png", "trust/shap_waterfall_example.png"], ["trust/comparables_distance.png"]], "s9_bullets")
        # 10 test + November
        s = self.slide("s10_title")
        res = self.v.csv("final_eval/results_table")
        if res is not None and not res.empty:
            res = res.set_index(res.columns[0]).head(6)
            res.index.name = "model"
            self.table(s, res.iloc[:, :6], 0.4, 1.2, 8.5, 0.36 * (len(res) + 1), size=10, first_col=2.5)
            self.picture(s, ["final_eval/nov_holdout.png"], 0.4, 4.2, 8.5, 2.5)
        else:
            self.picture(s, ["final_eval/results_table.csv"], 0.4, 1.2, 8.5, 2.6)
            self.picture(s, ["final_eval/nov_holdout.png"], 0.4, 4.0, 8.5, 2.7)
        self.bullets(s, 9.1, 1.2, 4.0, 5.6, self.t("s10_bullets"), size=12)
        # 11 errors
        self.figure_slide("s11_title", [["final_eval/residuals.png"]], "s11_bullets", left_w=8.0)
        # 12 app
        self.figure_slide("s13_title", [["final/app_example.png"]], "s13_bullets", left_w=8.4)
        # 13 RL + limitations
        s = self.slide("s14_title")
        self.bullets(s, 0.5, 1.2, 6.1, 5.6, self.t("s14_left"), size=13)
        self.bullets(s, 6.9, 1.2, 6.0, 5.6, self.t("s14_right"), size=13)
        # 14 contributions + AI
        s = self.slide("s15_title")
        self.bullets(s, 0.5, 1.2, 6.1, 5.6, self.t("s15_left"), size=13)
        self.bullets(s, 6.9, 1.2, 6.0, 5.6, self.t("s15_right"), size=13)
        # 15 conclusions
        s = self.slide("s12_title")
        self.bullets(s, 0.5, 1.2, 6.1, 5.6, self.t("s12_left"), size=13)
        self.bullets(s, 6.9, 1.2, 6.0, 5.6, self.t("s12_right"), size=13)
        return self.prs


def build(values: Values | None = None, content: dict | None = None) -> Deck:
    deck = Deck(values or Values(), content or json.loads(CONTENT_JSON.read_text(encoding="utf-8")))
    deck.build()
    return deck


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(SLIDES))
    args = ap.parse_args(argv)
    deck = build()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    deck.prs.save(args.out)
    print(f"{args.out}: {deck.n} slides, {len(deck.missing_figures)} figures pending")
    if deck.missing_figures:
        print("  pending figures:", ", ".join(deck.missing_figures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
