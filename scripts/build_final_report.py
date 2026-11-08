"""Collect the Final tables into FINAL_REPORT.md (the Final counterpart of ENDTERM_REPORT.md).

    ~/Downloads/mla/mla_midtem/.venv/bin/python scripts/build_final_report.py [--out PATH] [--strict]

Reads only `outputs/*/` (see scripts/final_build_common.py for the files and the placeholder syntax).
A table whose output is not there yet is replaced by a one-line "Pending" note, so the report can be
built at any stage. `--strict` exits 1 when anything is still pending.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_build_common import (PRODUCERS, REPORT, Values, blocks_summary, cv_view, fill,  # noqa: E402
                                placeholders, track_status)


def _md(df: pd.DataFrame | None, **kw) -> str | None:
    return None if df is None or df.empty else df.to_markdown(**kw)


class Report:
    def __init__(self, values: Values):
        self.v = values
        self.parts: list[str] = []
        self.pending: list[str] = []

    def text(self, s: str) -> None:
        self.parts.append(fill(s.strip("\n"), self.v, missing="keep"))

    def table(self, title: str, rel: str, body: str | None, note: str = "") -> None:
        self.parts.append(f"## {title}" if title else "")
        if body is None:
            self.pending.append(rel)
            who = PRODUCERS.get(rel.split("/")[0], "the track scripts")
            self.parts.append(f"_Pending: `outputs/{rel}` not produced yet ({who})._")
        else:
            if note:
                self.parts.append(fill(note.strip(), self.v, missing="keep"))
            self.parts.append(body)

    def render(self) -> str:
        return "\n\n".join(p for p in self.parts if p) + "\n"


def build(values: Values | None = None) -> tuple[str, list[str], list[str]]:
    v = values or Values()
    r = Report(v)
    r.text("""
# Final report — apartment asking-price prediction (SE-2424, Balgabekov & Kaiyrbekuly)

Full text, code and figures: `Final_SE-2424_Balgabekov_Kaiyrbekuly.ipynb`; slides: `slides/Final_SE-2424_slides.pptx`. This file collects the tables. All CV numbers: 5-fold CV on the 3,852 training ads, mean ± fold SD. Track tables (Sections 1–3) use the Endterm folds (KFold 5, shuffle, seed 42); the model choice (Section 4) uses the **repaired folds** of the leakage audit (the same folds with every duplicate group moved whole into one fold, `outputs/leakage/folds_group.csv`) as the primary scheme and reports the random folds too. **Selection rule:** kept only if the CV MAE is below the base by more than the fixed Endterm margin of 0.354 M KZT (its fold SD): below **{{final/selection:rule_threshold_M|.3f}} M** on the repaired folds (base {{final/selection:base_cv_mae_M|.3f}} ± {{final/selection:base_cv_sd_M|.3f}}), below {{base.threshold_M|.3f}} M on the random folds. (An earlier draft used 6.459 = 6.723 − 0.264, the fold SD of the repaired folds; the leakage review corrected it, because the fold SD moves 0.26–0.50 M with the fold seed.)

**Base (Endterm, reproduced):** CV MAE **{{base.cv_mae_M|.3f}} ± {{base.cv_mae_sd_M|.3f}} M**, MdAPE {{base.cv_mdape|.2f}} %; test MAE {{endterm.test_mae_M|.2f}} M, MdAPE {{endterm.test_mdape|.2f}} %, R² {{endterm.test_r2|.3f}}.
**Final model:** {{final.chosen}}: CV MAE **{{final/selection:whole_model_comparison.cv_mae_M|.3f}} ± {{final/selection:whole_model_comparison.cv_mae_sd_M|.3f}} M** on the repaired folds ({{final.cv_mae_M|.3f}} ± {{final.cv_mae_sd_M|.3f}} on the random folds); test MAE **{{final.test_mae_M|.2f}} M** (95 % bootstrap {{final_eval/test_metrics:chosen.bootstrap95_mae.0|.2f}}–{{final_eval/test_metrics:chosen.bootstrap95_mae.1|.2f}}; paired difference to the Endterm {{final_eval/test_metrics:paired_diff_mae_M.0|+.2f}} to {{final_eval/test_metrics:paired_diff_mae_M.1|+.2f}} M), MdAPE **{{final.test_mdape|.2f}} %**, R² {{final.test_r2|.3f}}. November hold-out ({{nov.n}} ads, listing level, reduced model on the card fields): MdAPE {{final_eval/nov_holdout:chosen_on_oct_test.mdape|.2f}} % on the October test → {{nov.mdape|.2f}} % in November.
""")
    r.table("Status of the track outputs", "", track_status(v).to_markdown())

    # base reproduction
    b, t = v.json("core/base_cv"), v.json("core/endterm_repro_test")
    body = None
    if b and t:
        ref = b["endterm_reference"]
        body = pd.DataFrame({"Endterm (submitted)": [ref["cv_mae_M"], ref["cv_mae_sd_M"], ref["cv_mdape"], ref["cv_r2"], ref["test_mae_M"], ref["test_mdape"], ref["test_r2"]],
                             "krisha_ml reproduction": [b["cv_mae_M"], b["cv_mae_sd_M"], b["cv_mdape"], b["cv_r2"], t["test"]["mae_M"], t["test"]["mdape"], t["test"]["r2"]]},
                            index=["CV MAE (M)", "CV fold SD (M)", "CV MdAPE %", "CV R²", "test MAE (M)", "test MdAPE %", "test R²"]).round(4).to_markdown()
    r.table("1. The base: Endterm model reproduced", "core/base_cv.json", body)

    # leakage
    leak = v.json("leakage/leakage_summary")
    body = None
    if leak:
        rows = {k: leak[k] for k in ("all_links", "same_flat") if k in leak}
        cols = ["groups", "ads_in_groups", "groups_crossing_split", "test_ads_with_train_twin", "test_ads_with_train_twin_pct"]
        body = pd.DataFrame(rows).T[cols].to_markdown() + "\n\nPairs by signal: " + ", ".join(f"{k} {n}" for k, n in leak.get("pairs_by_signal", {}).items())
    r.table("2. Leakage audit: duplicate ads", "leakage/leakage_summary.json", body,
            "{{leakage/leakage_summary:n_pairs}} duplicate pairs (photo hashes, image embeddings, structured fields, text) in {{leakage/leakage_summary:all_links.groups}} groups.")
    sch = v.csv("leakage/cv_schemes_summary")
    r.table("", "leakage/cv_schemes_summary.csv", _md(sch.set_index(sch.columns[0]) if sch is not None else None),
            "**CV schemes** (base model): random folds vs folds that keep duplicate groups / buildings / spatial blocks together.")

    # blocks
    r.table("3. Every block against the base (best variant per block)", "", blocks_summary(v).to_markdown())
    for title, rel in [("3.1 Location (OSM + raw ad-page fields)", "geo/cv"), ("3.2 Photos in the price model", "photos/price_cv"),
                       ("3.3 Text (TF-IDF, transformer embeddings, fine-tuning, red flags)", "text/cv")]:
        r.table(title, f"{rel}.csv", _md(cv_view(v.csv(rel))))
    clf = v.csv("photos/classifier_comparison")
    keep = [c for c in ["model", "macro_f1_cv", "macro_f1_sd", "accuracy_cv", "balanced_acc", "macro_f1_fold0", "accuracy_fold0"] if clf is not None and c in clf]
    r.table("3.4 Photos: condition classifier (fresh / good / needs repair / shell)", "photos/classifier_comparison.csv",
            _md(clf[keep].set_index("model").round(3) if clf is not None else None),
            "Frozen CNN + logistic regression, CLIP linear probe (5-fold grouped CV) and fine-tuned EfficientNet-B0 (fold 0 only).")
    fl = v.csv("text/flags_stats")
    r.table("3.5 Text red flags vs base residuals", "text/flags_stats.csv", _md(fl, index=False))

    # fusion
    cmp = v.csv("final/cv_comparison")
    body = None
    if cmp is not None:
        k = cmp[cmp["kind"] != "single"]
        g, rnd = k[k.folds == "group"].set_index("variant"), k[k.folds == "random"].set_index("variant")
        body = pd.DataFrame({"CV MAE repaired": g.cv_mae_M, "SD": g.cv_mae_sd_M, "Δ vs base": g.delta_vs_base_M,
                             "folds better": g.folds_better, "passes (repaired)": g.passes_1sd_rule,
                             "CV MAE random": rnd.cv_mae_M, "passes (random)": rnd.passes_1sd_rule}).round(3).to_markdown()
    r.table("4. Fusion and model selection", "final/cv_comparison.csv", body, """
Strategies: **A** residual stacking (base HGB + cross-fitted log-residual corrections per block), **B** blocks as extra HGB columns, **C** a small multimodal PyTorch network (tabular + text-embedding + photo-embedding branches), alone and **50/50 log-blended with the base** (weight fixed in advance). Whole models on both fold schemes:""")
    if cmp is not None:
        sg = cmp[cmp["kind"] == "single"].pivot_table(index="blocks", columns=["strategy", "folds"], values="cv_mae_M").round(3)
        r.table("4.1 Each block alone (CV MAE, M KZT)", "final/cv_comparison.csv", sg.to_markdown(),
                "No single block passes the rule in any strategy or fold scheme.")
    fs = v.csv("final/forward_selection")
    r.table("4.2 Strict per-step forward selection", "final/forward_selection.csv",
            _md(fs[["strategy", "step", "added", "cv_mae_M", "gain_M", "step_sd_M", "passes_step_rule", "set_passes_vs_base"]].set_index(["strategy", "step"]) if fs is not None else None),
            "Repaired folds, final blocks. Every step must lower the CV MAE by more than the fold SD of the current set, never less than 0.354 M. The first and largest step is already below the margin, so **the strict per-step reading keeps no block** (the Endterm base).")
    ab = v.csv("final/cv_network_ablation")
    r.table("4.3 Ablation of the network blend", "final/cv_network_ablation.csv",
            _md(ab[ab.variant_kind == "blend"][["variant", "cv_mae_M", "cv_mae_sd_M", "delta_vs_base_M", "folds_better"]].set_index("variant").round(3) if ab is not None else None),
            "Repaired folds. The blend with a tabular-only network is worse than the base, so the gain is not plain ensembling; it comes from the blocks: location matters most (without it the blend no longer passes), then the text and photo embeddings; dropping the photo-track tabular features or the raw fields changes the CV by less than fold noise.")
    r.text("""
**Decision (whole-model comparison).** The network + Endterm base 50/50 log-blend wins by **{{final/selection:whole_model_comparison.gain_vs_base_M|.3f}} M** > 0.354 M on the repaired folds (better on 5/5 folds; on the random folds {{final/selection:choice_other_folds.random.cv_mae_M|.3f}} M, {{final/selection:choice_other_folds.random.delta_vs_base_M|+.3f}}), its blocks beat the tabular-only control, and it is the lowest CV of all candidates on both schemes, so it is the **final model**. We report honestly that no single block is certified step by step; the final model is certified as a whole. Only training-ad CV was used; the test ads were scored once, afterwards.
""")

    # trust
    im = v.csv("trust/interval_methods")
    r.table("5. Trust layer: conformal intervals (out-of-fold, training ads)", "trust/interval_methods.csv",
            _md(im.set_index(["level", "variant"]).round(3) if im is not None else None),
            "Chosen: **{{trust/chosen_settings:intervals.chosen_variant}}** (valid per-city coverage at 80 % and 90 %, then the narrowest 90 % width).")
    cov = v.csv("trust/coverage_table")
    body = None
    if cov is not None and im is not None and "chosen" in im:
        ch = im.loc[im["chosen"].astype(str).str.lower() == "true", "variant"]
        if len(ch):
            body = _md(cov[(cov["variant"] == ch.iloc[0]) & (cov["level"] == 0.9)].drop(columns="variant").set_index(["group_type", "group"]).round(3))
    r.table("5.1 Coverage of the chosen 90 % interval by city, price band and fold", "trust/coverage_table.csv", body)
    ov = v.csv("trust/overpricing_shares")
    r.table("5.2 Over-pricing flags: interval rule vs the Endterm 20 % rule", "trust/overpricing_shares.csv",
            _md(ov.set_index(["group_type", "group"]).round(3) if ov is not None else None))
    sh = v.csv("trust/shap_global_importance")
    r.table("5.3 SHAP: mean |contribution| to log price, by source feature", "trust/shap_global_importance.csv",
            _md(sh.set_index(sh.columns[0]).head(15).round(4) if sh is not None else None))
    cw = v.csv("trust/comparables_warning_check")
    r.table("5.4 Comparable ads: does the 'few comparables' warning mark larger errors?", "trust/comparables_warning_check.csv",
            _md(cw.set_index(cw.columns[0]).round(3) if cw is not None else None))

    # evaluation
    res = v.csv("final_eval/results_table")
    r.table("6. One evaluation on the 963 saved test ads", "final_eval/results_table.csv",
            _md(res.set_index(res.columns[0]) if res is not None else None))
    nov = v.json("final_eval/nov_holdout")
    body = None
    if nov:
        body = pd.DataFrame({k: x for k, x in nov.items() if isinstance(x, dict) and k != "by_city"}).T.round(3).to_markdown()
        if isinstance(nov.get("by_city"), dict):
            body += "\n\n" + pd.DataFrame(nov["by_city"]).T.round(3).to_markdown()
    r.table("7. November temporal hold-out (8 Nov 2026, listing level: no coordinates, text or photos)", "final_eval/nov_holdout.json", body)
    err = v.csv("final_eval/error_by_segment")
    r.table("8. Error of the final model by band, city and rooms (test ads)", "final_eval/error_by_segment.csv",
            _md(err.set_index(err.columns[0]) if err is not None else None))

    r.text("""
## 9. The app

`app/gradio_app.py` (UI) on `app/predict.py` (inference API, no Gradio import). For a typed-in listing it builds the final model's inputs from the form: OSM geo features from the coordinates, regex text flags and the USER-base sentence embedding from the description, the CLIP ViT-B/32 embedding of up to three uploaded photos. Raw ad-page fields and the photo track's tabular features cannot come from a form and stay missing, as for training ads without them; the app says so in its notes and also shows the Endterm base estimate. Around the estimate: the trust track's conformal 80 % / 90 % intervals (absolute log-residual, Mondrian by estimate band, centred on the final estimate; calibrated on base OOF residuals, so conservative for the blend), an interval-based over/under-pricing verdict, TreeSHAP reasons (for the HGB half of the blend), five comparable training ads with a "few comparables" warning, text red flags with form-consistency checks, and the photo condition (CLIP probe) with a mismatch warning against the stated condition. Example card: `outputs/final/app_example.png`. Run: `python app/gradio_app.py` (http://127.0.0.1:7860).

## 10. Why not deep reinforcement learning

Reinforcement learning learns a policy from rewards for actions taken in an environment. The data contain no actions and no outcomes: asking prices only, no sale price, no time to sale, one snapshot per ad (days on market are not shown). A pricing agent would need exactly that reward (sold or not, time to sale, final price) and many interactions with buyers; any reward written from our data would be the asking price itself, and RL would reduce to the regression we already have, with more variance. The right tool for "is this price fair, and how sure are we" is supervised regression with calibrated uncertainty. RL becomes relevant with longitudinal data (price cuts and removal dates of the same ad).

## 11. Limitations

- **Asking price ≠ sale price.** The model learns the asking market; "over-priced" means "asked above comparable asking prices".
- **Reused test set.** The 963 saved test ads were scored at the Endterm and again here (once); no Final choice looked at them, but they are no longer a perfectly fresh sample.
- **November hold-out at listing level only.** The November ad pages (coordinates, description, photos) were blocked by the site's bot protection, so only the search cards (17 of 24 fields) were scraped; the fused model could not be tested on them, only a reduced model of the same family.
- **Whole-model certification.** The final model passes the rule as a whole; no block passes it step by step.
- **The app sees less than the CV** (no raw fields, no photo-track tabular features from a form); intervals are calibrated on the base model.
- **One October snapshot**, seller-reported condition labels, OSM completeness varying by district, five folds as a coarse ruler.

## 12. Contributions and AI tools

**Rakhat Balgabekov**: the shared `krisha_ml` foundation and the reproduction of the Endterm, the leakage audit and the repaired folds, the photo track, the November scrape, the deliverable builders. **Aitbek Kaiyrbekuly**: the location track, the text track, the trust layer, the error analysis and the slides. Both: the selection rule, the readings and the conclusions, discussed and written together.

**AI tools.** We used an AI coding assistant (Claude, via Claude Code) more heavily than in the Endterm, and state it plainly: several assistant sessions ran in parallel, one per track, on a shared code contract (`CONTRACT.md`) with one CV harness and one fixed selection rule. The assistant wrote most of the code in `krisha_ml/`, `scripts/` and `app/`, ran the experiments on our machine, drafted the readings, this report and the slides from the outputs, and reviewed the tracks against each other (e.g. the leakage review that corrected the rule threshold). We chose the question and the data, set the rule and the one-time use of the test ads, decided what to keep, checked the numbers against the outputs, and are responsible for every claim.
""")
    out = r.render()
    return out, sorted(set(placeholders(out))), [p for p in r.pending if p]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPORT))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    text, unfilled, pending = build()
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"{args.out}: {len(text.splitlines())} lines, {len(pending)} pending tables, {len(unfilled)} unfilled placeholders")
    if pending:
        print("  pending:", ", ".join(pending))
    return 1 if (args.strict and (pending or unfilled)) else 0


if __name__ == "__main__":
    sys.exit(main())
