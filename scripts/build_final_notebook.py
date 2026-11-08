"""Build the Final notebook: Final_SE-2424_Balgabekov_Kaiyrbekuly.ipynb (not executed).

    ~/Downloads/mla/mla_midtem/.venv/bin/python scripts/build_final_notebook.py [--out PATH] [--strict]

The notebook reads the track outputs in `outputs/` (and calls cheap `krisha_ml` functions); it never
retrains the expensive models, whose scripts are named in each section. Markdown readings contain
`{{key}}` placeholders (see scripts/final_build_common.py); the builder fills every placeholder whose
output already exists and leaves the rest visibly as `{{...}}`. Sections whose outputs are missing get a
"Pending" note. Re-run this builder after the tracks finish, then execute the notebook once:

    jupyter nbconvert --to notebook --execute --inplace Final_SE-2424_Balgabekov_Kaiyrbekuly.ipynb

`--strict` exits with status 1 when any placeholder is still unfilled (use it before the hand-in).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_build_common import NOTEBOOK, Values, fill, pending_note, placeholders  # noqa: E402

# --------------------------------------------------------------------------- notebook helper code
# Kept as a constant so that tests can exec() it against fixture outputs.
NB_SETUP = r'''import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

ROOT = Path.cwd() if (Path.cwd() / "krisha_ml").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from krisha_ml import core, raw  # noqa: E402  (the shared foundation: data, folds, model, metrics)

OUT = core.OUTPUT_DIR
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 3)


def _pending(rel):
    display(Markdown(f"> *Pending:* `outputs/{rel}` is not there yet (see the script named above)."))


def read_csv(rel, **kw):
    """Read outputs/<rel>; show a pending note and return None when the file is not there yet."""
    p = OUT / rel
    if not p.exists():
        _pending(rel)
        return None
    return pd.read_csv(p, **kw)


def read_json(rel):
    p = OUT / rel
    if not p.exists():
        _pending(rel)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def show_figure(rel, width=900):
    p = OUT / rel
    if not p.exists():
        _pending(rel)
        return False
    display(Image(filename=str(p), width=width))
    return True


def cv_view(t):
    """Every track's CV table in one format: mean ± fold SD, MdAPE, delta to the base, folds better, 1-SD rule."""
    if t is None or "cv_mae_M" not in t:
        return None
    v = pd.DataFrame({"variant": t.iloc[:, 0]})
    v["CV MAE (M KZT)"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(t["cv_mae_M"], t.get("cv_mae_sd_M", 0 * t["cv_mae_M"]))]
    for src, dst, r in [("cv_mdape", "CV MdAPE %", 2), ("cv_r2", "CV R²", 3), ("delta_vs_base_M", "Δ vs base (M)", 3)]:
        if src in t:
            v[dst] = t[src].round(r)
    if "folds_better" in t:
        v["folds better"] = t["folds_better"].map(lambda x: f"{int(x)}/5" if pd.notna(x) else "")
    if "passes_1sd_rule" in t:
        v["kept (1-SD rule)"] = t["passes_1sd_rule"].astype(str).str.lower().map({"true": "yes", "false": "no"}).fillna("")
    return v.set_index("variant").sort_values("CV MAE (M KZT)")


def show_cv(rel):
    v = cv_view(read_csv(rel))
    if v is not None:
        display(v)
    return v


BASE = read_json("core/base_cv.json")
THRESHOLD = BASE["cv_mae_M"] - BASE["cv_mae_sd_M"] if BASE else None
'''

# --------------------------------------------------------------------------- content
# Each item: ("md", text) | ("code", text) | ("needs", [output paths]) -> pending note when missing.
CELLS: list[tuple[str, object]] = []


def md(text: str) -> None:
    CELLS.append(("md", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


def needs(*paths: str) -> None:
    CELLS.append(("needs", list(paths)))


# ----------------------------------------------------------------- title + section 1
md("""
# Which apartments are over-priced, and how sure are we? Location, photos, text and a trust layer on krisha.kz ads

**Course project, Final (Week 10 defence)** · Machine Learning Algorithms · Group **SE-2424**
**Team:** Rakhat Balgabekov, Aitbek Kaiyrbekuly
**Direction:** 1. Apartments for sale · **Task:** regression · **Target:** asking price (KZT) · **Data:** 4,815 krisha.kz ads (Astana, Almaty, October 2026) + 1,507 November ads as a temporal hold-out

This notebook continues the Endterm notebook on **the same cleaned data, the same 3,852 training ads, the same five folds and the same 963 saved test ads**. The Endterm model (tuned histogram gradient boosting on 24 structured fields) is the **base**; every Final block (location, photos, text) is added to it and kept only if it beats the base by more than the Endterm's one fold SD (0.354 M KZT) in 5-fold CV on duplicate-aware folds. The test ads are used once, in Section 8. Heavy computations (OSM downloads, image and text encoders, fine-tuning) run in the scripts named in each section and write to `outputs/`; the notebook reads those files and recomputes the cheap parts with the shared package `krisha_ml`. Every table and figure is followed by our reading of it.
""")
code(NB_SETUP)
md("""
## 1. The question, and what changed since the Endterm

The question is the Endterm's, now with the second half the Endterm could not answer:

> **Which apartments in Astana and Almaty are asked above or below what comparable apartments go for, by how much, and how sure can we be?**

A **buyer** wants to know whether an ad is worth a viewing; a **seller** wants a defensible starting price; a **realtor** wants a reference for every new listing. The Endterm gave them a point estimate (test MAE {{endterm.test_mae_M}} M KZT, MdAPE {{endterm.test_mdape|.2f}} %) and a crude flag ("more than 20 % above the estimate"). Its own error analysis said where the remaining error lives: in what the 24 structured fields **do not contain**: the exact location beyond the district, the state of the flat that only the photos show, and the selling points written in the description. The Final therefore (1) checks first that the CV is honest (duplicate ads, Section 2), (2) adds three information sources the Endterm left unused (Sections 3–5), (3) keeps only what passes one fixed rule (Section 6), and (4) replaces the 20 % flag with calibrated price intervals, explanations and comparable ads (Section 7).

| | Endterm | Final |
|---|---|---|
| Data | 4,815 ads, 24 structured fields | + coordinates → OSM objects, + up to 5 photos per ad, + full description text, + 1,507 November ads |
| CV | KFold(5, shuffle, 42) | the same folds with every **duplicate group moved whole into one fold** (26 ads moved; primary scheme), random folds reported next to them |
| Model | tuned HGB on log price | the same HGB, 50/50 log-blended with a small **multimodal network** on location, raw fields, photo and text features (chosen by CV, Section 6) |
| Selection rule | lowest CV MAE | CV MAE better than the base by **> 0.354 M** (the Endterm fold SD, fixed): below **{{final/selection:rule_threshold_M|.3f}} M** on the repaired folds ({{base.threshold_M|.3f}} M on the random folds) |
| Output for the user | point estimate, > 20 % flag | estimate + **conformal 80/90 % interval**, SHAP reasons, 5 comparable ads, interval-based flag |
| Evaluation | 963 test ads once | the same 963 test ads once + **November temporal hold-out** |
""")
code("""
# The base: the submitted Endterm model, reproduced by krisha_ml (scripts/verify_endterm.py)
repro = read_json("core/endterm_repro_test.json")
display(pd.DataFrame({
    "Endterm (submitted)": {k: core.ENDTERM_REFERENCE[k] for k in ["cv_mae_M", "cv_mae_sd_M", "cv_mdape", "cv_r2", "test_mae_M", "test_mdape", "test_r2"]},
    "krisha_ml reproduction": {**{k: BASE[k] for k in ["cv_mae_M", "cv_mae_sd_M", "cv_mdape", "cv_r2"]},
                               "test_mae_M": repro["test"]["mae_M"], "test_mdape": repro["test"]["mdape"], "test_r2": repro["test"]["r2"]},
}).round(4))
print(f"random (Endterm) folds: kept only if CV MAE < {THRESHOLD:.3f} M KZT (base {BASE['cv_mae_M']:.3f} - margin {BASE['cv_mae_sd_M']:.4f})")
SEL = read_json("final/selection.json")
if SEL:
    print(f"repaired (duplicate-aware) folds, the primary scheme: kept only if CV MAE < {SEL['rule_threshold_M']:.3f} M KZT "
          f"(base {SEL['base_cv_mae_M']:.3f} ± {SEL['base_cv_sd_M']:.3f} - fixed margin {SEL['rule_margin_M']:.4f})")
""")
md("""
**Reading.** The shared package reproduces the submitted Endterm to four decimals: CV MAE **{{base.cv_mae_M|.3f}} ± {{base.cv_mae_sd_M|.3f}} M KZT**, test MAE {{endterm.test_mae_M|.2f}} M (the test number is shown only to prove the reproduction; it played no part in any choice below). Every number in Sections 2–7 is computed on the 3,852 training ads. The yardstick is strict on purpose: a block has to lower the CV MAE by more than **0.354 M**, the Endterm's fold SD. After the leakage audit (Section 2.2) the primary folds are the Endterm folds with every duplicate group in one fold; there the base scores {{final/selection:base_cv_mae_M|.3f}} M and the threshold is **{{final/selection:rule_threshold_M|.3f}} M**. The margin is kept at 0.354 M rather than the {{final/selection:base_cv_sd_M|.3f}} M fold SD of the repaired folds, because the fold SD itself moves between 0.26 and 0.50 M with the fold seed; a smaller SD from one split would loosen the rule by chance (an earlier draft used 6.459 = 6.723 − 0.264, and the leakage review corrected it). With five folds, smaller gains cannot be told apart from the luck of the split.
""")

# ----------------------------------------------------------------- section 2
md("""
## 2. Data and leakage audit

### 2.1 The data and the preprocessing pipeline
The pipeline is the Endterm's, packed into `krisha_ml.core` so that every track uses it unchanged: the cleaned table (`data/clean/krisha_apartments_clean.csv`, cleaning steps in the Midterm notebook), coordinates outside the city box set to missing (10 ads), 24 features (11 numeric, 10 categorical, 3 boolean), median imputation with missing-value indicators for numbers, one-hot encoding with a minimum frequency of 10 for categories, and the target on the log scale (`TransformedTargetRegressor`), so that the model minimises relative rather than absolute errors and predicts back in KZT. Anything fitted on data in the Final (TF-IDF vocabularies, PCA, probes, conformal quantiles) is fitted **inside each training fold** through `core.cv_evaluate(feature_fn=...)` or a residual corrector, never on all ads.
""")
code("""
train = core.load_train()
print(f"training ads: {len(train.ids):,}; features: {len(core.FEATURES)}; test ads (kept aside): {len(core.load_test_ids()):,}")
display(train.df.groupby("city").agg(ads=("price", "size"), median_price_M=("price", lambda s: s.median() / 1e6),
                                      median_area=("area", "median"), with_coordinates=("lat", lambda s: s.notna().mean())).round(2))
display(pd.Series({"numeric": core.NUM, "categorical": core.CAT, "boolean": core.BOOL}).map(", ".join).to_frame("features"))
""")
md("""
**Reading.** The two cities are almost balanced in the training rows, and nearly every ad has coordinates, which is what makes the location block of Section 3 possible. The feature list is exactly the Endterm's 24 columns; every new block below is an *addition* to it, so a change in the CV MAE can be attributed to that block alone.

### 2.2 Duplicate ads: is the Endterm CV honest?
A krisha.kz flat is often listed twice: reposted by the owner, or listed by the owner and an agency, with the same photos and nearly the same price. If one copy sits in a training fold and the other in the validation fold, the model can "remember" the price, and the CV MAE is optimistic. We looked for duplicates with four independent signals (`scripts/leakage_audit.py`, `krisha_ml.dedup`): perceptual hashes of the photos (pHash/dHash), cosine similarity of image embeddings, structured fields (area within 1 %, price within 5 %, within 150 m), and TF-IDF similarity of the descriptions. Photos shared by more than five ads (stock pictures of a residential complex) are ignored, because they connect different flats.
""")
needs("leakage/leakage_summary.json", "leakage/cv_schemes_summary.csv", "leakage/cv_schemes.png")
code("""
leak = read_json("leakage/leakage_summary.json")
if leak:
    rows = {k: leak[k] for k in ("all_links", "same_flat") if k in leak}
    display(pd.DataFrame(rows).T[["groups", "ads_in_groups", "groups_crossing_split", "test_ads_with_train_twin", "test_ads_with_train_twin_pct"]])
    display(pd.Series(leak["pairs_by_signal"], name="pairs").sort_values(ascending=False).to_frame())
""")
md("""
**Reading.** The four signals find **{{leakage/leakage_summary:n_pairs}} duplicate pairs**, which merge into **{{leakage/leakage_summary:all_links.groups}} groups** covering {{leakage/leakage_summary:all_links.ads_in_groups}} ads ({{leakage/leakage_summary:same_flat.groups}} groups are certainly the same flat). Most pairs are found by the photos (pHash and embeddings agree), a few only by the structured fields or the text, which is why one signal alone would miss some. Only **{{leakage/leakage_summary:all_links.test_ads_with_train_twin}} test ads ({{leakage/leakage_summary:all_links.test_ads_with_train_twin_pct|.2f}} %)** have a twin in the training rows: the saved test split is almost clean, and its number in Section 8 cannot be inflated by more than this small share. The cleaning step of the Midterm (same address, rooms, floor and area) caught the obvious reposts; what remains are reposts with a changed street spelling or price.
""")
code("""
schemes = read_csv("leakage/cv_schemes_summary.csv", index_col=0)
if schemes is not None:
    display(schemes)
show_figure("leakage/cv_schemes.png")
""")
md("""
**Reading.** The base model scored with folds that keep every duplicate group together (GroupKFold) gives **{{leakage/leakage_summary:recommended_folds.base_cv_mae_M|.3f}} ± {{leakage/leakage_summary:recommended_folds.base_cv_sd_M|.3f}} M**, against {{base.cv_mae_M|.3f}} M for the Endterm's random folds. The difference is far smaller than one fold SD: duplicates are too few to change the Endterm's conclusions. Within the duplicate groups the effect is visible but small (ads whose twin sits in another fold: MAE {{leakage/leakage_summary:twin_analysis.twin_in_other_fold.random_mae_M|.2f}} M with random folds, {{leakage/leakage_summary:twin_analysis.twin_in_other_fold.repaired_mae_M|.2f}} M with repaired folds), because boosting on 24 fields does not memorise single ads the way a nearest-neighbour model would; simply copying the twin's price would have given a median error of {{leakage/leakage_summary:twin_analysis.twin_price_as_prediction.mdape|.1f}} %. Over 20 fold seeds, keeping duplicates together costs only +0.015 ± 0.058 M (a placebo that moves 26 random ads: −0.003), so duplicate leakage is small; the Endterm's seed-42 folds are, however, the best of 20 seeds (6.616 against a 6.720 average), i.e. its CV is ~0.1 M optimistic because the hyper-parameters were tuned on them. The track tables of Sections 3–5 are on the Endterm folds (threshold {{base.threshold_M|.3f}} M); **the model choice of Section 6 uses the repaired folds** (`outputs/leakage/folds_group.csv`) as the primary scheme and reports the random folds next to them: a block that helps only by matching duplicates would gain more on the random folds than on the repaired ones.
""")

# ----------------------------------------------------------------- section 3
md("""
## 3. Location features

**Concept.** Boosting already splits on latitude and longitude, but a tree sees coordinates only as two numbers and has to approximate "close to a park" or "far from the centre" with many axis-parallel cuts; with 3,852 ads it can only afford a few. We therefore compute what a buyer means by location, as features: the great-circle (haversine) distance to the city centre, the distance to the nearest park, school, kindergarten, supermarket, clinic and food place, and the number of such objects within 500 m and 1 km, from OpenStreetMap (Overpass API, downloaded once, cached in `cache/osm/`, `krisha_ml.geo`). Nearest-neighbour queries use a `BallTree` with the haversine metric. These are static, price-free features computed per ad, so they may go into `extra` without fold fitting. We also parse the raw ad-page fields the Endterm left unused (flooring, doors, furniture, internet, security; `features_rawfields.csv`).
""")
needs("geo/cv.csv", "geo/residual_map.png")
code("""
geo_cv = show_cv("geo/cv.csv")
pois = read_csv("geo/osm_poi_summary.csv")
if pois is not None:
    display(pois)
""")
md("""
**Reading.** The best location variant, **{{geo/cv@best:variant}}**, scores **{{geo/cv@best:cv_mae_M|.3f}} ± {{geo/cv@best:cv_mae_sd_M|.3f}} M**, {{geo/cv@best:delta_vs_base_M|+.3f}} M against the base, better on {{geo/cv@best:folds_better}} of 5 folds; {{geo/cv@n_pass}} of {{geo/cv@n}} variants pass the 1-SD rule. The single feature "distance to the centre" ({{geo/cv[distance to centre]:cv_mae_M|.3f}} M) already gives most of the gain of the full OSM block ({{geo/cv[geo (OSM)]:cv_mae_M|.3f}} M): what the coordinates lacked was a *smooth* distance, not more objects. The raw ad-page fields on their own make the model slightly worse ({{geo/cv[base + raw fields]:cv_mae_M|.3f}} M): 16 sparse columns, each filled in only half of the ads, add more noise than signal. The fold SD of the location variants is smaller than the base's, i.e. they help most on the fold where the base was worst, which is a hint of real signal even where the mean gain is under the threshold.
""")
code("""
show_figure("geo/residual_map.png")
show_figure("geo/permutation_importance.png", width=700)
""")
md("""
**Reading.** The residual map of the base model shows where the Endterm was wrong by location: patches of one colour (a prestige street, a district edge) are what a location feature can remove; scattered colours mean the remaining error is about the flat, not about the place. The permutation importance of the location model says which new columns it uses: shuffling the distance to the centre raises the CV MAE by 1.54 M, far more than any other OSM column (parks within 1 km: 0.42 M, food places within 1 km: 0.36 M), and all geo columns together by 4.34 M, about as much as latitude + longitude (4.40 M): the trees can rebuild most of the location from either, which is why the block adds little on top of the coordinates.
""")

# ----------------------------------------------------------------- section 4
md("""
## 4. Photos: CLIP, a fine-tuned CNN, Grad-CAM and a mismatch flag

**Concepts.** A convolutional network (CNN) turns an image into a vector of learned features; ImageNet-pretrained networks (MobileNetV3, EfficientNet-B0) already recognise textures and objects, so with ~3,000 labelled ads we can either **freeze** them and train a logistic regression on top (a *linear probe*), or **fine-tune** the last layers with data augmentation. **CLIP** (contrastive language–image pre-training) maps images and sentences into the same space, which allows **zero-shot** classification by comparing a photo with prompts such as "a kitchen with designer renovation" or "an unfinished apartment with bare concrete walls", without any training. **Grad-CAM** weights the last convolutional feature maps by the gradient of the predicted class to show *where* the network looked.

**The task for the photos.** Price is a weak and noisy label for a picture, so we first teach the photos something they can see: the **condition of the flat** (fresh renovation, good, needs repair, shell/pre-finish), which the seller states in a field. All photo models are evaluated with folds that keep duplicate-photo clusters together (StratifiedGroupKFold), because a reposted ad with the same photos would otherwise appear on both sides. Scripts: `photo_01_embed.py` (embeddings, CLIP zero-shot), `photo_02_classify.py` (probes, fine-tuning), `photo_03_price.py` (photo features in the price model); module `krisha_ml.photos`. Photos stay local (`data/photos/`) and are never committed.
""")
needs("photos/clip_zero_shot_ad.csv")
code("""
zs = read_csv("photos/clip_zero_shot_ad.csv")
if zs is not None:
    print(f"ads with at least one usable photo: {len(zs):,}")
    rooms = [c for c in zs if c.startswith("clip_share_")]
    display(zs[rooms].mean().sort_values(ascending=False).rename("mean share of photos").to_frame().round(3))
    cond = core.load_clean()[["id", "condition"]].merge(zs[["id", "clip_quality_index"]], on="id")
    display(cond.groupby("condition")["clip_quality_index"].describe()[["count", "mean", "50%"]].round(3))
""")
md("""
**Reading.** CLIP zero-shot sorts the photos into rooms without a single label: living rooms (27 % of the photos), corridors (20 %), bedrooms (17 %) and kitchens (16 %) dominate; 9 % are exteriors and about 7 % floor plans, bathrooms or window views. Exteriors and plans say nothing about the condition of the flat, which is why the "interior-weighted" variants below down-weight them. The CLIP quality index (prompt probabilities from "designer renovation" to "bare concrete") is highest for ads stated as fresh renovation (mean 0.92), then good (0.18), and negative for rough finish (−0.41), needs repair (−0.69) and free layout (−0.70): zero-shot CLIP already separates renovated from unfinished flats, although it does not separate "needs repair" from "shell".
""")
needs("photos/classifier_comparison.csv", "photos/confusion_best_probe.png", "photos/training_curves.png")
code("""
clf = read_csv("photos/classifier_comparison.csv")
if clf is not None:
    display(clf.set_index("model")[[c for c in ["C", "macro_f1_cv", "macro_f1_sd", "accuracy_cv", "balanced_acc", "macro_f1_fold0", "accuracy_fold0"] if c in clf]].round(3))
show_figure("photos/confusion_best_probe.png", width=520)
show_figure("photos/training_curves.png", width=760)
""")
md("""
**Reading.** On {{photos/classifier_comparison:n_ads}} labelled training ads, the best model is **{{photos/classifier_comparison:best_probe}}**: macro-F1 {{photos/classifier_comparison[(c) CLIP linear probe]:macro_f1_cv|.3f}} for the CLIP linear probe against {{photos/classifier_comparison[frozen mnv3]:macro_f1_cv|.3f}} for frozen MobileNetV3 and {{photos/classifier_comparison[frozen effnet_b0 + LR]:macro_f1_cv|.3f}} for frozen EfficientNet-B0. The fine-tuned EfficientNet-B0 reaches macro-F1 {{photos/classifier_comparison[fine-tuned]:macro_f1_fold0|.3f}} on fold 0 (compare with the probes' fold-0 column, not with their 5-fold mean). CLIP wins because it was pre-trained on hundreds of millions of image–text pairs that include interior photos and words like "renovation", while ImageNet classes say nothing about wallpaper; fine-tuning a CNN on ~3,000 ads with noisy labels improves the training loss much faster than the validation loss (the curves), i.e. it overfits despite augmentation. The confusion matrix shows where the errors are: neighbouring classes (fresh vs good, needs repair vs shell) are confused, opposite ends almost never, so the classifier is ordinal in practice, and part of the confusion is in the labels themselves (sellers call a flat "fresh renovation" generously).
""")
needs("photos/gradcam_correct.png", "photos/gradcam_incorrect.png")
code("""
# Grad-CAM on the fine-tuned EfficientNet-B0 (fold 0 weights saved by photo_02_classify.py).
# The panels show krisha.kz photos, so they stay local (gitignored, data card: photos are never committed)
# and are not embedded in this notebook; set FINAL_SHOW_PHOTOS=1 to display them on a machine that has them.
import os
for f in ("photos/gradcam_correct.png", "photos/gradcam_incorrect.png"):
    if os.environ.get("FINAL_SHOW_PHOTOS") == "1":
        show_figure(f, width=900)
    else:
        print(f"outputs/{f}: local only (shows krisha.kz photos), "
              f"{'present' if (ROOT / 'outputs' / f).exists() else 'missing'}; rebuild with scripts/photo_05_gradcam.py")
""")
md("""
**Reading.** The two Grad-CAM panels (kept local and not embedded here, because they show the sellers' photos) show where the fine-tuned EfficientNet-B0 looked, for correctly (top) and wrongly (bottom) classified photos. The heat maps are a sanity check, not a proof: they tell whether the network relies on the room surfaces that carry the label (walls, floors, finishing) or on objects that only correlate with it (furniture, a window view). Because the label belongs to the flat while one photo may not show the parts that decide it, the ad-level prediction averages over up to five photos.
""")
needs("photos/mismatch_summary.json", "photos/mismatch_examples.csv")
code("""
# Mismatch flag: the photos contradict the stated condition
mm = read_csv("photos/mismatch_examples.csv")
mms = read_json("photos/mismatch_summary.json")
if mms:
    display(pd.DataFrame(mms["counts"]).T)
if mm is None and zs is not None:   # cheap fallback from CLIP zero-shot only, clearly labelled as such
    d = core.load_clean()[["id", "condition", "price"]].merge(zs, on="id")
    bad = d["clip_mean_needs_repair"] + d["clip_mean_shell"] + d["clip_mean_prefinish"]
    d["mismatch"] = (d["condition"].eq("fresh_renovation") & (bad > 0.5)) | (d["condition"].isin(["needs_repair"]) & (d["clip_mean_designer"] > 0.6))
    print("fallback (CLIP zero-shot, not the trained probe):")
    display(d.groupby("condition")["mismatch"].agg(["sum", "mean"]).round(3))
elif mm is not None:
    display(mm.head(10))
""")
md("""
**Reading.** The mismatch flag marks ads whose photos contradict the stated condition, e.g. "fresh renovation" in the field and bare walls in the pictures: {{photos/mismatch_summary:counts.train.photos_poor_vs_stated_good}} training ads look worse than stated and {{photos/mismatch_summary:counts.train.photos_good_vs_stated_poor}} look better (about 2 % of the training ads; {{photos/mismatch_summary:reliable_counts_train.photos_poor_vs_stated_good}} + {{photos/mismatch_summary:reliable_counts_train.photos_good_vs_stated_poor}} with a confident probe). For a buyer this is a question to ask before a viewing; for the model it is a hint that the stated `condition` field is noisy, which limits what any condition feature can add. It is a flag for a human, not an accusation: dark photos, a single exterior shot or a staged photo of a show flat cause false alarms.
""")
needs("photos/price_cv.csv")
code("""
show_cv("photos/price_cv.csv")
show_figure("photos/price_cv_delta.png", width=760)
""")
md("""
**Reading.** In the price model the photos help less than in the condition task: the best photo variant, **{{photos/price_cv@best:variant}}**, scores {{photos/price_cv@best:cv_mae_M|.3f}} ± {{photos/price_cv@best:cv_mae_sd_M|.3f}} M ({{photos/price_cv@best:delta_vs_base_M|+.3f}} M, {{photos/price_cv@best:folds_better}} of 5 folds better), and {{photos/price_cv@n_pass}} of {{photos/price_cv@n}} variants pass the rule. The reason is that the base model already has the stated `condition`, building year and building type, which carry most of what a photo says about price; a photo adds the *degree* of renovation within a class, which matters for some ads and not for most. Raw embedding components as features do worse than a few interpretable probabilities, because 16 noisy PCA columns give the trees more ways to overfit.
""")

# ----------------------------------------------------------------- section 5
md("""
## 5. Text: TF-IDF against transformers, and red flags

**Concepts.** **TF-IDF** represents a description as a sparse vector of word and character n-gram counts, weighted down for words that occur in every ad; with **Ridge** regression (L2 penalty, chosen by generalised cross-validation inside each training fold) it learns one weight per n-gram. A **transformer** sentence encoder (rubert-tiny2, multilingual USER-base) maps the whole text to a dense vector that captures meaning rather than exact words; we use it frozen (embedding + Ridge) and fine-tuned on the residuals. Because prices in the text would leak the target, `raw.sanitize_text` removes every sentence that mentions money, all digits and all contact data before any model sees the text.

**How the text enters the model.** As a **residual corrector**: the text model predicts the base model's out-of-fold log error, so it can only add what the 24 fields miss (`core.cv_evaluate_residual`; the residuals come from an inner 5-fold CV, so the corrector never sees an error of a model that was trained on the same ad). Script `run_text.py`, module `krisha_ml.text`.
""")
needs("text/cv.csv")
code("""
text_cv = show_cv("text/cv.csv")
show_figure("text/cv_text.png", width=820)
""")
md("""
**Reading.** The best text variant, **{{text/cv@best:variant}}**, scores **{{text/cv@best:cv_mae_M|.3f}} ± {{text/cv@best:cv_mae_sd_M|.3f}} M**, {{text/cv@best:delta_vs_base_M|+.3f}} M against the base and better on **{{text/cv@best:folds_better}} of 5 folds**; {{text/cv@n_pass}} of {{text/cv@n}} text variants pass the 1-SD rule. TF-IDF word + character n-grams ({{text/cv[TF-IDF word+char + Ridge residual]:cv_mae_M|.3f}} M) and the frozen USER-base embedding ({{text/cv[user-base embedding]:cv_mae_M|.3f}} M) are equally good, and both beat feeding the embedding to the trees as 32 PCA features ({{text/cv[PCA(32)]:cv_mae_M|.3f}} M, worse than the base): a linear model on the residual uses all dimensions with a penalty, while the trees spend splits on noise. A gain on every fold is consistent, but its size stays below the rule's threshold of one fold SD ({{base.cv_mae_sd_M|.3f}} M), which is the honest summary of the text block: real, small, and below what five folds can certify. The transformer is not better than TF-IDF here because the descriptions are short, formulaic and in a mix of Russian and Kazakh, where the exact selling words ("designer renovation", "first line", "with furniture") matter more than paraphrase.
""")
code("""
top = read_csv("text/top_terms.csv")
if top is not None:
    display(top.head(25))
show_figure("text/top_terms.png", width=820)
""")
md("""
**Reading.** The n-grams that push the price *below* the base estimate are the words of a compromise or a hurry: «срочная продажа» (urgent sale), «документы» and «под» (as in «под ипотеку», documents and mortgage conditions), «студия», «ключи» (keys handed over: often a new, unfinished flat), «в общежитии» (in a dormitory), «звоните» (typical of agency templates). The ones that push it *up* name the building and the neighbourhood: «класс» (comfort/business class), «park» (complex names), «супермаркеты аптеки» (shops nearby), «полностью» (fully furnished / renovated). These are facts the 24 fields record coarsely or not at all, which is why a linear residual model on the text helps on every fold.
""")
needs("text/flags_stats.csv", "text/flags_residuals.png")
code("""
# Red flags: regex rules with negation handling (krisha_ml.text.extract_flags), checked against the base residuals
try:
    from krisha_ml import text as ktext
    flags = ktext.extract_flags(train.df)
    display(flags.drop(columns="id", errors="ignore").mean().sort_values(ascending=False).rename("share of training ads").to_frame().round(3))
except Exception as exc:   # the text module is optional for this cell
    print("flag extraction skipped:", exc)
stats = read_csv("text/flags_stats.csv")
if stats is not None:
    display(stats)
show_figure("text/flags_residuals.png", width=760)
""")
md("""
**Reading.** Each flag is rare (1–15 % of the ads), and the table shows whether the base model already prices it (residual = log(asking / OOF estimate)): ads saying **cash only** are asked a median 6.5 % *below* the base estimate, **assignment of rights** (unfinished building) 5.4 % below, **former dormitory** 4.8 % below, **rough finish** 3.7 % and **urgent sale** 2.8 % below, all with 95 % intervals below zero; **designer renovation** is asked 2.0 % *above*. The market discounts legal and construction risk that the 24 fields do not record. Encumbrance, "price negotiable" and "no encumbrance" have intervals touching zero. Negation matters ("no encumbrance" is the opposite of "encumbrance"), which is why the rules handle it. The flags are shown to the user next to the estimate; as features of the HGB they add little (−0.05 M), because they are too rare to move a 5-fold MAE.
""")

# ----------------------------------------------------------------- section 6
md("""
## 6. Fusion and model selection

**The rule.** A model is kept only if its 5-fold CV MAE is below the base by more than the fixed Endterm margin of 0.354 M, on the repaired (duplicate-aware) folds: below **{{final/selection:rule_threshold_M|.3f}} M** (base {{final/selection:base_cv_mae_M|.3f}} ± {{final/selection:base_cv_sd_M|.3f}} M); the same candidate is reported on the random Endterm folds. The rule was fixed before the fused numbers were seen, and the test ads play no part. **Three fusion strategies** (`krisha_ml.fusion`, `scripts/run_final.py`): **A** residual stacking (the base HGB plus cross-fitted log-residual corrections, one per block: HGB for static blocks, Ridge for TF-IDF and embeddings, blended with non-negative weights learned on inner out-of-fold corrections); **B** the blocks as extra HGB columns (embeddings PCA-reduced inside the fold, text as an inner-OOF TF-IDF price prior); **C** a small PyTorch network (tabular branch + text-embedding branch + photo-embedding branch, L1 loss on the standardised log price, early stopping, 2 seeds), alone and **50/50 log-blended with the base** (weight fixed in advance, not tuned).

**Two readings of the rule.** (1) *Strict per-step forward selection* for A and B: blocks are added greedily, and each step must lower the CV MAE by more than max(fold SD of the current set, 0.354 M). (2) *Whole-model comparison*: a complete fused model (all blocks, one strategy) is compared with the base by the same rule. Reading (1) asks "is every single block certified?", reading (2) "is the whole model certified?". We report both and choose by (2), with one safeguard taken from (1): the gain of the network blend must come from the blocks, not from blending alone, so the blend of the base with a *tabular-only* network is the control.
""")
code("""
cmp = read_csv("final/cv_comparison.csv")
if cmp is not None:
    keep = cmp[(cmp["kind"] != "single")]
    v = keep.pivot_table(index="variant", columns="folds", values="cv_mae_M", sort=False)
    v["SD (repaired)"] = keep[keep.folds == "group"].set_index("variant")["cv_mae_sd_M"]
    v["Δ vs base (repaired)"] = keep[keep.folds == "group"].set_index("variant")["delta_vs_base_M"]
    v["passes rule (repaired)"] = keep[keep.folds == "group"].set_index("variant")["passes_1sd_rule"]
    v["passes rule (random)"] = keep[keep.folds == "random"].set_index("variant")["passes_1sd_rule"]
    display(v.rename(columns={"group": "CV MAE repaired folds", "random": "CV MAE random folds"}).round(3))
    singles = cmp[cmp["kind"] == "single"].pivot_table(index="blocks", columns=["strategy", "folds"], values="cv_mae_M")
    display(singles.round(3))
show_figure("final/fig_model_comparison.png", width=820)
""")
md("""
**Reading.** On the repaired folds no single block passes the rule in any strategy: the best single blocks lower the CV MAE by 0.15–0.24 M (residual stacking A: photo features {{final/cv_comparison[A: photo]:cv_mae_M|.3f}} M, text TF-IDF 6.513 M), against a margin of 0.354 M. Together they do more: strategy A with all seven blocks reaches {{final/cv_comparison[A: geo+rawfields+photo+text_flags+text+text_emb+photo_emb]:cv_mae_M|.3f}} M, the HGB with all blocks as features (B) {{final/cv_comparison[B: geo+rawfields+photo+text_flags+text+text_emb+photo_emb]:cv_mae_M|.3f}} M (just above the threshold). The network alone is clearly worse than the base ({{final/cv_comparison[C: network, all blocks]:cv_mae_M|.3f}} M; tabular only {{final/cv_comparison[C: network, tabular only]:cv_mae_M|.3f}} M: 3,852 rows are few for a network), but its errors are different from the trees', so the **50/50 log-blend with the base** gives the lowest CV MAE of all: **{{final/selection:whole_model_comparison.cv_mae_M|.3f}} ± {{final/selection:whole_model_comparison.cv_mae_sd_M|.3f}} M**, {{final/selection:whole_model_comparison.gain_vs_base_M|.3f}} M below the base (more than the 0.354 M margin), better on all five folds, and the same on the random folds ({{final/selection:choice_other_folds.random.cv_mae_M|.3f}} M against 6.616). The gain is about the same on both fold schemes, so it does not come from duplicate ads.
""")
md("""
**Strict per-step forward selection and the ablation of the blend** (repaired folds, final blocks, step margin = max(fold SD of the current set, 0.354 M)).
""")
code("""
fs = read_csv("final/forward_selection.csv")
if fs is not None:
    display(fs[["strategy", "step", "added", "cv_mae_M", "gain_M", "step_sd_M", "folds_better_than_prev", "passes_step_rule", "set_passes_vs_base"]].set_index(["strategy", "step"]))
abl = read_csv("final/cv_network_ablation.csv")
if abl is not None:
    display(abl[abl.variant_kind == "blend"][["variant", "cv_mae_M", "cv_mae_sd_M", "delta_vs_base_M", "folds_better"]].set_index("variant").round(3))
show_figure("final/fig_ablation_waterfall.png", width=760)
""")
md("""
**Reading.** Read strictly step by step, **no block is ever certified**: the first and largest step (photo features in A, −0.24 M; location in B, −0.21 M) is already below the 0.354 M margin, and every later step adds less (A: text embedding 0.13, location 0.09, then ≤ 0.01 M; B: text flags 0.10, then 0.01 M). So the strict per-step reading keeps the Endterm base. The ablation answers why the blend still wins: blending the base with a network that sees only the 24 tabular fields gives {{final/cv_network_ablation[Cb: tabular only]:cv_mae_M|.3f}} M, *worse* than the base, so the gain is not plain ensembling; it comes from the blocks, and from several of them at once: location matters most (without it the blend gets {{final/cv_network_ablation[Cb: all minus geo]:cv_mae_M|.3f}} M, +0.29 M, and no longer passes), the text and photo embeddings add 0.09 and 0.06 M, the embeddings alone already give most of the gain ({{final/cv_network_ablation[Cb: embeddings only]:cv_mae_M|.3f}} M), and dropping the photo-track tabular features or the raw fields would even lower the CV slightly ({{final/cv_network_ablation[Cb: all minus photo]:cv_mae_M|.3f}} and {{final/cv_network_ablation[Cb: all minus rawfields]:cv_mae_M|.3f}} M, within fold noise). We keep the block set fixed before the ablation rather than re-pick it from these numbers.

**The decision.** The final model is the **network + Endterm base 50/50 log-blend** with the blocks geo, raw fields, photo features, text flags, text embedding and photo embedding. It wins the whole-model comparison by {{final/selection:whole_model_comparison.gain_vs_base_M|.3f}} M > 0.354 M on the primary folds, its blocks beat the tabular-only control, and it is the lowest CV of every candidate on both fold schemes. We do not claim that any *single* block is certified: the per-step reading keeps none, and we say so. The choice used only training-ad CV; the test ads were scored once, afterwards (Section 8).
""")

# ----------------------------------------------------------------- section 7
md("""
## 7. The trust layer: intervals, explanations, comparables and the over-pricing flag

A point estimate with an MAE of 6 M KZT cannot tell a buyer whether *this* ad is over-priced: the error is 2 M for a small flat and 30 M for a penthouse. We add four things (script `run_trust.py`, module `krisha_ml.trust`).

**Conformal prediction.** Split/cross conformal prediction turns any model into intervals with a guaranteed coverage: compute a non-conformity score on calibration ads that the model did not see (here, out-of-fold), take its (1 − α) quantile with the finite-sample correction ⌈(n+1)(1−α)⌉/n, and widen every prediction by it. The guarantee is marginal (on average over ads), so we check coverage per city and per price band. We compare (a) intervals of constant width on the log-residual of the base model and (b) **conformalized quantile regression (CQR)**: two boosting models predict the 5 % and 95 % (10 %, 90 %) quantiles, and conformal calibration corrects their coverage, giving intervals that are wide where the market is uncertain and narrow where it is not; calibration can be per city (Mondrian).
""")
needs("trust/interval_methods.csv", "trust/coverage_table.csv", "trust/chosen_settings.json")
code("""
im = read_csv("trust/interval_methods.csv")
if im is not None:
    display(im.set_index(["level", "variant"]).round(3))
show_figure("trust/interval_methods_width.png", width=760)
""")
md("""
**Reading.** All methods reach their nominal coverage on average (90.6–91.5 % at the 90 % level), as conformal theory promises; they differ in *where* they are wide. The chosen method, **{{trust/chosen_settings:intervals.chosen_variant}}**, covers {{trust/chosen_settings:intervals.oof.0.9.coverage|.1%}} of the training ads out of fold at the 90 % level (80 % level: {{trust/chosen_settings:intervals.oof.0.8.coverage|.1%}}) with a mean width of {{trust/chosen_settings:intervals.oof.0.9.mean_width_M|.1f}} M KZT and a median relative width of {{trust/chosen_settings:intervals.oof.0.9.median_rel_width_pct|.0f}} %. It was chosen by a fixed rule: valid coverage in both cities at both levels, then the narrowest mean 90 % width.
""")
code("""
cov = read_csv("trust/coverage_table.csv")
if cov is not None and im is not None:
    chosen = im.loc[im["chosen"].astype(str).str.lower() == "true", "variant"].iloc[0] if "chosen" in im else cov["variant"].iloc[0]
    display(cov[(cov["variant"] == chosen) & (cov["level"] == 0.9)].drop(columns=["variant"]).set_index(["group_type", "group"]).round(3))
show_figure("trust/coverage_chosen.png", width=760)
""")
md("""
**Reading.** Coverage per city and per band is the real test of an interval: a method can reach 90 % overall by over-covering standard flats and under-covering expensive ones. With one global width the 90 % interval covers only 74 % of the ads whose *estimate* is above 100 M; calibrating separately per band of the estimate (known at prediction time) fixes this: every estimate band is covered 90.3–91.9 %, both cities 90.3–91.2 %, every fold 90.3–91.0 %, at the price of a wider interval for expensive flats (upper bound +23 % over the estimate for 25–40 M, +54 % above 100 M). By band of the *true* price, coverage is still lower at the extremes (84.5 % above 100 M, 88.1 % below 25 M): the ads that turn out most expensive are exactly those the model under-predicts, a regression-to-the-mean effect that no calibration on the estimate can remove.
""")
needs("trust/overpricing_shares.csv", "trust/flag_threshold_by_band.png")
code("""
ov = read_csv("trust/overpricing_shares.csv")
if ov is not None:
    display(ov.set_index(["group_type", "group"]).round(3))
xt = read_csv("trust/overpricing_crosstab.csv", index_col=0)
if xt is not None:
    display(xt)
show_figure("trust/flag_threshold_by_band.png", width=760)
""")
md("""
**Reading.** The new flag says "over-priced" only when the asking price is above the upper end of the 90 % interval. Out of fold it flags {{trust/chosen_settings:flags.share_over_interval90|.1%}} of the training ads, against {{trust/chosen_settings:flags.share_over_endterm20|.1%}} for the Endterm's "> 20 % above the estimate" rule; every ad the interval flags is also flagged by the 20 % rule (Jaccard {{trust/chosen_settings:flags.jaccard_over|.2f}}), so the new flag is the stricter half. The difference grows with the price: the 20 % rule flags 8 % of the ads estimated below 25 M but 20 % of those above 100 M, the interval rule 4.5–5.4 % in every band, because its threshold adapts to how well the model knows each kind of flat (+23–28 % for standard flats, +54 % above 100 M).

**SHAP.** Shapley values split one prediction into additive contributions of the features, so that base value + contributions = the predicted log price; for trees they are computed exactly (TreeSHAP). We sum the contributions of one-hot columns back to their source feature.
""")
needs("trust/shap_global_importance.csv", "trust/shap_beeswarm.png", "trust/shap_waterfall_example.png")
code("""
shap_imp = read_csv("trust/shap_global_importance.csv", index_col=0)
if shap_imp is not None:
    display(shap_imp.head(12).round(4))
show_figure("trust/shap_beeswarm.png", width=760)
show_figure("trust/shap_waterfall_example.png", width=760)
""")
md("""
**Reading.** SHAP confirms the Endterm's permutation importance: **{{trust/chosen_settings:shap.top_features.0.feature}}** has the largest mean absolute contribution to the log price (0.29), followed by {{trust/chosen_settings:shap.top_features.1.feature}} (0.16) and {{trust/chosen_settings:shap.top_features.2.feature}} (0.05). The ranking is stable when the one-hot model is replaced by an ordinal twin (Spearman {{trust/chosen_settings:shap.importance_spearman_onehot_vs_ordinal|.2f}}), so it does not depend on how the categories were encoded. The waterfall shows the reasons for one ad as percentages of the estimate. SHAP explains the *model*, not the market: a feature correlated with a missing one (ceiling height with building class) takes its credit. In the app, SHAP explains the HGB half of the final blend (the network has no exact TreeSHAP), and the app says so.

**Comparable ads.** For every ad we show the 5 most similar training ads in the same city (log area, rooms, year, floor, position in km), with their prices per m², and warn when even the 5th neighbour is far, i.e. when the estimate rests on few comparables (the DBSCAN "noise" idea of the Endterm made quantitative).
""")
needs("trust/comparables_warning_check.csv", "trust/comparables_distance.png")
code("""
cw = read_csv("trust/comparables_warning_check.csv", index_col=0)
if cw is not None:
    display(cw.round(3))
ce = read_csv("trust/comparables_example.csv")
if ce is not None:
    display(ce)
show_figure("trust/comparables_distance.png", width=760)
""")
md("""
**Reading.** The warning fires for {{trust/chosen_settings:comparables.oof_warning_share|.1%}} of the training ads, and on those ads the base model's error is twice as large as on the rest (MAE 12.6 against 6.3 M, MdAPE 11.3 against 8.8 %): the warning is informative, it marks exactly the ads where the estimate should be trusted less. The comparables themselves are the most convincing output for a non-technical user, because they are real ads that can be opened and checked.
""")

# ----------------------------------------------------------------- section 8
md("""
## 8. The single test evaluation and the November temporal hold-out

The 963 test ads, saved at the Midterm, are scored **once**, by the model chosen in Section 6, after every choice was fixed; the Endterm base is scored on the same ads for comparison. Confidence intervals are 95 % bootstrap intervals over the test ads. The November hold-out is a different kind of test: **1,507 ads scraped on 8 November 2026**, a month after the training data, from the search pages only (the ad pages were blocked by the site's bot protection that day), so they have no coordinates, no full description and no photos, and kitchen area, balcony and parking are missing. It answers a question the random split cannot: does the model still work a month later, on ads it could not have seen, with the inputs it would have in a quick check from a search page?
""")
needs("final_eval/test_metrics.json", "final_eval/results_table.csv")
code("""
tm = read_json("final_eval/test_metrics.json")
if tm:
    display(pd.DataFrame({k: v for k, v in tm.items() if isinstance(v, dict)}).T.round(3))
res = read_csv("final_eval/results_table.csv", index_col=0)
if res is not None:
    display(res)
""")
md("""
**Reading.** On the 963 test ads the chosen model reaches **MAE {{final.test_mae_M|.2f}} M KZT** (95 % bootstrap {{final_eval/test_metrics:chosen.bootstrap95_mae.0|.2f}}–{{final_eval/test_metrics:chosen.bootstrap95_mae.1|.2f}}), **MdAPE {{final.test_mdape|.2f}} %**, R² {{final.test_r2|.3f}}, {{final.within20|.0%}} of ads within ±20 %, against MAE {{endterm.test_mae_M|.2f}} M and MdAPE {{endterm.test_mdape|.2f}} % for the Endterm model on the same ads. The paired bootstrap of the difference (same resampled ads for both models) gives **{{final_eval/test_metrics:paired_diff_mae_M.0|+.2f}} to {{final_eval/test_metrics:paired_diff_mae_M.1|+.2f}} M**: the improvement of the CV (−0.62 M on the repaired folds) is confirmed on unseen ads, about −0.49 M. It is an improvement of the typical ad, not of the extremes: RMSE ({{final_eval/test_metrics:chosen.rmse_M|.2f}} against {{final_eval/test_metrics:base.rmse_M|.2f}} M) and R² ({{final.test_r2|.3f}} against {{final_eval/test_metrics:base.r2|.3f}}) are unchanged, because the largest errors (flats above 100 M) are not smaller. Two caveats: this is the **second** use of the 963 test ads (the Endterm scored them once), so they are no longer a perfectly fresh sample, although no Final choice looked at them; and 13 test ads (1.35 %) have a duplicate in the training ads, too few to move the MAE.
""")
needs("final_eval/nov_holdout.json")
code("""
nov_meta = read_json("nov_holdout/build_meta.json")
if nov_meta:
    print(f"November ads: {nov_meta['rows_out']:,} {nov_meta['by_city']}, scraped {nov_meta['scraped_at_range'][0][:10]}; "
          f"missing vs October: {', '.join(nov_meta['fields_missing_vs_october'])}")
nov = read_json("final_eval/nov_holdout.json")
if nov:
    display(pd.DataFrame({k: v for k, v in nov.items() if isinstance(v, dict) and k != "by_city"}).T.round(3))
    if "by_city" in nov:
        display(pd.DataFrame(nov["by_city"]).T.round(3))
show_figure("final_eval/nov_holdout.png", width=760)
""")
md("""
**Reading.** The November cards carry 17 of the 24 fields and none of the Final blocks, so the fused model cannot be applied to them. To measure *time* separately from *missing inputs*, we retrained the Endterm HGB on the October training ads with only the fields the cards carry, and scored it on the October test ads and on the November ads: MdAPE **{{final_eval/nov_holdout:chosen_on_oct_test.mdape|.2f}} % → {{nov.mdape|.2f}} %**, MAE {{final_eval/nov_holdout:chosen_on_oct_test.mae_M|.2f}} → {{nov.mae_M|.2f}} M, median log error {{final_eval/nov_holdout:chosen.median_log_error|+.3f}} (the October model is not biased a month later; median price per m² rose 1.8 % in Almaty and 1.0 % in Astana). A month of drift costs about 0.35 percentage points of MdAPE, within the bootstrap intervals; the missing inputs cost much more (the full model with the missing fields left empty: MdAPE {{nov.base_mdape|.2f}} %, so a model must be trained for the inputs it will see). Limits: the hold-out is at listing level only (the ad pages were blocked by the site's bot protection on 8 November), and the intervals were not re-checked on it.
""")

# ----------------------------------------------------------------- section 9
md("""
## 9. Error analysis
""")
needs("final_eval/error_by_segment.csv", "final_eval/worst_cases.csv", "final_eval/residuals.png")
code("""
err = read_csv("final_eval/error_by_segment.csv", index_col=0)
if err is not None:
    display(err)
show_figure("final_eval/residuals.png", width=820)
worst = read_csv("final_eval/worst_cases.csv")
if worst is not None:
    display(worst.head(10))
""")
md("""
**Reading.** The error grows with the price in absolute terms (MAE 2.1 M below 25 M, 28.9 M above 100 M), and in relative terms the 25–60 M bands are the best (MdAPE 6.7 and 7.8 %), the tails the worst (8.4 % below 25 M, 12.9 % above 100 M). The model still under-predicts the most expensive band (median log error −0.09) and over-predicts the cheapest (+0.06): a model of the log price shrinks extremes towards the mean when the tails have few ads. The final model is better than the base in every city, price band and room count except **4+ rooms** (MAE +0.84 M, interval crossing zero, 67 ads) and the band above 100 M (−0.17 M, interval ±4.5 M): the gain is in the bulk of the market. Of the 20 largest errors, 17 are flats asked above 100 M; these are the ads the trust layer marks with its widest intervals.
""")

# ----------------------------------------------------------------- section 10
md("""
### 9.1 The app: one listing end to end
`app/gradio_app.py` (UI) runs on `app/predict.py`. For a typed-in listing it builds the final model's inputs from the form: OSM geo features from the coordinates, text flags and the USER-base embedding from the description, the CLIP embedding of up to three photos; raw ad-page fields and the photo track's tabular features cannot come from a form and stay missing (as for training ads without them), and the app says so. Around the estimate it shows the conformal 80 / 90 % interval, an interval-based verdict, SHAP reasons (for the HGB half of the blend), five comparable training ads, text red flags and the photo condition with a mismatch check. The card below was rendered by `scripts/make_app_example.py` for a hand-made listing (not a test ad).
""")
code("""
show_figure("final/app_example.png", width=900)
""")
md("""
**Reading.** The final estimate (46.5 M) and the Endterm base agree for this ordinary flat; the asking price of 52 M is 12 % above the estimate but inside the 80 % interval, so the verdict is "fair", where the Endterm's 20 % rule would also not have fired. The reasons are the location (+9 %) and the stated fresh renovation (+5 %); the photos are not conclusive against the stated condition, and the description raises "urgent sale" and "price negotiable", which in the training ads go with asking prices 1–3 % below the estimate.
""")
md("""
## 10. Limitations, why not deep reinforcement learning, and conclusions

**Limitations and biases.**
- **Asking prices, not sale prices.** krisha.kz shows what sellers ask; the deal price, often lower, is not public. The model learns the asking market, with its optimism, and "over-priced" means "asked above comparable asking prices".
- **One snapshot.** 4,815 ads from 4–8 October 2026 plus 1,507 from 8 November; seasonal effects and price trends beyond one month are not covered.
- **Coverage bias.** Ads with missing fields, few photos or short texts are more often cheap or agency-posted; the blocks help these ads least.
- **Five folds are a coarse ruler.** The rule rejects small but consistent gains (the text block improved every fold); with a larger dataset or repeated CV they might pass.
- **Labels for the photo model are seller-reported** and noisy; the mismatch flag shows how often they disagree with the pictures.
- **OSM completeness** varies between districts; a missing park in OSM looks like no park.
- **November hold-out at listing level only**: the ad pages (coordinates, description, photos) were blocked by the site's bot protection on 8 November, so the fused model could not be tested a month later, only a reduced model of the same family.
- **A reused test set.** The 963 saved test ads were scored at the Endterm and again here; no Final choice looked at them, but they are no longer a perfectly fresh sample.
- **The app sees less than the CV.** A form cannot supply the raw ad-page fields and the photo track's tabular features; the app leaves them missing (as for training ads without them), so its estimates for a typed-in flat have not been validated by CV in exactly that input pattern.
- **Whole-model, not per-block, certification.** The final model passes the rule as a whole; no single block passes it step by step.

**Why no deep reinforcement learning.** Reinforcement learning learns a *policy* from *rewards* for actions taken in an environment. Our data contain no actions and no outcomes: we see asking prices only, not whether or when a flat sold, at what price, or how long it stayed on the market (days-on-market is not shown and we have one snapshot, so we cannot follow an ad over time). A pricing agent would need exactly that reward (sold or not, time to sale, final price) and many interactions with buyers; without it, any "reward" we wrote would be the asking price itself, and RL would collapse into the regression we already have, with more variance. The right tool for "is this price fair" is supervised regression with calibrated uncertainty, which is what we built; RL would become relevant with longitudinal data (price changes and removal dates of the same ad), which is the first item of future work.

**Conclusions.**
1. The Endterm model is reproduced exactly ({{base.cv_mae_M|.3f}} ± {{base.cv_mae_sd_M|.3f}} M CV), and its CV is honest: duplicate ads are few ({{leakage/leakage_summary:all_links.test_ads_with_train_twin_pct|.2f}} % of the test ads have a twin) and grouped folds change the score by much less than one SD.
2. Location, photos and text each contain price information the 24 fields miss (best CV gains {{geo/cv@best:delta_vs_base_M|+.3f}}, {{photos/price_cv@best:delta_vs_base_M|+.3f}} and {{text/cv@best:delta_vs_base_M|+.3f}} M), but no single block passes the 0.354 M rule. Together, in a 50/50 blend of a multimodal network with the Endterm model, they do: CV {{final/selection:whole_model_comparison.cv_mae_M|.3f}} M on the repaired folds (base {{final/selection:base_cv_mae_M|.3f}}).
3. On the 963 saved test ads, scored once: MAE {{final.test_mae_M|.2f}} M (Endterm {{endterm.test_mae_M|.2f}}; paired 95 % interval of the difference {{final_eval/test_metrics:paired_diff_mae_M.0|+.2f}} to {{final_eval/test_metrics:paired_diff_mae_M.1|+.2f}} M), MdAPE {{final.test_mdape|.2f}} % (Endterm {{endterm.test_mdape|.2f}} %). A month later, with the fields of a search card only, the October model loses ~0.35 pp of MdAPE: the drift is small.
4. The largest gain for the user is not a lower MAE but the trust layer: a calibrated 90 % interval, the reasons (SHAP), five real comparable ads and a flag whose threshold adapts to how well the model knows the flat.
""")

# ----------------------------------------------------------------- section 11
md("""
## 11. Contributions and AI tools

**Rakhat Balgabekov** — the shared `krisha_ml` foundation and the reproduction of the Endterm, the leakage audit and grouped folds, the photo track (CLIP, probes, fine-tuning, Grad-CAM), the November scrape, the deliverable builders.
**Aitbek Kaiyrbekuly** — the location track (OSM features, raw ad-page fields), the text track (TF-IDF, transformers, red flags), the trust layer (conformal intervals, SHAP, comparables), the error analysis and the slides.
Both: the selection rule, the readings under every table and plot and the conclusions were discussed and written together; both can explain every part of the notebook.

**AI tools.** We used an AI coding assistant (Claude, via Claude Code) more heavily than in the Endterm, and we state it plainly. For the Final we ran several assistant sessions in parallel, one per track (leakage, location, photos, text, trust, integration), on a shared code contract (`CONTRACT.md`) with one CV harness and one fixed selection rule, so that no track could choose its own yardstick. The assistant wrote most of the code in `krisha_ml/`, `scripts/` and `app/`, ran the experiments on our machine, drafted the readings, the report and the slides from the outputs, and reviewed the tracks against each other (for example, the leakage review that corrected the rule threshold from 6.459 to 6.370 M). We chose the question and the data, set the selection rule and the one-time use of the test ads, decided what to keep, checked the numbers and the readings against the outputs, and are responsible for every claim; both of us can explain every part of the notebook.
""")


# --------------------------------------------------------------------------- build
def build(values: Values | None = None) -> tuple[nbf.NotebookNode, list[str]]:
    values = values or Values()
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    cells, unfilled = [], []
    for kind, body in CELLS:
        if kind == "needs":
            note = pending_note(body, values)
            if note:
                cells.append(nbf.v4.new_markdown_cell(note))
        elif kind == "md":
            text = fill(body, values, missing="keep")
            unfilled += placeholders(text)
            cells.append(nbf.v4.new_markdown_cell(text))
        else:
            cells.append(nbf.v4.new_code_cell(body))
    nb["cells"] = cells
    return nb, unfilled


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(NOTEBOOK))
    ap.add_argument("--strict", action="store_true", help="exit 1 if any placeholder is unfilled")
    args = ap.parse_args(argv)
    nb, unfilled = build()
    nbf.validate(nb)
    nbf.write(nb, args.out)
    n_pending = sum("**Pending.**" in "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"{args.out}: {len(nb['cells'])} cells, {n_pending} pending notes, {len(unfilled)} unfilled placeholders")
    if unfilled:
        print("  unfilled:", ", ".join(sorted(set(unfilled))))
    return 1 if (args.strict and unfilled) else 0


if __name__ == "__main__":
    sys.exit(main())
