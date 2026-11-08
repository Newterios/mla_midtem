# Final report — apartment asking-price prediction (SE-2424, Balgabekov & Kaiyrbekuly)

Full text, code and figures: `Final_SE-2424_Balgabekov_Kaiyrbekuly.ipynb`; slides: `slides/Final_SE-2424_slides.pptx`. This file collects the tables. All CV numbers: 5-fold CV on the 3,852 training ads, mean ± fold SD. Track tables (Sections 1–3) use the Endterm folds (KFold 5, shuffle, seed 42); the model choice (Section 4) uses the **repaired folds** of the leakage audit (the same folds with every duplicate group moved whole into one fold, `outputs/leakage/folds_group.csv`) as the primary scheme and reports the random folds too. **Selection rule:** kept only if the CV MAE is below the base by more than the fixed Endterm margin of 0.354 M KZT (its fold SD): below **6.370 M** on the repaired folds (base 6.723 ± 0.264), below 6.262 M on the random folds. (An earlier draft used 6.459 = 6.723 − 0.264, the fold SD of the repaired folds; the leakage review corrected it, because the fold SD moves 0.26–0.50 M with the fold seed.)

**Base (Endterm, reproduced):** CV MAE **6.616 ± 0.353 M**, MdAPE 8.93 %; test MAE 6.29 M, MdAPE 9.13 %, R² 0.855.
**Final model:** C network + base, 50/50 log-blend (geo+rawfields+photo+text_flags+text_emb+photo_emb): CV MAE **6.100 ± 0.388 M** on the repaired folds (6.092 ± 0.468 on the random folds); test MAE **5.80 M** (95 % bootstrap 5.09–6.67; paired difference to the Endterm -0.80 to -0.16 M), MdAPE **7.82 %**, R² 0.853. November hold-out (1,507 ads, listing level, reduced model on the card fields): MdAPE 10.16 % on the October test → 10.51 % in November.

**Sources, rights and matching.** Every new input attaches to the same 4,815 ads by the ad id of the October scrape; the saved train/test split is unchanged. *Text:* the description field of the public ad pages read on 4–8 October (robots.txt allows the ad pages, no login; data card), masked for contacts at the Endterm and stripped of money sentences, digits and contacts before any model sees it (4,735 ads with a description, 4,593 with 100+ characters). *Photos:* up to five image links per ad from the same pages, downloaded once into a local cache for feature extraction only, not committed, redistributed or shown (Grad-CAM figures with krisha.kz photos stay local); 4,574 ads (95 %) have a usable photo, the rest are "no photo", as in training. *Location:* OpenStreetMap amenities around the 4,802 ads with coordinates, via the Overpass API, fetched once and cached (© OpenStreetMap contributors, ODbL). *November hold-out:* 1,507 search-page cards scraped on 8 November from the public listing pages (ad pages were blocked by the site's bot protection), a separate sample never joined to the training rows. An ad with a missing source keeps its structured fields and gets missing indicators; no ad is dropped.

## Status of the track outputs

| track                       | outputs present   | missing   |
|:----------------------------|:------------------|:----------|
| core (Endterm reproduction) | 2/2               | -         |
| leakage audit               | 3/3               | -         |
| location                    | 3/3               | -         |
| photos                      | 3/3               | -         |
| text                        | 3/3               | -         |
| trust                       | 3/3               | -         |
| fusion                      | 2/2               | -         |
| final evaluation            | 2/2               | -         |

## 1. The base: Endterm model reproduced

|                |   Endterm (submitted) |   krisha_ml reproduction |
|:---------------|----------------------:|-------------------------:|
| CV MAE (M)     |                6.616  |                   6.616  |
| CV fold SD (M) |                0.3535 |                   0.3535 |
| CV MdAPE %     |                8.934  |                   8.934  |
| CV R²          |                0.8569 |                   0.8569 |
| test MAE (M)   |                6.2909 |                   6.2909 |
| test MdAPE %   |                9.1293 |                   9.1293 |
| test R²        |                0.8549 |                   0.8549 |

## 2. Leakage audit: duplicate ads

48 duplicate pairs (photo hashes, image embeddings, structured fields, text) in 43 groups.

|           |   groups |   ads_in_groups |   groups_crossing_split |   test_ads_with_train_twin |   test_ads_with_train_twin_pct |
|:----------|---------:|----------------:|------------------------:|---------------------------:|-------------------------------:|
| all_links |       43 |              90 |                      13 |                         13 |                           1.35 |
| same_flat |       18 |              36 |                       7 |                          7 |                           0.73 |

Pairs by signal: phash+emb 28, struct 6, phash 6, text 2, phash+emb+struct+text 2, emb+struct+text 1, emb 1, struct+text 1, phash+emb+struct 1

**CV schemes** (base model): random folds vs folds that keep duplicate groups / buildings / spatial blocks together.

| scheme        | description                                            |   cv_mae_mean |   seed_sd |   fold_sd_mean |   mdape |     r2 |   n_seeds |   delta_vs_random_M |
|:--------------|:-------------------------------------------------------|--------------:|----------:|---------------:|--------:|-------:|----------:|--------------------:|
| random        | Random KFold (Endterm)                                 |        6.7197 |    0.0576 |         0.3707 |  8.9595 | 0.8469 |        20 |              0      |
| random_dupfix | Same KFold, duplicate groups kept together             |        6.7344 |    0.0644 |         0.3667 |  8.9552 | 0.8461 |        20 |              0.0147 |
| dup           | GroupKFold by duplicate group                          |        6.7238 |    0.0806 |         0.3724 |  8.9217 | 0.8474 |        10 |              0.0041 |
| building      | GroupKFold by building (+dups)                         |        7.4901 |    0.1143 |         0.5689 |  9.9161 | 0.816  |        10 |              0.7704 |
| spatial1km    | Spatial 1 km blocks (+dups)                            |        7.6921 |    0.0929 |         0.6996 | 10.2592 | 0.8128 |        10 |              0.9724 |
| spatial3km    | Spatial 3 km blocks (+dups)                            |        8.3949 |    0.1871 |         1.1766 | 11.0186 | 0.7768 |        10 |              1.6752 |
| placebo       | Placebo: same KFold, 26 random non-duplicate ads moved |        6.7166 |    0.0688 |         0.3754 |  8.8972 | 0.8466 |        20 |             -0.0031 |

## 3. Every block against the base (best variant per block)

| block                       | best variant                                                             | CV MAE (M KZT)   |   Δ vs base (M) | folds better   | variants kept   |
|:----------------------------|:-------------------------------------------------------------------------|:-----------------|----------------:|:---------------|:----------------|
| location (OSM + raw fields) | base + geo + raw fields                                                  | 6.481 ± 0.280    |          -0.135 | 4/5            | 0/4             |
| photos                      | residual: HGB on condition probs + zero-shot                             | 6.426 ± 0.440    |          -0.19  | 4/5            | 0/14            |
| text                        | TF-IDF word+char + user-base embedding + regex flags, one Ridge residual | 6.336 ± 0.392    |          -0.28  | 5/5            | 0/20            |
| fusion of kept blocks       | C: network, all blocks, 50/50 log-blend with base                        | 6.092 ± 0.468    |          -0.524 | 5/5            | 5/38            |

## 3.1 Location (OSM + raw ad-page fields)

| variant                        | CV MAE (M KZT)   |   CV MdAPE % |   Δ vs base (M) | folds better   | kept (1-SD rule)   |
|:-------------------------------|:-----------------|-------------:|----------------:|:---------------|:-------------------|
| base (Endterm)                 | 6.616 ± 0.353    |         8.93 |           0     | 0/5            | no                 |
| base + geo (OSM)               | 6.531 ± 0.251    |         8.82 |          -0.085 | 3/5            | no                 |
| base + raw fields              | 6.766 ± 0.291    |         8.91 |           0.15  | 1/5            | no                 |
| base + geo + raw fields        | 6.481 ± 0.280    |         8.53 |          -0.135 | 4/5            | no                 |
| base + distance to centre only | 6.543 ± 0.243    |         8.67 |          -0.073 | 3/5            | no                 |

## 3.2 Photos in the price model

| variant                                               | CV MAE (M KZT)   |   CV MdAPE % |   Δ vs base (M) | folds better   | kept (1-SD rule)   |
|:------------------------------------------------------|:-----------------|-------------:|----------------:|:---------------|:-------------------|
| Endterm base (tuned HGB, 24 features)                 | 6.616 ± 0.353    |         8.93 |         nan     |                |                    |
| photo condition probabilities (nested OOF probe)      | 6.617 ± 0.298    |         8.43 |           0.001 | 3/5            | no                 |
| condition probs + stated condition filled from photos | 6.655 ± 0.336    |         8.69 |           0.039 | 1/5            | no                 |
| condition probs only where condition is not stated    | 6.500 ± 0.221    |         8.65 |          -0.116 | 4/5            | no                 |
| CLIP zero-shot quality + room shares                  | 6.824 ± 0.413    |         8.91 |           0.208 | 1/5            | no                 |
| CLIP zero-shot room shares only                       | 6.873 ± 0.274    |         9.24 |           0.257 | 1/5            | no                 |
| CLIP embedding PCA(16)                                | 6.707 ± 0.408    |         8.83 |           0.091 | 2/5            | no                 |
| legacy MobileNetV3 embedding PCA(16)                  | 7.056 ± 0.340    |         9.43 |           0.44  | 0/5            | no                 |
| condition probs + zero-shot                           | 6.779 ± 0.415    |         8.92 |           0.163 | 2/5            | no                 |
| condition probs + zero-shot + CLIP PCA(16)            | 6.885 ± 0.456    |         9.2  |           0.269 | 0/5            | no                 |
| residual: RidgeCV on CLIP ad embedding                | 6.437 ± 0.385    |         8.83 |          -0.179 | 5/5            | no                 |
| residual: RidgeCV on CLIP interior-weighted embedding | 6.441 ± 0.391    |         8.77 |          -0.174 | 5/5            | no                 |
| residual: RidgeCV on legacy MobileNetV3 embedding     | 6.517 ± 0.389    |         8.85 |          -0.099 | 5/5            | no                 |
| residual: HGB on condition probs + zero-shot          | 6.426 ± 0.440    |         8.66 |          -0.19  | 4/5            | no                 |

## 3.3 Text (TF-IDF, transformer embeddings, fine-tuning, red flags)

| variant                                                                    | CV MAE (M KZT)   |   CV MdAPE % |   Δ vs base (M) | folds better   | kept (1-SD rule)   |
|:---------------------------------------------------------------------------|:-----------------|-------------:|----------------:|:---------------|:-------------------|
| TF-IDF word 1-2 + Ridge residual                                           | 6.398 ± 0.381    |         8.72 |          -0.218 | 5/5            | no                 |
| TF-IDF char 3-5 + Ridge residual                                           | 6.402 ± 0.385    |         8.71 |          -0.214 | 5/5            | no                 |
| TF-IDF word+char + Ridge residual                                          | 6.388 ± 0.387    |         8.7  |          -0.228 | 5/5            | no                 |
| TF-IDF word+char + Ridge residual, price-weighted                          | 6.436 ± 0.372    |         8.8  |          -0.18  | 4/5            | no                 |
| TF-IDF word+char OOF residual as 1 HGB feature                             | 6.571 ± 0.285    |         8.82 |          -0.044 | 3/5            | no                 |
| rubert-tiny2 embedding PCA(32) as HGB features                             | 7.313 ± 0.293    |         9.8  |           0.697 | 0/5            | no                 |
| user-base embedding PCA(32) as HGB features                                | 7.150 ± 0.344    |         9.5  |           0.534 | 0/5            | no                 |
| e5-base embedding PCA(32) as HGB features                                  | 7.278 ± 0.279    |         9.5  |           0.662 | 0/5            | no                 |
| user-base embedding + Ridge residual                                       | 6.402 ± 0.370    |         8.69 |          -0.214 | 5/5            | no                 |
| rubert-tiny2 embedding + Ridge residual                                    | 6.469 ± 0.352    |         8.7  |          -0.147 | 5/5            | no                 |
| TF-IDF word+char + user-base embedding, one Ridge residual                 | 6.353 ± 0.380    |         8.61 |          -0.263 | 5/5            | no                 |
| TF-IDF word+char + user-base embedding, one Ridge residual, price-weighted | 6.360 ± 0.375    |         8.76 |          -0.256 | 5/5            | no                 |
| TF-IDF word+char + user-base embedding + regex flags, one Ridge residual   | 6.336 ± 0.392    |         8.64 |          -0.28  | 5/5            | no                 |
| user-base embedding PCA(8) as HGB features                                 | 6.921 ± 0.366    |         9.18 |           0.305 | 0/5            | no                 |
| TF-IDF word+char + all 3 embeddings + regex flags, one Ridge residual      | 6.353 ± 0.382    |         8.59 |          -0.264 | 5/5            | no                 |
| regex flags (12) as HGB features                                           | 6.588 ± 0.312    |         8.9  |          -0.028 | 3/5            | no                 |
| regex flags (12) + Ridge residual                                          | 6.531 ± 0.390    |         8.75 |          -0.085 | 5/5            | no                 |
| flag similarity scores (12, user-base) as HGB features                     | 7.092 ± 0.350    |         9.44 |           0.476 | 0/5            | no                 |
| rubert-tiny2 fine-tuned on fold residuals (3 ep)                           | 6.496 ± 0.389    |         8.8  |          -0.12  | 5/5            | no                 |
| e5-base embedding + Ridge residual                                         | 6.477 ± 0.364    |         8.81 |          -0.139 | 5/5            | no                 |

## 3.4 Photos: condition classifier (fresh / good / needs repair / shell)

Frozen CNN + logistic regression, CLIP linear probe (5-fold grouped CV) and fine-tuned EfficientNet-B0 (fold 0 only).

| model                                                            |   macro_f1_cv |   macro_f1_sd |   accuracy_cv |   balanced_acc |   macro_f1_fold0 |   accuracy_fold0 |
|:-----------------------------------------------------------------|--------------:|--------------:|--------------:|---------------:|-----------------:|-----------------:|
| (a) frozen mnv3 + LR                                             |         0.592 |         0.028 |         0.678 |          0.622 |            0.552 |            0.656 |
| (a) frozen effnet_b0 + LR                                        |         0.608 |         0.029 |         0.697 |          0.625 |            0.588 |            0.689 |
| (c) CLIP linear probe                                            |         0.632 |         0.016 |         0.721 |          0.657 |            0.642 |            0.727 |
| (c) CLIP linear probe [interior-weighted]                        |         0.638 |         0.016 |         0.723 |          0.667 |            0.622 |            0.709 |
| (a) frozen effnet_b0 + LR [interior-weighted]                    |         0.613 |         0.032 |         0.699 |          0.623 |            0.584 |            0.678 |
| (b) fine-tuned EfficientNet-B0                                   |       nan     |       nan     |       nan     |          0.586 |            0.546 |            0.616 |
| (b) fine-tuned EfficientNet-B0 (lr 0.0002, interior photos only) |       nan     |       nan     |       nan     |          0.598 |            0.577 |            0.68  |

## 3.5 Text red flags vs base residuals

| flag                 |   n_train_ads |   share_pct |   median_resid_pct_flagged |   ci95_lo_pct |   ci95_hi_pct |   median_resid_pct_rest |   mean_abs_resid_pct_flagged |   mannwhitney_p |   n_all_clean_ads |   sim_auc_vs_regex | sim_model   |
|:---------------------|--------------:|------------:|---------------------------:|--------------:|--------------:|------------------------:|-----------------------------:|----------------:|------------------:|-------------------:|:------------|
| cash_only            |           114 |        2.96 |                      -6.5  |         -9.05 |         -3.64 |                   -0.33 |                        14.27 |     1.31968e-06 |               136 |              0.738 | user-base   |
| two_level            |            36 |        0.93 |                      -6.17 |        -10.06 |         -1.76 |                   -0.47 |                        17.27 |     0.0412769   |                44 |              0.772 | user-base   |
| assignment_of_rights |            85 |        2.21 |                      -5.41 |         -6.58 |         -2.39 |                   -0.35 |                        12.06 |     0.000149971 |               122 |              0.867 | user-base   |
| former_dorm_text     |            32 |        0.83 |                      -4.81 |         -9.91 |         -0.36 |                   -0.49 |                        13.25 |     0.0465504   |                38 |              0.928 | user-base   |
| needs_repair         |            46 |        1.19 |                      -3.66 |         -7.29 |         -0.73 |                   -0.48 |                        11.02 |     0.119955    |                60 |              0.755 | user-base   |
| rough_finish         |           387 |       10.05 |                      -3.65 |         -5.92 |         -2.33 |                   -0.15 |                        12.96 |     7.71915e-07 |               492 |              0.688 | user-base   |
| urgent               |           293 |        7.61 |                      -2.82 |         -4.57 |         -1.47 |                   -0.29 |                        11.9  |     1.56297e-05 |               358 |              0.738 | user-base   |
| panoramic_view       |           309 |        8.02 |                      -1.99 |         -3.51 |         -0.53 |                   -0.33 |                        13.03 |     0.513205    |               387 |              0.692 | user-base   |
| encumbrance          |           149 |        3.87 |                      -1.35 |         -3.28 |          0.46 |                   -0.5  |                        10.04 |     0.731052    |               184 |              0.68  | user-base   |
| bargain              |           571 |       14.82 |                      -1.27 |         -1.99 |         -0    |                   -0.35 |                        12.38 |     0.492018    |               716 |              0.707 | user-base   |
| no_encumbrance       |           304 |        7.89 |                      -1.22 |         -3.09 |          0.18 |                   -0.47 |                        11.01 |     0.060193    |               384 |              0.699 | user-base   |
| designer_renovation  |           376 |        9.76 |                       2.01 |          0.31 |          4.06 |                   -0.83 |                        14.06 |     0.000276701 |               467 |              0.66  | user-base   |

## 4. Fusion and model selection

Strategies: **A** residual stacking (base HGB + cross-fitted log-residual corrections per block), **B** blocks as extra HGB columns, **C** a small multimodal PyTorch network (tabular + text-embedding + photo-embedding branches), alone and **50/50 log-blended with the base** (weight fixed in advance). Whole models on both fold schemes:

| variant                                                   |   CV MAE repaired |    SD |   Δ vs base |   folds better | passes (repaired)   |   CV MAE random | passes (random)   |
|:----------------------------------------------------------|------------------:|------:|------------:|---------------:|:--------------------|----------------:|:------------------|
| Endterm base                                              |             6.723 | 0.264 |       0     |              0 | False               |           6.616 | False             |
| A: geo+rawfields+photo+text_flags+text+text_emb+photo_emb |             6.253 | 0.372 |      -0.47  |              5 | True                |           6.17  | True              |
| B: geo+rawfields+photo+text_flags+text+text_emb+photo_emb |             6.365 | 0.397 |      -0.358 |              4 | True                |           6.445 | False             |
| C: network, tabular only                                  |             8.596 | 0.316 |       1.872 |              0 | False               |           8.598 | False             |
| C: network, all blocks                                    |             7.293 | 0.346 |       0.57  |              0 | False               |           7.258 | False             |
| C: network, all blocks, 50/50 log-blend with base         |             6.1   | 0.388 |      -0.623 |              5 | True                |           6.092 | True              |

## 4.1 Each block alone (CV MAE, M KZT)

No single block passes the rule in any strategy or fold scheme.

| blocks     |   ('A', 'group') |   ('A', 'random') |   ('B', 'group') |   ('B', 'random') |
|:-----------|-----------------:|------------------:|-----------------:|------------------:|
| geo        |            6.572 |             6.489 |            6.51  |             6.531 |
| photo      |            6.482 |             6.352 |            6.816 |             6.776 |
| photo_emb  |            6.548 |             6.435 |            6.81  |             6.712 |
| rawfields  |            6.7   |             6.57  |            6.753 |             6.766 |
| text       |            6.513 |             6.402 |            6.691 |             6.849 |
| text_emb   |            6.519 |             6.402 |            7.016 |             7.052 |
| text_flags |            6.678 |             6.559 |            6.586 |             6.588 |

## 4.2 Strict per-step forward selection

Repaired folds, final blocks. Every step must lower the CV MAE by more than the fold SD of the current set, never less than 0.354 M. The first and largest step is already below the margin, so **the strict per-step reading keeps no block** (the Endterm base).

|          | added      |   cv_mae_M |   gain_M |   step_sd_M | passes_step_rule   | set_passes_vs_base   |
|:---------|:-----------|-----------:|---------:|------------:|:-------------------|:---------------------|
| ('A', 1) | photo      |     6.4824 |   0.2406 |      0.264  | False              | False                |
| ('A', 2) | text_emb   |     6.3499 |   0.1325 |      0.3209 | False              | True                 |
| ('A', 3) | geo        |     6.2578 |   0.0921 |      0.3481 | False              | True                 |
| ('A', 4) | text       |     6.2483 |   0.0096 |      0.3465 | False              | True                 |
| ('A', 5) | text_flags |     6.2465 |   0.0018 |      0.368  | False              | True                 |
| ('A', 6) | rawfields  |     6.2458 |   0.0006 |      0.3723 | False              | True                 |
| ('B', 1) | geo        |     6.5097 |   0.2133 |      0.264  | False              | False                |
| ('B', 2) | text_flags |     6.4082 |   0.1015 |      0.3273 | False              | False                |
| ('B', 3) | rawfields  |     6.3977 |   0.0105 |      0.2735 | False              | False                |

## 4.3 Ablation of the network blend

Repaired folds. The blend with a tabular-only network is worse than the base, so the gain is not plain ensembling; it comes from the blocks: location matters most (without it the blend no longer passes), then the text and photo embeddings; dropping the photo-track tabular features or the raw fields changes the CV by less than fold noise.

| variant                                 |   cv_mae_M |   cv_mae_sd_M |   delta_vs_base_M |   folds_better |
|:----------------------------------------|-----------:|--------------:|------------------:|---------------:|
| Cb: tabular only (ensembling control)   |      7.134 |         0.229 |             0.411 |              0 |
| Cb: embeddings only: text_emb+photo_emb |      6.293 |         0.346 |            -0.43  |              5 |
| Cb: all minus geo                       |      6.391 |         0.462 |            -0.332 |              4 |
| Cb: all minus rawfields                 |      6.076 |         0.466 |            -0.647 |              5 |
| Cb: all minus photo                     |      6.033 |         0.366 |            -0.69  |              5 |
| Cb: all minus text_flags                |      6.104 |         0.389 |            -0.62  |              5 |
| Cb: all minus text_emb                  |      6.191 |         0.337 |            -0.532 |              5 |
| Cb: all minus photo_emb                 |      6.161 |         0.381 |            -0.562 |              5 |
| Cb: all blocks                          |      6.1   |         0.388 |            -0.623 |              5 |

**Decision (whole-model comparison).** The network + Endterm base 50/50 log-blend wins by **0.623 M** > 0.354 M on the repaired folds (better on 5/5 folds; on the random folds 6.092 M, -0.524), its blocks beat the tabular-only control, and it is the lowest CV of all candidates on both schemes, so it is the **final model**. We report honestly that no single block is certified step by step; the final model is certified as a whole. Only training-ad CV was used; the test ads were scored once, afterwards.

## 5. Trust layer: conformal intervals (out-of-fold, training ads)

Chosen: **abs-residual (base HGB) | Mondrian estimate band** (valid per-city coverage at 80 % and 90 %, then the narrowest 90 % width).

|                                                           |   coverage |   coverage_fold_sd |   coverage_Almaty |   coverage_Astana |   min_band_coverage |   min_estimate_band_coverage |   mean_width_M |   median_width_M |   median_rel_width_pct |   share_above |   share_below | valid_per_city   | valid_per_estimate_band   | valid_both_levels   | chosen   |
|:----------------------------------------------------------|-----------:|-------------------:|------------------:|------------------:|--------------------:|-----------------------------:|---------------:|-----------------:|-----------------------:|--------------:|--------------:|:-----------------|:--------------------------|:--------------------|:---------|
| (0.8, 'abs-residual (base HGB) | Mondrian city')          |      0.815 |              0.013 |             0.82  |             0.809 |               0.585 |                        0.608 |         18.826 |           15.511 |                 38.012 |         0.094 |         0.092 | True             | False                     | False               | False    |
| (0.8, 'abs-residual (base HGB) | global')                 |      0.815 |              0.014 |             0.816 |             0.815 |               0.585 |                        0.608 |         18.835 |           15.543 |                 38.075 |         0.093 |         0.092 | True             | False                     | False               | False    |
| (0.8, 'abs-residual (base HGB) | Mondrian estimate band') |      0.818 |              0.018 |             0.822 |             0.814 |               0.74  |                        0.802 |         21.252 |           15.007 |                 36.101 |         0.094 |         0.088 | True             | True                      | True                | True     |
| (0.8, 'CQR endterm crossfit | Mondrian city')             |      0.81  |              0.01  |             0.808 |             0.811 |               0.713 |                        0.792 |         24.244 |           16.229 |                 41.146 |         0.099 |         0.091 | True             | True                      | True                | False    |
| (0.8, 'CQR smooth crossfit | global')                     |      0.811 |              0.018 |             0.797 |             0.824 |               0.679 |                        0.791 |         24.246 |           16.244 |                 40.909 |         0.096 |         0.092 | True             | True                      | False               | False    |
| (0.8, 'CQR endterm crossfit | global')                    |      0.81  |              0.012 |             0.799 |             0.819 |               0.713 |                        0.788 |         24.253 |           16.212 |                 41.239 |         0.098 |         0.092 | True             | True                      | True                | False    |
| (0.8, 'CQR smooth crossfit | Mondrian city')              |      0.81  |              0.016 |             0.803 |             0.816 |               0.672 |                        0.787 |         24.262 |           16.272 |                 40.904 |         0.098 |         0.093 | True             | True                      | False               | False    |
| (0.8, 'CQR endterm split | global')                       |      0.801 |              0.014 |             0.793 |             0.808 |               0.672 |                        0.767 |         24.425 |           16.556 |                 41.509 |         0.101 |         0.098 | True             | True                      | False               | False    |
| (0.8, 'CQR endterm split | Mondrian city')                |      0.804 |              0.015 |             0.802 |             0.806 |               0.668 |                        0.764 |         24.514 |           16.725 |                 41.832 |         0.1   |         0.095 | True             | True                      | False               | False    |
| (0.8, 'CQR smooth split | global')                        |      0.808 |              0.025 |             0.799 |             0.816 |               0.657 |                        0.775 |         24.74  |           16.52  |                 42.362 |         0.098 |         0.094 | True             | True                      | False               | False    |
| (0.8, 'CQR smooth split | Mondrian city')                 |      0.81  |              0.023 |             0.804 |             0.815 |               0.649 |                        0.767 |         24.81  |           16.625 |                 42.471 |         0.099 |         0.092 | True             | True                      | False               | False    |
| (0.9, 'abs-residual (base HGB) | global')                 |      0.911 |              0.007 |             0.914 |             0.908 |               0.713 |                        0.744 |         25.633 |           21.166 |                 51.733 |         0.045 |         0.044 | True             | False                     | False               | False    |
| (0.9, 'abs-residual (base HGB) | Mondrian city')          |      0.91  |              0.006 |             0.914 |             0.906 |               0.713 |                        0.74  |         25.633 |           21.209 |                 51.812 |         0.046 |         0.045 | True             | False                     | False               | False    |
| (0.9, 'abs-residual (base HGB) | Mondrian estimate band') |      0.907 |              0.003 |             0.912 |             0.903 |               0.845 |                        0.903 |         29.073 |           19.976 |                 48.137 |         0.049 |         0.044 | True             | True                      | True                | True     |
| (0.9, 'CQR endterm crossfit | global')                    |      0.912 |              0.009 |             0.91  |             0.914 |               0.789 |                        0.876 |         33.314 |           23.158 |                 57.864 |         0.048 |         0.041 | True             | True                      | True                | False    |
| (0.9, 'CQR endterm crossfit | Mondrian city')             |      0.912 |              0.01  |             0.912 |             0.912 |               0.789 |                        0.876 |         33.356 |           23.181 |                 57.968 |         0.046 |         0.041 | True             | True                      | True                | False    |
| (0.9, 'CQR smooth crossfit | global')                     |      0.906 |              0.01  |             0.903 |             0.909 |               0.736 |                        0.857 |         33.476 |           23.15  |                 58.555 |         0.05  |         0.044 | True             | False                     | False               | False    |
| (0.9, 'CQR smooth crossfit | Mondrian city')              |      0.908 |              0.008 |             0.906 |             0.91  |               0.732 |                        0.853 |         33.524 |           23.205 |                 58.735 |         0.049 |         0.043 | True             | False                     | False               | False    |
| (0.9, 'CQR endterm split | Mondrian city')                |      0.906 |              0.014 |             0.912 |             0.9   |               0.758 |                        0.849 |         34.283 |           24.136 |                 60.186 |         0.051 |         0.043 | True             | False                     | False               | False    |
| (0.9, 'CQR endterm split | global')                       |      0.907 |              0.013 |             0.909 |             0.906 |               0.762 |                        0.849 |         34.411 |           24.214 |                 60.544 |         0.05  |         0.043 | True             | False                     | False               | False    |
| (0.9, 'CQR smooth split | global')                        |      0.913 |              0.013 |             0.916 |             0.911 |               0.732 |                        0.841 |         34.773 |           24.222 |                 61.739 |         0.048 |         0.039 | True             | False                     | False               | False    |
| (0.9, 'CQR smooth split | Mondrian city')                 |      0.915 |              0.014 |             0.916 |             0.913 |               0.736 |                        0.849 |         35.08  |           24.506 |                 62.332 |         0.047 |         0.038 | True             | False                     | False               | False    |

## 5.1 Coverage of the chosen 90 % interval by city, price band and fold

|                               |   level |    n |   coverage |   share_above |   share_below |   mean_width_M |   median_width_M |   median_rel_width_pct |
|:------------------------------|--------:|-----:|-----------:|--------------:|--------------:|---------------:|-----------------:|-----------------------:|
| ('overall', 'all')            |     0.9 | 3852 |      0.907 |         0.049 |         0.044 |         29.073 |           19.976 |                 48.137 |
| ('city', 'Almaty')            |     0.9 | 1849 |      0.912 |         0.049 |         0.039 |         32.308 |           21.234 |                 48.535 |
| ('city', 'Astana')            |     0.9 | 2003 |      0.903 |         0.049 |         0.048 |         26.087 |           16.316 |                 47.877 |
| ('price band', '25-40 M')     |     0.9 | 1381 |      0.919 |         0.028 |         0.053 |         14.576 |           13.822 |                 43.658 |
| ('price band', '40-60 M')     |     0.9 | 1097 |      0.922 |         0.052 |         0.026 |         24.309 |           23.164 |                 48.472 |
| ('price band', '60-100 M')    |     0.9 |  629 |      0.903 |         0.075 |         0.022 |         47.526 |           45.289 |                 61.864 |
| ('price band', '<25 M')       |     0.9 |  480 |      0.881 |         0.01  |         0.108 |         10.436 |           10.707 |                 50.115 |
| ('price band', '>=100 M')     |     0.9 |  265 |      0.845 |         0.151 |         0.004 |        114.305 |          108.08  |                 79.416 |
| ('estimate band', '25-40 M')  |     0.9 | 1374 |      0.906 |         0.05  |         0.044 |         13.71  |           13.679 |                 42.429 |
| ('estimate band', '40-60 M')  |     0.9 | 1149 |      0.91  |         0.046 |         0.044 |         23.605 |           23.306 |                 49.24  |
| ('estimate band', '60-100 M') |     0.9 |  607 |      0.903 |         0.054 |         0.043 |         47.831 |           46.251 |                 64.595 |
| ('estimate band', '<25 M')    |     0.9 |  464 |      0.905 |         0.045 |         0.05  |         10.354 |           10.572 |                 48.852 |
| ('estimate band', '>=100 M')  |     0.9 |  258 |      0.919 |         0.05  |         0.031 |        124.779 |          110.902 |                 89.825 |
| ('fold', '1')                 |     0.9 |  771 |      0.905 |         0.053 |         0.042 |         26.58  |           20.022 |                 46.56  |
| ('fold', '2')                 |     0.9 |  771 |      0.903 |         0.053 |         0.044 |         28.401 |           19.209 |                 46.275 |
| ('fold', '3')                 |     0.9 |  770 |      0.908 |         0.048 |         0.044 |         28.9   |           20.852 |                 49.546 |
| ('fold', '4')                 |     0.9 |  770 |      0.91  |         0.04  |         0.049 |         29.62  |           19.682 |                 49.398 |
| ('fold', '5')                 |     0.9 |  770 |      0.91  |         0.049 |         0.04  |         31.87  |           20.465 |                 48.797 |

## 5.2 Over-pricing flags: interval rule vs the Endterm 20 % rule

|                               |    n |   over_interval90 |   under_interval90 |   over_endterm20 |   under_endterm20 |   over_both |
|:------------------------------|-----:|------------------:|-------------------:|-----------------:|------------------:|------------:|
| ('overall', 'all')            | 3852 |             0.049 |              0.044 |            0.099 |             0.06  |       0.049 |
| ('city', 'Almaty')            | 1849 |             0.049 |              0.039 |            0.101 |             0.055 |       0.049 |
| ('city', 'Astana')            | 2003 |             0.049 |              0.048 |            0.097 |             0.065 |       0.049 |
| ('price band', '<25 M')       |  480 |             0.01  |              0.108 |            0.025 |             0.112 |       0.01  |
| ('price band', '25-40 M')     | 1381 |             0.028 |              0.053 |            0.046 |             0.051 |       0.028 |
| ('price band', '40-60 M')     | 1097 |             0.052 |              0.026 |            0.096 |             0.049 |       0.052 |
| ('price band', '60-100 M')    |  629 |             0.075 |              0.022 |            0.172 |             0.06  |       0.075 |
| ('price band', '>=100 M')     |  265 |             0.151 |              0.004 |            0.347 |             0.057 |       0.151 |
| ('estimate band', '<25 M')    |  464 |             0.045 |              0.05  |            0.084 |             0.062 |       0.045 |
| ('estimate band', '25-40 M')  | 1374 |             0.05  |              0.044 |            0.074 |             0.036 |       0.05  |
| ('estimate band', '40-60 M')  | 1149 |             0.046 |              0.044 |            0.097 |             0.053 |       0.046 |
| ('estimate band', '60-100 M') |  607 |             0.054 |              0.043 |            0.128 |             0.087 |       0.054 |
| ('estimate band', '>=100 M')  |  258 |             0.05  |              0.031 |            0.198 |             0.155 |       0.05  |

## 5.3 SHAP: mean |contribution| to log price, by source feature

| feature        |   mean_abs_shap_log |   mean_abs_shap_log_ordinal_twin |   rank |   rank_ordinal_twin |
|:---------------|--------------------:|---------------------------------:|-------:|--------------------:|
| area           |              0.287  |                           0.283  |      1 |                   1 |
| location       |              0.1569 |                           0.157  |      2 |                   2 |
| ceiling_height |              0.0528 |                           0.0546 |      3 |                   3 |
| year_built     |              0.0525 |                           0.05   |      4 |                   4 |
| rooms          |              0.0266 |                           0.0265 |      5 |                   5 |
| condition      |              0.0262 |                           0.0256 |      6 |                   6 |
| furnished      |              0.0241 |                           0.025  |      7 |                   7 |
| n_photos       |              0.0196 |                           0.0202 |      8 |                   8 |
| floor          |              0.0135 |                           0.014  |      9 |                   9 |
| total_floors   |              0.012  |                           0.0136 |     10 |                  10 |
| parking        |              0.01   |                           0.0091 |     11 |                  11 |
| has_complex    |              0.0083 |                           0.0089 |     12 |                  12 |
| building_type  |              0.0066 |                           0.0061 |     13 |                  14 |
| kitchen_area   |              0.0063 |                           0.007  |     14 |                  13 |
| is_urgent      |              0.0054 |                           0.0058 |     15 |                  15 |

## 5.4 Comparable ads: does the 'few comparables' warning mark larger errors?

| Unnamed: 0         |    n |   share |   base_mae_M |   base_mdape_pct |   median_rel_width90_pct |   over_or_under_flag_share |
|:-------------------|-----:|--------:|-------------:|-----------------:|-------------------------:|---------------------------:|
| few comparables    |  198 |   0.051 |       12.614 |           11.345 |                   50.778 |                      0.187 |
| enough comparables | 3654 |   0.949 |        6.291 |            8.766 |                   48.314 |                      0.088 |

## 6. One evaluation on the 963 saved test ads

| model                                                                                 |   cv_mae_M |   cv_mdape |   test_mae_M |   test_rmse_M |   test_mdape |   test_r2 |   test_within10 |   test_within20 |
|:--------------------------------------------------------------------------------------|-----------:|-----------:|-------------:|--------------:|-------------:|----------:|----------------:|----------------:|
| Endterm base                                                                          |     6.616  |      8.934 |       6.2909 |        13.895 |       9.1293 |    0.8549 |          0.5369 |          0.8432 |
| C network + base, 50/50 log-blend (geo+rawfields+photo+text_flags+text_emb+photo_emb) |     6.0922 |    nan     |       5.8013 |        14.003 |       7.8241 |    0.8526 |          0.5992 |          0.8837 |

## 7. November temporal hold-out (8 Nov 2026, listing level: no coordinates, text or photos)

|                    |    n |   mae_M |   mae_ci_low |   mae_ci_high |   mdape |   mdape_ci_low |   mdape_ci_high |    r2 |   within10 |   median_log_error |
|:-------------------|-----:|--------:|-------------:|--------------:|--------:|---------------:|----------------:|------:|-----------:|-------------------:|
| chosen             | 1507 |   7.719 |        7.047 |         8.461 |  10.507 |          9.754 |          11.264 | 0.78  |      0.486 |              0.007 |
| base               | 1507 |  12.161 |       11.374 |        13.11  |  18.672 |         17.735 |          19.784 | 0.639 |      0.293 |             -0.019 |
| chosen_on_oct_test |  963 |   7.596 |        6.749 |         8.608 |  10.157 |          9.304 |          10.899 | 0.789 |      0.497 |             -0.002 |

|        |   n |   mae_M |   mae_ci_low |   mae_ci_high |   mdape |   mdape_ci_low |   mdape_ci_high |    r2 |   within10 |   median_log_error |
|:-------|----:|--------:|-------------:|--------------:|--------:|---------------:|----------------:|------:|-----------:|-------------------:|
| Almaty | 754 |   8.336 |        7.267 |         9.565 |   9.858 |          8.954 |          11.015 | 0.782 |      0.505 |              0.002 |
| Astana | 753 |   7.103 |        6.361 |         7.92  |  10.956 |          9.959 |          11.899 | 0.759 |      0.467 |              0.01  |

## 8. Error of the final model by band, city and rooms (test ads)

| segment    | value                   |   n |    mae_M |   mae_ci_low |   mae_ci_high |    mdape |   median_log_error |   within10 |   base_mae_M |   base_mdape |   diff_mae_M |   diff_ci_low |   diff_ci_high |
|:-----------|:------------------------|----:|---------:|-------------:|--------------:|---------:|-------------------:|-----------:|-------------:|-------------:|-------------:|--------------:|---------------:|
| city       | Almaty                  | 463 |  6.64743 |      5.34448 |       8.40329 |  7.56921 |            0.00262 |    0.61123 |      7.01372 |      8.70152 |     -0.36629 |      -0.90235 |        0.22216 |
| city       | Astana                  | 500 |  5.01775 |      4.45423 |       5.62295 |  8.30316 |            0.01198 |    0.588   |      5.62157 |      9.42546 |     -0.60383 |      -0.97745 |       -0.25939 |
| price_band | 25-40M                  | 338 |  2.79028 |      2.51654 |       3.06286 |  6.66605 |            0.0261  |    0.66568 |      3.21003 |      7.9197  |     -0.41976 |      -0.62357 |       -0.22958 |
| price_band | 40-60M                  | 271 |  4.67531 |      4.24914 |       5.11744 |  7.82407 |           -0.00166 |    0.61624 |      5.00762 |      8.565   |     -0.33231 |      -0.71106 |        0.0142  |
| price_band | 60-100M                 | 178 |  8.62735 |      7.71751 |       9.60548 |  9.90428 |           -0.04191 |    0.51685 |      9.78759 |     11.551   |     -1.16024 |      -2.04985 |       -0.29961 |
| price_band | <25M                    | 121 |  2.09192 |      1.81143 |       2.39431 |  8.43281 |            0.06071 |    0.57851 |      2.28885 |      8.89952 |     -0.19692 |      -0.43007 |        0.03348 |
| price_band | >=100M                  |  55 | 28.8677  |     18.9919  |      41.8416  | 12.8591  |           -0.09413 |    0.41818 |     29.0353  |     13.5244  |     -0.16762 |      -4.49794 |        4.48361 |
| rooms      | 1                       | 200 |  2.42452 |      2.07486 |       2.79523 |  6.84399 |            0.01017 |    0.66    |      2.78281 |      8.89344 |     -0.35829 |      -0.54843 |       -0.1691  |
| rooms      | 2                       | 389 |  3.81616 |      3.42941 |       4.25482 |  7.36148 |            0.02187 |    0.62211 |      4.33162 |      8.05908 |     -0.51547 |      -0.83421 |       -0.22553 |
| rooms      | 3                       | 307 |  6.54804 |      5.85267 |       7.32001 |  8.16342 |           -0.00564 |    0.5798  |      7.38041 |      9.73301 |     -0.83237 |      -1.39891 |       -0.27868 |
| rooms      | 4+                      |  67 | 23.985   |     15.5537  |      34.4452  | 13.2932  |           -0.07337 |    0.37313 |     23.1461  |     13.3159  |      0.83891 |      -2.3697  |        4.43672 |
| photos     | no photos               |  45 |  7.34916 |      3.63629 |      13.8358  |  8.66459 |           -0.00983 |    0.64444 |      8.79563 |     10.131   |     -1.44647 |      -2.62839 |       -0.39611 |
| photos     | with photo embedding    | 918 |  5.72541 |      5.00349 |       6.59985 |  7.81593 |            0.006   |    0.59695 |      6.16812 |      9.09961 |     -0.44272 |      -0.78938 |       -0.10045 |
| text       | description >=100 chars | 914 |  5.93394 |      5.1701  |       6.892   |  8.06962 |            0.0056  |    0.5919  |      6.39947 |      9.27045 |     -0.46553 |      -0.80935 |       -0.1235  |
| text       | short/no description    |  49 |  3.32683 |      2.2748  |       4.58845 |  4.26733 |            0.00509 |    0.73469 |      4.26583 |      5.85356 |     -0.939   |      -1.86301 |       -0.13665 |
| duplicate  | in duplicate group      |  15 |  2.05133 |      1.25623 |       3.01559 |  5.42695 |            0.02104 |    0.73333 |      2.76088 |      4.82483 |     -0.70955 |      -1.86896 |        0.28507 |
| duplicate  | unique ad               | 948 |  5.86062 |      5.13766 |       6.7493  |  7.89789 |            0.00523 |    0.59705 |      6.34676 |      9.16235 |     -0.48614 |      -0.81048 |       -0.14893 |
| train_twin | has a twin in train     |  13 |  2.00202 |      1.10466 |       3.08183 |  3.77605 |           -0.00444 |    0.76923 |      2.85198 |      4.82483 |     -0.84996 |      -2.21117 |        0.28413 |
| train_twin | no train twin           | 950 |  5.85327 |      5.11881 |       6.76301 |  7.89789 |            0.00547 |    0.59684 |      6.33796 |      9.16235 |     -0.48469 |      -0.8102  |       -0.14979 |

## 8.1 The Endterm hypothesis, tested on the ads the Endterm named

| ads | n | median APE, Endterm base % | median APE, final % | final closer on |
|:--|--:|--:|--:|:--|
| the Endterm's 8 worst test ads (text-explained discounts, no comparables) | 8 | 55.0 | 44.2 | 7/8 |
| the Midterm's 8 worst test ads (mostly luxury) | 8 | 37.0 | 37.0 | 4/8 |
| the base's 50 worst test ads | 50 | 35.9 | 32.0 | 41/50 |
| all test ads | 963 | 9.13 | 7.82 | 534/963 (55 %) |

The Endterm's plan said the text would address its worst errors, ads that are cheap for a reason only the description gives. On exactly those eight ads the final model is closer on 7 of 8 and cuts the median error from 55 to 44 % (the "cash only" one-room flat: 20.8 → 18.5 M against 13.9 M asked; the 99 m² flat in rough state: 58.4 → 53.3 M against 34 M), without fixing any of them: a 30–50 % discount for a legal or construction status is more than a residual model learns from a few dozen such ads. The luxury ads of the Midterm's list do not improve. Confirmed in direction, modest in size; detail in `outputs/final_eval/endterm_hypothesis.csv` and `endterm_worst8_final.csv`.

## 9. The app

`app/gradio_app.py` (UI) on `app/predict.py` (inference API, no Gradio import). For a typed-in listing it builds the final model's inputs from the form: OSM geo features from the coordinates, regex text flags and the USER-base sentence embedding from the description, the CLIP ViT-B/32 embedding of up to three uploaded photos. Raw ad-page fields and the photo track's tabular features cannot come from a form and stay missing, as for training ads without them; the app says so in its notes and also shows the Endterm base estimate. Around the estimate: the trust track's conformal 80 % / 90 % intervals (absolute log-residual, Mondrian by estimate band, centred on the final estimate; calibrated on base OOF residuals, so conservative for the blend), an interval-based over/under-pricing verdict, TreeSHAP reasons (for the HGB half of the blend), five comparable training ads with a "few comparables" warning, text red flags with form-consistency checks, and the photo condition (CLIP probe) with a mismatch warning against the stated condition. Example card: `outputs/final/app_example.png`. Run: `python app/gradio_app.py` (http://127.0.0.1:7860).

## 10. Existing solutions

Pricing a flat from its characteristics is the classic *hedonic* problem (Rosen, 1974): the price as a sum of implicit prices of attributes, estimated by regression. Our Midterm linear model on log price is that model; krisha.kz's own hint on an ad page, the average price per m² in the district, is its one-attribute version and our "realtor rule" baseline (test MdAPE 14.3 %). Industrial estimators such as Zillow's Zestimate moved from hedonic models to ensembles of tree models and neural networks on millions of transactions, and the Kaggle *House Prices* competition settled the tabular question in favour of gradient boosting and stacking, where our Endterm landed with far less data (9.1 %). For photos, Poursaeed, Matera and Belongie (2018) showed that a CNN estimate of the luxury level of listing photos improves a price model, and Law, Paige and Russell (2019) added street-view and satellite images; our condition classifier and CLIP features are the small-data version of the idea, with the same finding: the photos carry price information, most of it already in the fields. For text, TF-IDF or transformer features of the description are commonly added as a residual correction, which is how we use them. Calibrated uncertainty is a more recent concern: conformal prediction (Vovk, Gammerman and Shafer, 2005; Romano, Patterson and Candès, 2019) gives intervals with guaranteed coverage without a distributional assumption and is the method behind our trust layer. Against this background the project is small (4,815 ads, two cities, asking prices) but complete: the same saved split from the Midterm on, one selection rule fixed before the experiments, a single scripted test evaluation, and an honest answer that the photos and the text help only in combination.

## 11. Feedback from the earlier stages

The Midterm came back with 82/100 and no written comments; we reviewed it against its rubric ourselves, and the Endterm's change log (Endterm notebook, Section 2) lists six fixes with before/after evidence: the sharpened question, the contact-data scrub, the dropped scraper artefact `has_detail`, ten wrong coordinates, model choice by CV only, ensembles for the expensive bands. Three observations of the Endterm's error analysis shaped the Final: the worst errors were explained by words no field carries (the text track and red flags, tested in 8.1 on the very ads the Endterm named), the test set had been looked at at every stage (the November hold-out and a scripted test run that refuses to be repeated), and ads with few comparable flats had the widest errors (the comparables warning and the conformal intervals in place of the fixed 20 % rule). The Endterm defence takes place in Week 10; its feedback and, if we receive one, the list of fixes from the pre-final defence will be added here and attached to the submission.

## 12. Why not deep reinforcement learning

Reinforcement learning learns a policy from rewards for actions taken in an environment. The data contain no actions and no outcomes: asking prices only, no sale price, no time to sale, one snapshot per ad (days on market are not shown). A pricing agent would need exactly that reward (sold or not, time to sale, final price) and many interactions with buyers; any reward written from our data would be the asking price itself, and RL would reduce to the regression we already have, with more variance. The right tool for "is this price fair, and how sure are we" is supervised regression with calibrated uncertainty. RL becomes relevant with longitudinal data (price cuts and removal dates of the same ad).

## 13. Limitations

- **Asking price ≠ sale price.** The model learns the asking market; "over-priced" means "asked above comparable asking prices".
- **Reused test set.** The 963 saved test ads were scored at the Endterm and again here (once); no Final choice looked at them, but they are no longer a perfectly fresh sample.
- **November hold-out at listing level only.** The November ad pages (coordinates, description, photos) were blocked by the site's bot protection, so only the search cards (17 of 24 fields) were scraped; the fused model could not be tested on them, only a reduced model of the same family.
- **Whole-model certification.** The final model passes the rule as a whole; no block passes it step by step.
- **The app sees less than the CV** (no raw fields, no photo-track tabular features from a form); intervals are calibrated on the base model.
- **One October snapshot**, seller-reported condition labels, OSM completeness varying by district, five folds as a coarse ruler.

## 14. Contributions and AI tools

**Rakhat Balgabekov**: the shared `krisha_ml` foundation and the reproduction of the Endterm, the leakage audit and the repaired folds, the photo track, the November scrape, the assembly of the notebook, the report and the slides from the outputs. **Aitbek Kaiyrbekuly**: the location track, the text track, the trust layer, the error analysis and the slides. Both: the selection rule, the readings and the conclusions, discussed and written together.

**AI tools.** We used an AI coding assistant (Claude) to help write and debug the code of `krisha_ml`, the scripts and the app, to review the Endterm against the rubric when planning the Final, to draft the slides and the report tables from our outputs, and to check the English of our texts. Every experiment was run and checked by us; the choice of methods, the selection rule, the one-time use of the test ads, the readings of the results and the conclusions are our own.
