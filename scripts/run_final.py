"""Integration & evaluation of the Final: fuse the feature tracks, choose by CV, test ONCE, November drift.

Stages (default: all except test, in this order; the ONE test evaluation runs only when named):
  compare   CV of every fusion strategy (A residual stacking, B blocks as HGB features, C multimodal network),
            each block alone and all blocks together, under the random (Endterm) and the group-aware folds
  select    forward selection of blocks for A and B with the team 1-SD rule (primary folds = group if present)
  ablate    network blend: tabular-only control, embeddings only, leave-one-block-out (primary folds)
  decide    the model choice from the CV tables (lowest CV among candidates that pass the rule; else the base)
  fit       the chosen model refitted on all 3,852 training ads -> models/final_model.joblib + metadata json
  test      the ONE evaluation on the 963 test ads: bootstrap CIs (2,000 paired resamples), error by segment
  nov       November hold-out: reduced-feature model (fields the Nov cards carry) on Oct test and Nov ads
  figures   model comparison, ablation waterfall, residuals, segments, drift
  export    copy the results into the layout the notebook, the report and the slides read (outputs/fusion/, outputs/final_eval/)

Outputs: outputs/final/*.csv|json|png (quick mode: outputs/final_quick/), logs on stdout.
Usage:
  python scripts/run_final.py                       # everything but the test (~30-40 min; run in the background)
  python scripts/run_final.py test figures export   # the ONE test evaluation, after every choice is fixed
  python scripts/run_final.py compare select        # CV part only
  python scripts/run_final.py --quick               # tiny subsets, few trees: a smoke test (~1 min)
  python scripts/run_final.py test --allow-retest   # re-run the test stage (recorded in test_metrics.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # tracks share an oversubscribed CPU: OpenMP spin-waits make 2+ threads ~100x slower; parallelism comes from --jobs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from krisha_ml import core, fusion  # noqa: E402
from krisha_ml.fusion import STRATEGY_LABEL, Settings, cv_row, write_json  # noqa: E402

# chart palette (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "figure.dpi": 130})


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- run context
class Run:
    def __init__(self, args):
        self.args = args
        self.quick = args.quick or args.dry_run
        self.settings = Settings.quick() if self.quick else Settings()
        if args.jobs is not None:
            self.settings.n_jobs = args.jobs
        if args.nn_device:
            self.settings.nn_device = args.nn_device
        self.out = Path(args.out) if args.out else core.OUTPUT_DIR / ("final_quick" if self.quick else "final")
        self.models_dir = self.out / "models" if self.quick else fusion.MODELS_DIR
        self.cache_dir = self.out / "cache" if self.quick else core.CACHE_DIR
        self.out.mkdir(parents=True, exist_ok=True)
        self._train = None

    # data ------------------------------------------------------------------
    def setup(self):
        if self._train is not None:
            return
        train = core.load_train()
        rows = np.arange(len(train))
        if self.quick:
            rows = np.sort(np.random.default_rng(core.SEED).choice(len(train), self.args.quick_n, replace=False))
        self.train_df = train.df.iloc[rows].reset_index(drop=True)
        self.y = train.y[rows]
        self.ids = train.ids[rows]
        self.blocks, msgs = fusion.load_blocks(self.args.blocks or None)
        for m in msgs:
            log(m)
        self.frame = fusion.attach_blocks(self.train_df, self.blocks)
        self.schemes, msgs = fusion.fold_schemes(self.ids, full_train=not self.quick)
        for m in msgs:
            log(m)
        self.primary = "group" if "group" in self.schemes else "random"
        self.scheme_order = [self.primary] + [s for s in self.schemes if s != self.primary]
        self.stack = {s: fusion.StackingCV(self.frame, self.y, sch, self.settings, self.cache_dir) for s, sch in self.schemes.items()}
        self.feat = {s: fusion.FeatureCV(self.frame, self.y, sch, self.settings) for s, sch in self.schemes.items()}
        self.base = {s: self.stack[s].base_result() for s in self.schemes}
        for s in self.scheme_order:
            log(f"base CV ({s} folds): {self.base[s]}")
        write_json(self.out / "blocks.json", {"blocks": [b.summary() for b in self.blocks.values()], "messages": msgs,
                                              "primary_folds": self.primary, "quick": self.quick, "n_train_rows": len(self.y)})
        self._train = True

    def block_list(self, names):
        return [self.blocks[n] for n in names]


# --------------------------------------------------------------------------- stage: compare
def stage_compare(run: Run):
    run.setup()
    rows = []
    path = run.out / "cv_comparison.csv"
    names = list(run.blocks)
    nn_blocks = [n for n in names if n != "text"]
    for s in run.scheme_order:
        base = run.base[s]
        rows.append(cv_row(base, base, folds=s, strategy="base", blocks="", kind="base"))
        for strat, ev in (("A", lambda bl: run.stack[s].evaluate(bl)[0]), ("B", run.feat[s].evaluate)):
            for n in names:
                t0 = time.time()
                r = ev([run.blocks[n]])
                rows.append(cv_row(r, base, folds=s, strategy=strat, blocks=n, kind="single", seconds=round(time.time() - t0, 1)))
                log(f"[{s}] {r}  delta {r.mean - base.mean:+.4f}")
            t0 = time.time()
            r = ev(run.block_list(names))
            rows.append(cv_row(r, base, folds=s, strategy=strat, blocks="+".join(names), kind="all", seconds=round(time.time() - t0, 1)))
            log(f"[{s}] {r}  delta {r.mean - base.mean:+.4f}")
            pd.DataFrame(rows).to_csv(path, index=False)
        sch = run.schemes[s]
        c_all = None
        for label, bl in (("C: network, tabular only", []), ("C: network, all blocks", run.block_list(nn_blocks))):
            t0 = time.time()
            r = fusion.cv_model(label, "C", bl, run.frame, run.y, sch, run.settings)
            rows.append(cv_row(r, base, folds=s, strategy="C", blocks="+".join(b.name for b in bl), kind="all" if bl else "tabular",
                               seconds=round(time.time() - t0, 1)))
            log(f"[{s}] {r}  delta {r.mean - base.mean:+.4f}  ({time.time() - t0:.0f} s)")
            c_all = r if bl else c_all
        if c_all is not None:   # fixed 50/50 geometric blend with the base OOF (weight not tuned)
            oof = np.exp(0.5 * np.log(c_all.oof_pred) + 0.5 * np.log(base.oof_pred))
            r = fusion.result_from_oof("C: network, all blocks, 50/50 log-blend with base", run.y, oof, sch.folds)
            rows.append(cv_row(r, base, folds=s, strategy="C", blocks="+".join(nn_blocks), kind="blend"))
            log(f"[{s}] {r}  delta {r.mean - base.mean:+.4f}")
        pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    write_json(run.out / "cv_comparison.json", df.to_dict(orient="records"))
    log(f"wrote {path} ({len(df)} rows)")


# --------------------------------------------------------------------------- stage: select
def stage_select(run: Run):
    run.setup()
    s = run.primary
    base = run.base[s]
    candidates = [n for n, b in run.blocks.items() if b.keep is not False]
    vetoed = [n for n, b in run.blocks.items() if b.keep is False]
    if vetoed:
        log(f"blocks vetoed by their track (measured, never selected): {vetoed}")
    log(f"forward selection on {s} folds, base {base.mean:.4f} +- {base.sd:.4f}, rule margin {fusion.rule_margin(base):.4f} "
        f"(threshold {base.mean - fusion.rule_margin(base):.4f}), candidates {candidates}")
    sel, path_rows = {}, []
    evaluators = {"A": lambda names: run.stack[s].evaluate(run.block_list(names))[0],
                  "B": lambda names: run.feat[s].evaluate(run.block_list(names))}
    for strat, ev in evaluators.items():
        memo = {}

        def cached(names, ev=ev, memo=memo):
            key = tuple(sorted(names))
            if key not in memo:
                memo[key] = ev(list(names))
            return memo[key]

        log(f"strategy {strat}")
        res = fusion.forward_select(cached, candidates, base, log=log, min_margin=fusion.RULE_MARGIN_M)
        sel[strat] = res
        for p in res["path"]:
            r = p.pop("_result")
            p.update(strategy=strat, folds=s, folds_mae_M=" / ".join(f"{v:.3f}" for v in r.fold_mae))
            path_rows.append(p)
    pd.DataFrame(path_rows).to_csv(run.out / "forward_selection.csv", index=False)

    # kept / greedy sets on every fold scheme
    rows = []
    for strat in ("A", "B"):
        for which in ("kept", "greedy"):
            names = sel[strat][which]
            for sc in run.scheme_order:
                r = (run.stack[sc].evaluate(run.block_list(names))[0] if strat == "A" else run.feat[sc].evaluate(run.block_list(names)))
                rows.append(cv_row(r, run.base[sc], folds=sc, strategy=strat, set=which, blocks="+".join(names)))
    sets = pd.DataFrame(rows)
    sets.to_csv(run.out / "selected_sets.csv", index=False)

    write_json(run.out / "forward_sets.json", {"primary_folds": s, "vetoed_blocks": vetoed,
                                               **{k: {"kept": v["kept"], "greedy": v["greedy"]} for k, v in sel.items()}})


# --------------------------------------------------------------------------- stage: ablate
def stage_ablate(run: Run):
    """Where does the network blend's gain come from? On the primary folds: blend with the tabular-only
    network (pure ensembling control), embeddings only, and leave-one-block-out of the full blend."""
    run.setup()
    s = run.primary
    sch, base = run.schemes[s], run.base[s]
    nn_blocks = [n for n in run.blocks if n != "text"]
    variants = [("tabular only (ensembling control)", [])]
    emb = [n for n in nn_blocks if run.blocks[n].kind == "emb"]
    if emb and len(emb) < len(nn_blocks):
        variants.append(("embeddings only: " + "+".join(emb), emb))
    variants += [(f"all minus {n}", [m for m in nn_blocks if m != n]) for n in nn_blocks]
    variants.append(("all blocks", nn_blocks))
    rows = []
    for label, names in variants:
        t0 = time.time()
        r = fusion.cv_model(f"C: {label}", "C", run.block_list(names), run.frame, run.y, sch, run.settings)
        oof = np.exp(0.5 * np.log(r.oof_pred) + 0.5 * np.log(base.oof_pred))
        rb = fusion.result_from_oof(f"Cb: {label}", run.y, oof, sch.folds)
        for res, kind in ((r, "network"), (rb, "blend")):
            rows.append(cv_row(res, base, folds=s, variant_kind=kind, blocks="+".join(names), seconds=round(time.time() - t0, 1)))
        log(f"[{s}] {rb}  delta {rb.mean - base.mean:+.4f} (network alone {r.mean:.3f}, {time.time() - t0:.0f} s)")
        pd.DataFrame(rows).to_csv(run.out / "cv_network_ablation.csv", index=False)


# --------------------------------------------------------------------------- stage: decide
def stage_decide(run: Run):
    """The model choice from the CV tables only (no refitting): cheap to re-run with another --rule."""
    need = [run.out / f for f in ("cv_comparison.csv", "selected_sets.csv", "forward_sets.json")]
    old = run.out / "selection.json"
    if not need[2].exists() and old.exists():   # written by an older select stage: same content inside selection.json
        o = json.loads(old.read_text())
        write_json(need[2], {"primary_folds": o["primary_folds"], "vetoed_blocks": o.get("vetoed_blocks", []), **o["forward_selection"]})
    if not all(p.exists() for p in need):
        raise SystemExit(f"decide needs {[p.name for p in need]}: run compare and select first")
    cmp, sets = pd.read_csv(need[0]), pd.read_csv(need[1])
    fsets = json.loads(need[2].read_text())
    s = fsets["primary_folds"]
    b = cmp[(cmp.strategy == "base") & (cmp.folds == s)].iloc[0]
    base_mean, base_sd = float(b.cv_mae_M), float(b.cv_mae_sd_M)
    margin = max(fusion.RULE_MARGIN_M, base_sd)
    cands = []
    for strat in ("A", "B"):
        r = sets[(sets.strategy == strat) & (sets.set == "kept") & (sets.folds == s)]
        if fsets[strat]["kept"] and len(r):
            cands.append({"strategy": strat, "blocks": fsets[strat]["kept"], "cv_mae_M": float(r.iloc[0].cv_mae_M),
                          "reason": "forward selection, every added block passed the 1-SD rule"})
    for kind, strat, why in (("all", "C", "network with all its blocks passes the 1-SD rule as a whole"),
                             ("blend", "Cb", "fixed 50/50 log-blend of the network with the base passes the 1-SD rule as a whole")):
        r = cmp[(cmp.strategy == "C") & (cmp.kind == kind) & (cmp.folds == s)]
        if len(r) and bool(r.iloc[0].passes_1sd_rule):
            cands.append({"strategy": strat, "blocks": str(r.iloc[0].blocks).split("+"), "cv_mae_M": float(r.iloc[0].cv_mae_M), "reason": why})
    abl_path = run.out / "cv_network_ablation.csv"
    if abl_path.exists():   # same step rule for the network blend: blocks must beat the tabular-only blend by > its SD
        abl = pd.read_csv(abl_path)
        abl = abl[(abl.folds == s) & (abl.variant_kind == "blend")]
        tab, full = abl[abl.blocks.isna()], abl[abl.variant.str.endswith(": all blocks")]
        if len(tab) and len(full):
            tab, full = tab.iloc[0], full.iloc[0].copy()
            cur = cmp[(cmp.strategy == "C") & (cmp.kind == "blend") & (cmp.folds == s)]
            if len(cur):   # the ablation table may be older (preliminary): the blend's own CV comes from compare
                full["cv_mae_M"], full["passes_1sd_rule"] = float(cur.iloc[0].cv_mae_M), bool(cur.iloc[0].passes_1sd_rule)
                full["blocks"] = cur.iloc[0].blocks
            cands = [c for c in cands if c["strategy"] != "Cb"]
            if bool(tab.passes_1sd_rule):
                if (tab.cv_mae_M - full.cv_mae_M) > max(tab.cv_mae_sd_M, fusion.RULE_MARGIN_M):
                    cands.append({"strategy": "Cb", "blocks": str(full.blocks).split("+"), "cv_mae_M": float(full.cv_mae_M),
                                  "reason": "network blend passes vs base, and its blocks beat the tabular-only blend by > 1 SD"})
                else:
                    cands.append({"strategy": "Cb", "blocks": [], "cv_mae_M": float(tab.cv_mae_M),
                                  "reason": "tabular-only network blend passes vs base; the blocks on top of it do not pass the 1-SD step rule"})
            elif bool(full.passes_1sd_rule):
                cands.append({"strategy": "Cb", "blocks": str(full.blocks).split("+"), "cv_mae_M": float(full.cv_mae_M),
                              "reason": "network blend with its blocks passes vs base; the tabular-only blend does not"})
    if run.args.rule == "set":   # alternative reading: the greedy set is kept if it passes the rule vs base as a whole
        for strat in ("A", "B"):
            g = sets[(sets.strategy == strat) & (sets.set == "greedy") & (sets.folds == s)]
            if fsets[strat]["greedy"] and len(g) and bool(g.iloc[0].passes_1sd_rule):
                cands.append({"strategy": strat, "blocks": fsets[strat]["greedy"], "cv_mae_M": float(g.iloc[0].cv_mae_M),
                              "reason": "--rule set: greedy forward-selected set passes the 1-SD rule vs base as a whole"})
    if cands:
        choice = min(cands, key=lambda c: c["cv_mae_M"])
    else:
        choice = {"strategy": "base", "blocks": [], "cv_mae_M": base_mean,
                  "reason": "nothing passed the 1-SD rule (forward selection A/B, network C, network blend): the submitted Endterm model stays"}
    # the same candidate on the other fold scheme, for the report
    other = {}
    for sc in sorted(set(cmp.folds) - {s}):
        if choice["strategy"] in ("A", "B"):
            r = sets[(sets.strategy == choice["strategy"]) & (sets.folds == sc) & (sets.blocks.fillna("") == "+".join(choice["blocks"]))]
        elif choice["strategy"] == "Cb" and not choice["blocks"]:
            r = cmp.iloc[0:0]
        elif choice["strategy"] in ("C", "Cb"):
            r = cmp[(cmp.strategy == "C") & (cmp.kind == ("all" if choice["strategy"] == "C" else "blend")) & (cmp.folds == sc)]
        else:
            r = cmp[(cmp.strategy == "base") & (cmp.folds == sc)]
        if len(r):
            other[sc] = {k: r.iloc[0][k] for k in ("cv_mae_M", "cv_mae_sd_M", "delta_vs_base_M", "passes_1sd_rule")}
    fs = pd.read_csv(run.out / "forward_selection.csv") if (run.out / "forward_selection.csv").stat().st_size > 1 else pd.DataFrame()
    greedy_best = (fs.sort_values("cv_mae_M").iloc[0].strategy if len(fs) else "A")
    blend_row = cmp[(cmp.strategy == "C") & (cmp.kind == "blend") & (cmp.folds == s)]
    whole_model = None
    if len(blend_row):
        br = blend_row.iloc[0]
        whole_model = {"variant": br.variant, "cv_mae_M": float(br.cv_mae_M), "cv_mae_sd_M": float(br.cv_mae_sd_M),
                       "gain_vs_base_M": round(base_mean - float(br.cv_mae_M), 4), "margin_M": round(margin, 4),
                       "wins_by_more_than_1sd": bool(base_mean - float(br.cv_mae_M) > margin)}
    decision = {"primary_folds": s, "base_cv_mae_M": base_mean, "base_cv_sd_M": base_sd, "rule_margin_M": round(margin, 4),
                "rule_threshold_M": round(base_mean - margin, 4), "whole_model_comparison": whole_model,
                "rule": run.args.rule, "candidates": cands, "choice": choice, "choice_other_folds": other,
                "forward_selection": {k: fsets[k] for k in ("A", "B")}, "best_greedy_strategy": greedy_best,
                "vetoed_blocks": fsets.get("vetoed_blocks", []),
                "note": "greedy = steps that lower CV MAE at all (waterfall); kept = steps that pass the team rule; "
                        "among passing candidates the lowest primary-fold CV MAE wins"}
    write_json(run.out / "selection.json", decision)
    log(f"decision: {choice}")


# --------------------------------------------------------------------------- stage: fit
def _decision(run: Run) -> dict:
    p = run.out / "selection.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run the select stage first")
    return json.loads(p.read_text())


def stage_fit(run: Run):
    run.setup()
    dec = _decision(run)
    choice = dec["choice"]
    blocks = [run.blocks[n] for n in choice["blocks"] if n in run.blocks]
    missing = [n for n in choice["blocks"] if n not in run.blocks]
    if missing:
        raise SystemExit(f"chosen blocks no longer available: {missing}")
    groups = run.schemes[run.primary].groups
    t0 = time.time()
    model = fusion.make_model(choice["strategy"], blocks, run.settings).fit(run.frame, run.y, groups)
    log(f"fitted {STRATEGY_LABEL[choice['strategy']]} on {len(run.y)} training ads with blocks {choice['blocks']} in {time.time() - t0:.0f} s")
    run.models_dir.mkdir(parents=True, exist_ok=True)
    path = run.models_dir / "final_model.joblib"
    joblib.dump(model, path)
    cmp = run.out / "cv_comparison.csv"
    meta = fusion.model_metadata(model, blocks, run.settings, {
        "model_file": str(path.relative_to(core.ROOT)) if core.ROOT in path.parents else str(path),
        "selection": dec, "n_train_rows": len(run.y), "train_ids_sha256": hashlib.sha256(run.ids.tobytes()).hexdigest()[:16],
        "cv_comparison_file": str(cmp.name), "quick": run.quick})
    write_json(run.models_dir / "final_model_meta.json", meta)
    log(f"saved {path} + final_model_meta.json")


# --------------------------------------------------------------------------- stage: test (ONCE)
def stage_test(run: Run):
    run.setup()
    out_json = run.out / "test_metrics.json"
    if out_json.exists() and not run.args.allow_retest:
        raise SystemExit(f"{out_json} exists: the test set is evaluated once. Pass --allow-retest to override (it is recorded).")
    model = joblib.load(run.models_dir / "final_model.joblib")
    meta = json.loads((run.models_dir / "final_model_meta.json").read_text())
    names = [b["block"] for b in meta["blocks"]]
    for b in meta["blocks"]:
        if b["block"] in run.blocks and run.blocks[b["block"]].digest() != b["digest"]:
            log(f"WARNING: block {b['block']} changed since the model was fitted (digest {b['digest']} -> {run.blocks[b['block']].digest()})")
    train, test = core.load_train_test()
    tdf = test.df
    if run.quick:
        tdf = tdf.sample(min(run.args.quick_n // 2, len(tdf)), random_state=core.SEED).reset_index(drop=True)
    frame = fusion.attach_blocks(tdf, run.blocks)
    y = tdf[core.TARGET].to_numpy(float)
    pred = model.predict(frame)
    base_model = fusion.BaseModel(settings=run.settings).fit(run.frame, run.y)
    base_pred = base_model.predict(frame)
    log(f"test ({len(y)} ads): final MAE {np.abs(pred - y).mean() / 1e6:.4f} M, Endterm base {np.abs(base_pred - y).mean() / 1e6:.4f} M")
    boot = fusion.bootstrap_metrics(y, pred, base_pred, n_boot=run.settings.n_boot)
    seg = fusion.test_segments(frame, run.blocks)
    segt = fusion.segment_table(seg, y, pred, base_pred, n_boot=run.settings.n_boot)
    segt.round(5).to_csv(run.out / "test_segments.csv", index=False)
    pd.DataFrame({"id": tdf[core.ID], "price": y, "pred_final": pred, "pred_base": base_pred,
                  "log_error_final": np.log(pred / y), **{f"seg_{c}": seg[c] for c in seg.columns}}).to_csv(run.out / "test_predictions.csv", index=False)
    prev = json.loads(out_json.read_text()) if out_json.exists() else None
    write_json(out_json, {"model": meta["strategy_label"] or meta["strategy"], "blocks": names, "n_test": len(y),
                          "metrics": boot, "endterm_reference_test_mae_M": core.ENDTERM_REFERENCE["test_mae_M"],
                          "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "quick": run.quick,
                          "retest": prev is not None, "previous_evaluation": None if prev is None else prev.get("evaluated_at")})
    m = boot["mae_M"]
    log(f"MAE {m['value']:.4f} M [{m['ci_low']:.3f}, {m['ci_high']:.3f}], vs base {m['diff']:+.4f} [{m['diff_ci_low']:+.3f}, {m['diff_ci_high']:+.3f}]")


# --------------------------------------------------------------------------- stage: nov
def stage_nov(run: Run):
    if not fusion.NOV_CSV.exists():
        log(f"[skip] November hold-out: {fusion.NOV_CSV} missing"); return
    nov = fusion.load_nov()
    train, test = core.load_train_test()
    tr_df, te_df = train.df, test.df
    if run.quick:
        tr_df = tr_df.sample(run.args.quick_n, random_state=core.SEED).reset_index(drop=True)
        te_df = te_df.sample(run.args.quick_n // 2, random_state=core.SEED).reset_index(drop=True)
        nov = nov.sample(min(len(nov), run.args.quick_n // 2), random_state=core.SEED).reset_index(drop=True)
    num, cat, cov = fusion.reduced_features(nov)
    X_tr_all = core.encode_features(tr_df)
    cov["oct_coverage"] = [float((X_tr_all[c] != "unknown").mean()) if c in core.CAT else float(X_tr_all[c].notna().mean()) for c in cov.feature]
    cov.round(4).to_csv(run.out / "nov_feature_coverage.csv", index=False)
    log(f"reduced features ({len(num) + len(cat)}): {num + cat}; dropped: {list(cov.loc[~cov.kept, 'feature'])}")
    y_tr, y_te, y_nov = (d[core.TARGET].to_numpy(float) for d in (tr_df, te_df, nov))
    X_tr, X_te, X_nov = core.encode_features(tr_df), core.encode_features(te_df), core.encode_features(nov)
    reduced = fusion.make_reduced_model(num, cat, run.settings).fit(X_tr[num + cat], y_tr)
    full = core.make_base_model(**run.settings.hgb_overrides).fit(X_tr, y_tr)
    # CV cost of dropping the fields (October train rows, Endterm folds)
    if not run.quick:
        red_cv = core.cv_evaluate("reduced features (Nov fields)", model_factory=lambda n, c: fusion.make_reduced_model(num, cat, run.settings),
                                  n_jobs=run.settings.n_jobs)
        cv_info = red_cv.row(core.base_cv())
    else:
        cv_info = None
    rows = []
    sets = {"Oct test (963)": (X_te, y_te, te_df), "Nov hold-out": (X_nov, y_nov, nov)}
    for model_name, mdl, cols in (("reduced (Nov fields)", reduced, num + cat), ("full Endterm (24, missing as NaN)", full, None)):
        for set_name, (X, y, d) in sets.items():
            p = mdl.predict(X if cols is None else X[cols])
            subsets = {"all": np.ones(len(y), bool)}
            for c in sorted(d["city"].unique()):
                subsets[c] = (d["city"] == c).to_numpy()
            if "posted_after_october" in d:
                pa = d["posted_after_october"].astype(str).str.lower() == "true"
                subsets["posted after Oct"] = pa.to_numpy()
                subsets["listed since Oct"] = (~pa).to_numpy()
            for sub, mask in subsets.items():
                if mask.sum() < 5:
                    continue
                b = fusion.bootstrap_metrics(y[mask], p[mask], n_boot=run.settings.n_boot)
                rows.append({"model": model_name, "eval_set": set_name, "subset": sub, "n": int(mask.sum()),
                             "mae_M": b["mae_M"]["value"], "mae_ci_low": b["mae_M"]["ci_low"], "mae_ci_high": b["mae_M"]["ci_high"],
                             "mdape": b["mdape"]["value"], "mdape_ci_low": b["mdape"]["ci_low"], "mdape_ci_high": b["mdape"]["ci_high"],
                             "r2": b["r2"]["value"], "within10": b["within10"]["value"],
                             "median_log_error": float(np.median(np.log(p[mask] / y[mask]))),
                             "mean_log_error": float(np.mean(np.log(p[mask] / y[mask])))})
    res = pd.DataFrame(rows)
    res.round(5).to_csv(run.out / "nov_holdout.csv", index=False)
    drift = fusion.drift_table(np.r_[y_tr, y_te], pd.concat([tr_df, te_df], ignore_index=True), y_nov, nov)
    drift.round(3).to_csv(run.out / "nov_price_drift.csv", index=False)
    r = res.set_index(["model", "eval_set", "subset"])
    key = ("reduced (Nov fields)",)
    summary = {"n_nov": len(nov), "reduced_num": num, "reduced_cat": cat, "reduced_cv": cv_info,
               "reduced_oct_test_mae_M": float(r.loc[key + ("Oct test (963)", "all"), "mae_M"]),
               "reduced_nov_mae_M": float(r.loc[key + ("Nov hold-out", "all"), "mae_M"]),
               "reduced_oct_test_mdape": float(r.loc[key + ("Oct test (963)", "all"), "mdape"]),
               "reduced_nov_mdape": float(r.loc[key + ("Nov hold-out", "all"), "mdape"]),
               "reduced_nov_median_log_error": float(r.loc[key + ("Nov hold-out", "all"), "median_log_error"]),
               "note": "Nov ads are listing cards only (no coordinates, kitchen, photos count, balcony/parking/dorm, full text); "
                       "the reduced model is trained on October train rows with only the fields the cards carry, so the "
                       "Oct-test vs Nov gap is the temporal (and card-vs-page) drift, not a feature gap."}
    summary["mdape_gap_pp"] = summary["reduced_nov_mdape"] - summary["reduced_oct_test_mdape"]
    write_json(run.out / "nov_holdout.json", summary)
    log(f"Nov: reduced model Oct test MdAPE {summary['reduced_oct_test_mdape']:.2f} % -> Nov {summary['reduced_nov_mdape']:.2f} % "
        f"(MAE {summary['reduced_oct_test_mae_M']:.3f} -> {summary['reduced_nov_mae_M']:.3f} M); median log error {summary['reduced_nov_median_log_error']:+.4f}")


# --------------------------------------------------------------------------- stage: figures
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"figure {path.name}")


def fig_model_comparison(run: Run):
    p = run.out / "cv_comparison.csv"
    if not p.exists():
        return
    cmp = pd.read_csv(p)
    cmp["label"] = np.where(cmp.strategy == "base", "Endterm base",
                            np.where(cmp.strategy == "C", cmp.variant, cmp.strategy + ": " + np.where(cmp.kind == "all", "all blocks", cmp.blocks)))
    folds = [f for f in ("random", "group") if f in set(cmp.folds)]
    labels = list(dict.fromkeys(cmp.label))
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(labels) + 1.4))
    yy = np.arange(len(labels))[::-1]
    pos = dict(zip(labels, yy))
    for j, (f, color) in enumerate(zip(folds, (BLUE, ORANGE))):
        d = cmp[cmp.folds == f]
        off = (j - (len(folds) - 1) / 2) * 0.25
        ax.errorbar(d.cv_mae_M, [pos[l] + off for l in d.label], xerr=d.cv_mae_sd_M, fmt="o", ms=4, color=color,
                    elinewidth=1, capsize=0, label=f"{f} folds (mean ± fold SD)")
        b = d[d.strategy == "base"].iloc[0]
        ax.axvline(b.cv_mae_M, color=color, lw=0.8, ls=":")
        ax.axvline(b.rule_threshold_M, color=color, lw=0.8, ls="--")
    ax.set_yticks(yy, labels)
    ax.set_xlabel("5-fold CV MAE, M KZT (dotted: base, dashed: rule threshold = base − 0.354 M)")
    ax.set_title("Fusion strategies and blocks, training ads only", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    _save(fig, run.out / "fig_model_comparison.png")


def fig_waterfall(run: Run):
    p, sp = run.out / "forward_selection.csv", run.out / "selection.json"
    if not p.exists() or not sp.exists():
        return
    fs = pd.read_csv(p)
    dec = json.loads(sp.read_text())
    if fs.empty:
        log("waterfall: no block lowered the CV MAE at all; nothing to draw")
        return
    strat = dec["choice"]["strategy"] if dec["choice"]["strategy"] in ("A", "B") else dec["best_greedy_strategy"]
    d = fs[fs.strategy == strat].sort_values("step")
    base, thr = dec["base_cv_mae_M"], dec["rule_threshold_M"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    prev = base
    ax.bar(0, base, color=GREY, width=0.6)
    ax.text(0, base, f"{base:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    for i, r in enumerate(d.itertuples(), start=1):
        color = AQUA if r.passes_step_rule else ORANGE
        ax.bar(i, r.cv_mae_M - prev, bottom=prev, color=color, width=0.6)
        ax.text(i, min(prev, r.cv_mae_M), f"{r.cv_mae_M:.3f}\n(−{r.gain_M:.3f})", ha="center", va="top", fontsize=7, color=INK)
        prev = r.cv_mae_M
    ax.axhline(thr, color=INK, lw=0.8, ls="--")
    ax.text(len(d) + 0.4, thr, "1-SD rule", va="bottom", ha="right", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(d) + 1), ["base"] + [f"+{a}" for a in d.added])
    lo = min(d.cv_mae_M.min(), thr) - 0.15
    ax.set_ylim(lo, base + 0.1)
    ax.set_ylabel("CV MAE, M KZT")
    ax.set_title(f"Forward selection, strategy {strat} ({dec['primary_folds']} folds; preliminary run, before the final "
                 "photo features)\ngreen = step passes the 1-SD rule, orange = fails", loc="left", fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
    _save(fig, run.out / "fig_ablation_waterfall.png")


def fig_residuals(run: Run):
    p = run.out / "test_predictions.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    a1.scatter(d.price / 1e6, d.pred_final / 1e6, s=6, alpha=0.5, color=BLUE, edgecolors="none")
    lim = [d.price.min() / 1e6 * 0.8, d.price.max() / 1e6 * 1.2]
    a1.plot(lim, lim, color=INK, lw=0.8)
    a1.set(xscale="log", yscale="log", xlabel="asking price, M KZT", ylabel="predicted, M KZT")
    a1.set_title("Final model, test ads", loc="left", fontsize=10, color=INK)
    bands = ["<25M", "25-40M", "40-60M", "60-100M", ">=100M"]
    data_f = [np.log(d.pred_final / d.price)[d.seg_price_band == b] * 100 for b in bands]
    data_b = [np.log(d.pred_base / d.price)[d.seg_price_band == b] * 100 for b in bands]
    x = np.arange(len(bands))
    for data, off, color, lab in ((data_b, -0.18, GREY, "Endterm base"), (data_f, 0.18, BLUE, "final")):
        bp = a2.boxplot(data, positions=x + off, widths=0.3, patch_artist=True, showfliers=False, manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.7)
        for k in ("medians", "whiskers", "caps"):
            for line in bp[k]:
                line.set(color=INK if k == "medians" else color, lw=1)
        a2.plot([], [], color=color, lw=6, label=lab)
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_xticks(x, bands)
    a2.set(ylabel="log error x 100 (≈ % over/under)", xlabel="price band")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("Residuals by price band", loc="left", fontsize=10, color=INK)
    _save(fig, run.out / "fig_residuals.png")


def fig_segments(run: Run):
    p = run.out / "test_segments.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    s["label"] = s.segment + ": " + s.value + " (n=" + s.n.astype(str) + ")"
    fig, ax = plt.subplots(figsize=(7.5, 0.26 * len(s) + 1.2))
    yy = np.arange(len(s))[::-1]
    ax.errorbar(s.mae_M, yy + 0.15, xerr=[s.mae_M - s.mae_ci_low, s.mae_ci_high - s.mae_M], fmt="o", ms=4, color=BLUE,
                elinewidth=1, label="final (95 % bootstrap CI)")
    if "base_mae_M" in s:
        ax.plot(s.base_mae_M, yy - 0.15, "o", ms=4, color=GREY, label="Endterm base")
    ax.set_yticks(yy, s.label)
    ax.set_xlabel("test MAE, M KZT")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Test error by segment", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", visible=False)
    _save(fig, run.out / "fig_test_segments.png")


def fig_nov(run: Run):
    p = run.out / "nov_holdout.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.subset.isin(["all", "Almaty", "Astana", "posted after Oct", "listed since Oct"])]
    d["label"] = d.eval_set + " · " + d.subset
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.3 * d.label.nunique() + 1.5), sharey=True)
    labels = list(dict.fromkeys(d.label))
    yy = dict(zip(labels, np.arange(len(labels))[::-1]))
    for ax, metric, lo, hi, xl in ((axes[0], "mdape", "mdape_ci_low", "mdape_ci_high", "MdAPE, %"),
                                   (axes[1], "mae_M", "mae_ci_low", "mae_ci_high", "MAE, M KZT")):
        for (m, color, off) in (("reduced (Nov fields)", BLUE, 0.15), ("full Endterm (24, missing as NaN)", ORANGE, -0.15)):
            e = d[d.model == m]
            ax.errorbar(e[metric], [yy[l] + off for l in e.label], xerr=[e[metric] - e[lo], e[hi] - e[metric]], fmt="o", ms=4,
                        color=color, elinewidth=1, label=m)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(yy.values()), list(yy.keys()))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    axes[0].set_title("October model on November ads (temporal drift)", loc="left", fontsize=10, color=INK, pad=18)
    _save(fig, run.out / "fig_nov_drift.png")


def stage_figures(run: Run):
    for f in (fig_model_comparison, fig_waterfall, fig_residuals, fig_segments, fig_nov):
        try:
            f(run)
        except Exception as e:   # a figure must never hide the numbers
            log(f"figure {f.__name__} failed: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- stage: export
def _metric_block(b: dict) -> dict:
    """bootstrap_metrics output -> the flat metric dict the notebook, the report and the slides read."""
    out = {m: b[m]["value"] for m in ("mae_M", "rmse_M", "mdape", "r2", "within10", "within20") if m in b}
    for m, key in (("mae_M", "bootstrap95_mae"), ("mdape", "bootstrap95_mdape")):
        if m in b:
            out[key] = [b[m]["ci_low"], b[m]["ci_high"]]
    return out


def stage_export(run: Run):
    """Copy the integration results into the layout the notebook, the report and the slides read
    (scripts/final_build_common.py): outputs/fusion/ (CV choice) and outputs/final_eval/ (test + November).
    Pure re-formatting of files already written; whatever does not exist yet is skipped."""
    import shutil
    # quick mode keeps everything inside its own output directory
    fus, ev = (run.out / "fusion", run.out / "final_eval") if run.quick else (run.out.parent / "fusion", run.out.parent / "final_eval")
    fus.mkdir(parents=True, exist_ok=True); ev.mkdir(parents=True, exist_ok=True)
    cmp_p, sel_p = run.out / "cv_comparison.csv", run.out / "selection.json"
    cmp = pd.read_csv(cmp_p) if cmp_p.exists() else None
    if cmp is not None and sel_p.exists():
        dec = json.loads(sel_p.read_text())
        cmp["variant"] = cmp["variant"].replace({"Endterm base": "base (Endterm)"})
        keep = ["variant", "cv_mae_M", "cv_mae_sd_M", "cv_mdape", "cv_r2", "folds_mae_M", "delta_vs_base_M", "folds_better",
                "passes_1sd_rule", "blocks", "folds"]
        cmp[[c for c in keep if c in cmp.columns]].to_csv(fus / "selection.csv", index=False)
        ch = dec["choice"]
        other = dec.get("choice_other_folds", {})
        prim, sec = dec["primary_folds"], next(iter(other), None)
        cv = {prim: (ch["cv_mae_M"], None)}
        if sec:
            cv[sec] = (other[sec]["cv_mae_M"], other[sec]["cv_mae_sd_M"])
        if ch["strategy"] == "base":
            for f in cv:
                b = cmp[(cmp.strategy == "base") & (cmp.folds == f)]
                cv[f] = (float(b.cv_mae_M.iloc[0]), float(b.cv_mae_sd_M.iloc[0])) if len(b) else cv[f]
        else:
            r = cmp[(cmp.strategy == "C") & (cmp.kind == ("blend" if ch["strategy"] == "Cb" else "all")) & (cmp.folds == prim)] \
                if ch["strategy"] in ("C", "Cb") else pd.DataFrame()
            if len(r):
                cv[prim] = (cv[prim][0], float(r.cv_mae_sd_M.iloc[0]))
        rnd, grp = cv.get("random", (None, None)), cv.get("group", (None, None))
        write_json(fus / "selection.json", {
            "chosen": STRATEGY_LABEL.get(ch["strategy"], ch["strategy"]) + (f" ({'+'.join(ch['blocks'])})" if ch["blocks"] else ""),
            "strategy": ch["strategy"], "blocks": ch["blocks"], "cv_mae_M": rnd[0], "cv_mae_sd_M": rnd[1],
            "group_cv_mae_M": grp[0], "group_cv_sd_M": grp[1], "reason": ch["reason"], "primary_folds": prim,
            "note": "cv_mae_M = Endterm random folds (comparable with base.cv_mae_M); the choice was made on the "
                    f"{prim} folds; source outputs/final/selection.json"})
        if (run.out / "fig_model_comparison.png").exists():
            shutil.copyfile(run.out / "fig_model_comparison.png", fus / "cv_blocks.png")
        log(f"export: {fus.name}/selection.csv|json")
    tm_p, tp_p = run.out / "test_metrics.json", run.out / "test_predictions.csv"
    if tm_p.exists() and tp_p.exists():
        tm, tp = json.loads(tm_p.read_text()), pd.read_csv(tp_p)
        y = tp["price"].to_numpy(float)
        chosen = _metric_block(tm["metrics"])
        base = _metric_block(fusion.bootstrap_metrics(y, tp["pred_base"].to_numpy(float), n_boot=run.settings.n_boot))
        write_json(ev / "test_metrics.json", {"model": tm["model"], "chosen": chosen, "base": base, "n_test": tm["n_test"],
                                              "paired_diff_mae_M": [tm["metrics"]["mae_M"].get("diff_ci_low"), tm["metrics"]["mae_M"].get("diff_ci_high")],
                                              "evaluated_at": tm["evaluated_at"], "retest": tm.get("retest", False)})
        sel = json.loads((fus / "selection.json").read_text()) if (fus / "selection.json").exists() else {}
        bcv = cmp[(cmp.strategy == "base") & (cmp.folds == "random")] if cmp is not None else pd.DataFrame()
        rows = [("Endterm base", float(bcv.cv_mae_M.iloc[0]) if len(bcv) else None, float(bcv.cv_mdape.iloc[0]) if len(bcv) else None, base),
                (sel.get("chosen", tm["model"]), sel.get("cv_mae_M"), None, chosen)]
        pd.DataFrame([{"model": n, "cv_mae_M": c, "cv_mdape": md, **{f"test_{k}": v for k, v in m.items() if not k.startswith("bootstrap")}}
                      for n, c, md, m in rows]).round(4).to_csv(ev / "results_table.csv", index=False)
        tp["abs_error_M"] = (tp["pred_final"] - tp["price"]).abs() / 1e6
        tp["ape_pct"] = 100 * (tp["pred_final"] - tp["price"]).abs() / tp["price"]
        tp.sort_values("abs_error_M", ascending=False).head(20).round(4).to_csv(ev / "worst_cases.csv", index=False)
        log(f"export: {ev.name}/test_metrics.json, results_table.csv, worst_cases.csv")
    if (run.out / "test_segments.csv").exists():
        pd.read_csv(run.out / "test_segments.csv").to_csv(ev / "error_by_segment.csv", index=False)
    if (run.out / "fig_residuals.png").exists():
        shutil.copyfile(run.out / "fig_residuals.png", ev / "residuals.png")
    nj, nc = run.out / "nov_holdout.json", run.out / "nov_holdout.csv"
    if nj.exists() and nc.exists():
        s, d = json.loads(nj.read_text()), pd.read_csv(nc)
        cols = ["n", "mae_M", "mae_ci_low", "mae_ci_high", "mdape", "mdape_ci_low", "mdape_ci_high", "r2", "within10", "median_log_error"]
        pick = lambda m, e, sub: (lambda r: {c: r[c].iloc[0] for c in cols} if len(r) else None)(
            d[(d.model == m) & (d.eval_set == e) & (d.subset == sub)])
        red, full = "reduced (Nov fields)", "full Endterm (24, missing as NaN)"
        cov = pd.read_csv(run.out / "nov_feature_coverage.csv") if (run.out / "nov_feature_coverage.csv").exists() else None
        write_json(ev / "nov_holdout.json", {
            "n": s["n_nov"], "chosen": pick(red, "Nov hold-out", "all"), "base": pick(full, "Nov hold-out", "all"),
            "chosen_on_oct_test": pick(red, "Oct test (963)", "all"),
            "by_city": {c: pick(red, "Nov hold-out", c) for c in ("Almaty", "Astana") if pick(red, "Nov hold-out", c)},
            "features_missing": [] if cov is None else list(cov.loc[~cov.kept, "feature"]),
            "note": "'chosen' = the Endterm HGB retrained on the fields the November cards carry (the fused final model "
                    "needs photos/text/coordinates the cards lack); 'base' = the full 24-feature Endterm model with the "
                    "missing fields as NaN. Source: outputs/final/nov_holdout.*"})
        if (run.out / "fig_nov_drift.png").exists():
            shutil.copyfile(run.out / "fig_nov_drift.png", ev / "nov_holdout.png")
        log(f"export: {ev.name}/nov_holdout.json")


STAGES = {"compare": stage_compare, "select": stage_select, "ablate": stage_ablate, "decide": stage_decide, "fit": stage_fit, "test": stage_test, "nov": stage_nov,
          "figures": stage_figures, "export": stage_export}
# the one test evaluation never runs by accident: it must be named explicitly
DEFAULT_STAGES = [s for s in STAGES if s != "test"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", default=DEFAULT_STAGES,
                    help=f"any of {list(STAGES)} (default: all but 'test', which must be named explicitly)")
    ap.add_argument("--quick", action="store_true", help="tiny subsets, few trees/epochs; writes to outputs/final_quick/")
    ap.add_argument("--dry-run", action="store_true", help="alias of --quick (never touches models/ or outputs/final/)")
    ap.add_argument("--quick-n", type=int, default=360, help="training rows in quick mode")
    ap.add_argument("--blocks", nargs="*", help=f"restrict to these blocks (default all available: {fusion.BLOCK_ORDER})")
    ap.add_argument("--jobs", type=int, help="parallel folds (default 2; quick 1)")
    ap.add_argument("--nn-device", choices=["auto", "cpu", "mps"], help="device of the network (default auto)")
    ap.add_argument("--out", help="output directory (default outputs/final or outputs/final_quick)")
    ap.add_argument("--rule", choices=["step", "set"], default="step",
                    help="step (default, team rule): every added block must lower CV MAE by > 1 fold SD of the current set; "
                         "set: the greedy set is kept if it beats the base by > 1 base SD as a whole")
    ap.add_argument("--allow-retest", action="store_true", help="re-run the test stage although it was run (recorded)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bad = [s for s in args.stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {list(STAGES)}")
    run = Run(args)
    log(f"run_final: stages {args.stages}, quick={run.quick}, out={run.out}, settings n_jobs={run.settings.n_jobs}")
    for s in args.stages:
        t0 = time.time()
        log(f"== stage {s}")
        STAGES[s](run)
        log(f"== stage {s} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
