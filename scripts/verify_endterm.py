"""Verify that krisha_ml reproduces the submitted Endterm, and write the base CV numbers.

Run from the repo root:  python scripts/verify_endterm.py
Outputs (outputs/core/): base_cv.json, base_cv_folds.csv, base_oof.csv, endterm_repro_test.json.
The test check re-evaluates only the already-submitted Endterm model (allowed once); no choice is made on it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from krisha_ml import core  # noqa: E402

OUT = core.OUTPUT_DIR / "core"
OUT.mkdir(parents=True, exist_ok=True)
ref = core.ENDTERM_REFERENCE

t0 = time.time()
clean = core.load_clean()
_, fixed = core.fix_coordinates(pd.read_csv(core.CLEAN_CSV))
print(f"clean rows {len(clean)}, coordinates set to missing: {int(fixed.sum())}, ads without coordinates: {int(clean['lat'].isna().sum())}")

base = core.cv_evaluate("Endterm base (tuned HGB, 24 features)", n_jobs=5)
print(base, f"({time.time() - t0:.0f} s)")
print("folds MAE (M):", np.round(base.fold_mae, 3))
cv_ok = abs(base.mean - ref["cv_mae_M"]) < 1e-3 and abs(base.sd - ref["cv_mae_sd_M"]) < 1e-3

train = core.load_train()
row = base.row() | {"rule_threshold_M": round(base.sd, 4), "endterm_reference": ref, "matches_endterm": bool(cv_ok)}
(OUT / "base_cv.json").write_text(json.dumps(row, indent=2))
pd.DataFrame({"fold": range(1, 6), "mae_M": base.fold_mae, "mdape": base.fold_mdape, "r2": base.fold_r2}).to_csv(OUT / "base_cv_folds.csv", index=False)
fold_of = np.empty(len(train), dtype=int)
for k, (_, va) in enumerate(core.cv_folds()):
    fold_of[va] = k + 1
pd.DataFrame({"id": train.ids, "fold": fold_of, "price": train.y, "oof_pred": base.oof_pred,
              "log_resid": np.log(train.y) - np.log(base.oof_pred)}).to_csv(OUT / "base_oof.csv", index=False)

# one-time sanity reproduction of the submitted test numbers
train, test = core.load_train_test()
model = core.make_base_model().fit(train.X, train.y)
m = core.metrics(test.y, model.predict(test.X))
print("test (known submitted model):", {k: round(v, 4) for k, v in m.items()})
test_ok = abs(m["mae_M"] - ref["test_mae_M"]) < 1e-3 and abs(m["mdape"] - ref["test_mdape"]) < 1e-3
(OUT / "endterm_repro_test.json").write_text(json.dumps({"test": m, "reference": ref, "matches_endterm": bool(test_ok)}, indent=2))
print(f"CV reproduction {'OK' if cv_ok else 'MISMATCH'}, test reproduction {'OK' if test_ok else 'MISMATCH'}  ({time.time() - t0:.0f} s)")
sys.exit(0 if cv_ok and test_ok else 1)
