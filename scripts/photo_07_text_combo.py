"""Photo track step 7 (hint for the integration stage): photo + text residual correction together.

Neither the CLIP-embedding Ridge residual (photo track) nor the TF-IDF Ridge residual (CONTRACT smoke
example) passes the 1-SD rule alone (about -0.18 / -0.19 M, both 5/5 folds). This checks whether one
Ridge on [TF-IDF(description) | scaled CLIP ad embedding] does. Same harness, same folds, training ads only.
The alpha is chosen inside each outer fold by an inner 5-fold CV (no outer-validation information).
Output: outputs/photos/price_cv_text_combo.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from krisha_ml import core, raw  # noqa: E402
from krisha_ml import photos as ph  # noqa: E402

CLIP = ph.ad_embedding_table("clip")
ALPHAS = (3.0, 10.0, 30.0)
PHOTO_WEIGHT = (0.0, 0.5, 1.0)     # scale of the photo block relative to text (0 = text only)


def design(df_tr, df_va):
    vec = TfidfVectorizer(preprocessor=raw.sanitize_text, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    T_tr = vec.fit_transform(df_tr["description"].fillna("")); T_va = vec.transform(df_va["description"].fillna(""))
    P_tr, P_va = ph.lookup(CLIP, df_tr["id"]), ph.lookup(CLIP, df_va["id"])
    P_tr, P_va = np.nan_to_num(P_tr), np.nan_to_num(P_va)          # no photos -> zero vector (unit-norm rows otherwise)
    return T_tr, T_va, P_tr, P_va


def make(weight=None, alpha=None):
    def corr(df_tr, resid_tr, df_va):
        grid = [(a, w) for a in ALPHAS for w in PHOTO_WEIGHT] if weight is None else [(alpha, weight)]
        if len(grid) > 1:   # inner CV on the training fold only
            score = {}
            for itr, iva in KFold(5, shuffle=True, random_state=core.SEED).split(df_tr):
                a_tr, a_va = df_tr.iloc[itr].reset_index(drop=True), df_tr.iloc[iva].reset_index(drop=True)
                T1, T2, P1, P2 = design(a_tr, a_va)
                for a, w in grid:
                    m = Ridge(alpha=a).fit(sp.hstack([T1, sp.csr_matrix(P1 * w)]).tocsr(), resid_tr[itr])
                    p = m.predict(sp.hstack([T2, sp.csr_matrix(P2 * w)]).tocsr())
                    score[(a, w)] = score.get((a, w), 0) + np.abs(resid_tr[iva] - p).sum()
            a, w = min(score, key=score.get)
        else:
            a, w = grid[0]
        T1, T2, P1, P2 = design(df_tr, df_va)
        m = Ridge(alpha=a).fit(sp.hstack([T1, sp.csr_matrix(P1 * w)]).tocsr(), resid_tr)
        CHOICES.append((a, w))
        return m.predict(sp.hstack([T2, sp.csr_matrix(P2 * w)]).tocsr())
    return corr


CHOICES = []


def main():
    base = core.base_cv()
    rows = []
    for name, c in [("residual: Ridge on TF-IDF text only (alpha 10)", make(0.0, 10.0)),
                    ("residual: Ridge on TF-IDF + CLIP embedding (alpha 10, weight 1)", make(1.0, 10.0)),
                    ("residual: Ridge on TF-IDF + CLIP, alpha/weight by inner CV", make())]:
        CHOICES.clear()
        r = core.cv_evaluate_residual(c, name)
        rows.append({**r.row(base), "chosen_alpha_weight": list(CHOICES)})
        print(json.dumps(rows[-1], default=float), flush=True)
    json.dump(rows, open(ph.OUT / "price_cv_text_combo.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
