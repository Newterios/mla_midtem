"""Photo track step 6: where do photo features change the price error? (training ads, CV OOF only)

Splits the 3,852 training ads by stated condition (known / unknown) and photo availability and reports
MAE / MdAPE of the Endterm base and of every photo variant from outputs/photos/price_cv_oof.csv.
Output: outputs/photos/price_subgroups.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from krisha_ml import core  # noqa: E402
from krisha_ml import photos as ph  # noqa: E402


def main():
    train = core.load_train()
    oof = pd.read_csv(ph.OUT / "price_cv_oof.csv")
    assert (oof["id"].values == train.ids).all()
    base = core.base_cv().oof_pred
    stated = ph.condition_labels(train.df).values >= 0
    _, idx = ph.load_image_cache()
    has = np.isin(train.ids, idx["ad_id"].unique())
    groups = {"all": np.ones(len(train), bool), "condition stated": stated, "condition unknown": ~stated,
              "unknown + has photos": ~stated & has, "no photos": ~has}
    rows = []
    for variant, pred in [("Endterm base", base)] + [(c, oof[c].values) for c in oof.columns if c != "id"]:
        for g, m in groups.items():
            rows.append({"variant": variant, "group": g, "n": int(m.sum()),
                         "mae_M": np.abs(train.y[m] - pred[m]).mean() / 1e6, "mdape": core.mdape(train.y[m], pred[m])})
    t = pd.DataFrame(rows)
    b = t[t["variant"] == "Endterm base"].set_index("group")
    t["mae_delta_M"] = t["mae_M"] - t["group"].map(b["mae_M"])
    t["mdape_delta"] = t["mdape"] - t["group"].map(b["mdape"])
    t.round(4).to_csv(ph.OUT / "price_subgroups.csv", index=False)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
