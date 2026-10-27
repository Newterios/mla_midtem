# %% [markdown]
# # Apartment asking prices in Astana and Almaty: ensembles, market segments and a neural network
#
# **Course project, Endterm (Week 9)** · Machine Learning Algorithms · Group **SE-2424**
# **Team:** Rakhat Balgabekov, Aitbek Kaiyrbekuly
# **Direction:** 1. Apartments for sale · **Task:** regression · **Target:** asking price (KZT) · **Data:** 4,815 krisha.kz ads collected with our own scraper on 4 and 8 October 2026
#
# This notebook continues the Midterm notebook (`Midterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb`) on the **same cleaned data and the same saved test split**. It applies the Week 5–8 topics in the order promised in the Midterm plan: tree ensembles with hyperparameter tuning (Section 3), K-Means, DBSCAN and PCA (Section 4), a neural network in PyTorch (Section 5), then one comparison of every model on the fixed split (Section 6), the error analysis (Section 7) and the conclusions and the plan for the Final (Section 8). Every table and plot is followed by our reading of it, with our own numbers.
#
# ## 1. The question, sharpened after the Midterm
#
# The Midterm asked "how accurately can we predict the asking price". That is a generic question, and the Midterm feedback (82/100) made us state the question the way a user would ask it:
#
# > **Which apartments in Astana and Almaty are asked above or below what comparable apartments go for, and by how much?**
#
# A **buyer** opens an ad and wants a fair-price reference before spending an evening on a viewing: if the asking price is more than ~20 % above the estimate for comparable flats, the ad is worth questioning. A **seller** wants a defensible starting price. A **realtor** wants a quick reference for every new listing. All three need the same thing: a model of the *typical* asking price of an apartment with given characteristics, and an honest statement of how wide its error is, so that a deviation can be told from noise. The target is therefore still the asking price (regression), the main metric is still **MAE in KZT** (the error in the money the user thinks in), with **MdAPE** (median absolute percentage error) because our registration goal was a median error under 10 %. Section 7.4 turns the final model into the buyer's over-pricing flag and counts how many ads it flags.
#
# ## 2. What changed since the Midterm, and what did not
#
# | | Midterm | Endterm |
# |---|---|---|
# | Data | 4,815 cleaned ads | the same 4,815 ads; the free text was **checked for contact data** and 7 social-media handles masked (2.1) |
# | Test split | 963 ads, seed 42, saved in `data/clean/split_ids.csv` | **unchanged**; the file is read, not recreated |
# | Features | 25 columns incl. `has_detail` | 24 columns: `has_detail` **removed** (2.2) |
# | Model choice | by test MAE (a slip) | by **5-fold CV MAE only**; the test set is used once, in Section 6 |
# | Models | dummy, district rule, linear, Ridge, Lasso, KNN, tree | + random forest, gradient boosting, tuned by randomised search, a blend, PCA + KNN, K-Means segment feature, MLP in PyTorch |
#
# ### 2.1 Data ethics check: contact data in the free text
# The Midterm data card said that no seller name or phone number is stored. That was true for the structured fields, but the free-text `description` written by sellers had never been checked. `data_collection/scrub_personal_data.py` now scans the raw snapshot and the cleaned file for phone numbers, e-mail addresses and messenger or social-media handles and masks them with `[contact removed]`. Result: krisha.kz itself strips phone numbers from the description text, so **no phone number** was found; **7 ads** carried a realtor's social-media handle, which is now masked. The structured fields, the target and the split are untouched, so every Midterm number stays valid.

# %%
import json, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
SEED = 42
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
OUT = Path("endterm"); OUT.mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

clean = pd.read_csv("data/clean/krisha_apartments_clean.csv")
split = pd.read_csv("data/clean/split_ids.csv")
PHONE = re.compile(r"(?<![\d.,])(?:\+?7|8)?[\s\-\(]*7\d{2}[\s\-\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?![\d.,])")
desc = clean["description"].fillna("").astype(str)
print(f"rows: {len(clean):,}   descriptions with a masked contact: {int(desc.str.contains('[contact removed]', regex=False).sum())}"
      f"   descriptions that still match a phone pattern: {int(desc.map(lambda s: bool(PHONE.search(s))).sum())}"
      f"   that still contain an @handle: {int(desc.str.contains(r'(?<!\w)@\w{3,}', regex=True).sum())}")
ex = desc[desc.str.contains("[contact removed]", regex=False)].iloc[0]; i = ex.find("[contact removed]")
print("example of a masked description: ...", ex[max(0, i - 70):i + 40].replace("\n", " "), "...")

# %% [markdown]
# ### 2.2 Feature list and the fixed split
# `has_detail` (whether the ad page or only the search card was opened by our scraper) is a property of our data collection, not of the apartment; the Midterm excluded `is_hot` for exactly that reason and should have excluded `has_detail` too. It is dropped. Everything else is what a seller fills in when the ad is created, so it is available at prediction time. `price_m2` is derived from the target and is **never** a feature; it appears only in Section 4 to describe clusters *after* they are formed.
#
# **One more data-quality step, found while preparing the segment map of Section 4.1.** A few ads carry map coordinates that lie outside their own city (the seller pinned the map in another town, or the point is a placeholder). On a map they sit hundreds of kilometres away, and for K-Means and for the models they are wrong values, not missing ones. We set the coordinates of such ads to missing when they fall outside a generous box around the city (Astana: 50.8–51.4° N, 71.0–71.9° E; Almaty: 43.0–43.5° N, 76.6–77.3° E); the pipeline then imputes them with the training median and marks them with the missing indicator, exactly like the ads that had no coordinates to begin with. The rows stay in the data.
#
# The split is read from the file saved at the Midterm. The 5-fold cross-validation used for every choice below is defined once, on the training rows only.

# %%
BOX = {"Astana": (50.8, 51.4, 71.0, 71.9), "Almaty": (43.0, 43.5, 76.6, 77.3)}
outside = pd.Series(False, index=clean.index)
for city, (la0, la1, lo0, lo1) in BOX.items():
    m = clean["city"] == city
    outside |= m & clean["lat"].notna() & ~(clean["lat"].between(la0, la1) & clean["lon"].between(lo0, lo1))
print("ads with coordinates outside their city:", clean.loc[outside, "city"].value_counts().to_dict(), "-> lat/lon set to missing")
display(clean.loc[outside, ["id", "city", "district", "lat", "lon", "rooms", "area", "price"]].head(8))
clean.loc[outside, ["lat", "lon"]] = np.nan
print(f"ads without coordinates now: {int(clean['lat'].isna().sum())} of {len(clean):,}")

# %%
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
TARGET = "price"
NUM = ["area", "log_area", "rooms", "kitchen_area", "floor", "total_floors", "year_built", "ceiling_height", "n_photos", "lat", "lon"]
CAT = ["city", "district", "building_type", "condition", "bathroom", "furnished", "balcony", "parking", "former_dorm", "seller_type"]
BOOL = ["has_complex", "mortgaged", "is_urgent"]
FEATURES = NUM + CAT + BOOL
assert TARGET not in FEATURES and "price_m2" not in FEATURES and "has_detail" not in FEATURES

X = clean[FEATURES].copy()
for c in BOOL: X[c] = X[c].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
for c in CAT:  X[c] = X[c].astype(object).where(X[c].notna(), "unknown").astype(str)
for c in NUM:  X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
y = clean[TARGET].to_numpy(dtype="float64")

test_ids = set(split.loc[split["split"] == "test", "id"])
test_mask = clean["id"].isin(test_ids).to_numpy()
X_train, X_test, y_train, y_test = X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]
assert len(X_test) == len(test_ids) == 963 and not (set(X_train.index) & set(X_test.index))
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"features: {len(FEATURES)}  |  train {len(X_train):,} rows, test {len(X_test):,} rows (ids from data/clean/split_ids.csv)")

# %%
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, make_scorer

def mdape(y_true, y_pred): return float(np.median(np.abs(y_pred - y_true) / y_true) * 100)
SCORING = {"MAE": "neg_mean_absolute_error", "MdAPE": make_scorer(mdape, greater_is_better=False), "R2": "r2"}

def make_pre(scaled):
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))] + ([("sc", StandardScaler())] if scaled else [])
    return ColumnTransformer([("num", Pipeline(num_steps), NUM + BOOL),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), CAT)])

def logged(model, scaled):
    """preprocessing + model on log(price); predictions come back in KZT"""
    return TransformedTargetRegressor(regressor=Pipeline([("pre", make_pre(scaled)), ("model", model)]),
                                      func=np.log, inverse_func=np.exp, check_inverse=False)

registry = {}     # name -> (estimator, cv summary row)
def cv_eval(name, est, n_jobs=-1, train_score=False, register=True, verbose=True):
    t0 = time.time()
    r = cross_validate(est, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=n_jobs, return_train_score=train_score)
    row = {"CV MAE (M KZT)": -r["test_MAE"].mean() / 1e6, "CV MAE SD": r["test_MAE"].std() / 1e6,
           "CV MdAPE %": -r["test_MdAPE"].mean(), "CV R²": r["test_R2"].mean(), "fit time s": r["fit_time"].mean()}
    if train_score: row["train MAE (M KZT)"] = -r["train_MAE"].mean() / 1e6
    row["folds MAE"] = np.round(-r["test_MAE"] / 1e6, 2)
    if register: registry[name] = (est, row)
    if verbose: print(f"{name:42s} CV MAE {row['CV MAE (M KZT)']:.2f} ± {row['CV MAE SD']:.2f} M   MdAPE {row['CV MdAPE %']:.1f} %   R² {row['CV R²']:.3f}   ({time.time()-t0:.0f} s)")
    return row

def table(names=None):
    names = names or list(registry)
    t = pd.DataFrame({n: registry[n][1] for n in names}).T
    t["CV MAE (M KZT)"] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(t["CV MAE (M KZT)"], t["CV MAE SD"])]
    t = t.drop(columns=[c for c in ["CV MAE SD", "folds MAE", "train MAE (M KZT)"] if c in t.columns])
    return t.astype({"CV MdAPE %": float, "CV R²": float, "fit time s": float}).round(3)

# %% [markdown]
# ### 2.3 The Midterm models on the same folds
# The Midterm baselines and the best Midterm model are re-run here so that every number in this notebook comes from one notebook, one split and one set of folds. (The Midterm notebook shuffled the training rows before its folds, so its CV values differ in the second decimal; the test rows are identical.)

# %%
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

class DistrictMedianPerM2(BaseEstimator, RegressorMixin):
    """the realtor's rule: median price per m² of the same city+district on the training rows × area"""
    def fit(self, X, y):
        t = X[["city", "district", "area"]].copy(); t["pm2"] = y / t["area"]
        self.med_ = t.groupby(["city", "district"])["pm2"].median().to_dict()
        self.city_med_ = t.groupby("city")["pm2"].median().to_dict(); self.global_med_ = float(t["pm2"].median()); return self
    def predict(self, X):
        pm2 = [self.med_.get((c, d), self.city_med_.get(c, self.global_med_)) for c, d in zip(X["city"], X["district"])]
        return np.asarray(pm2) * X["area"].values

KNN_NUM = ["log_area", "area", "rooms", "floor", "total_floors", "year_built", "lat", "lon"]
pre_knn = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), KNN_NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), ["city", "district"])])
midterm_models = {
    "Dummy (median)": Pipeline([("pre", make_pre(True)), ("model", DummyRegressor(strategy="median"))]),
    "District median price/m² (realtor rule)": DistrictMedianPerM2(),
    "Midterm linear (log price)": logged(LinearRegression(), scaled=True),
    "Midterm KNN size+location (k=10)": TransformedTargetRegressor(regressor=Pipeline([("pre", pre_knn), ("model", KNeighborsRegressor(10, weights="distance"))]), func=np.log, inverse_func=np.exp, check_inverse=False),
    "Midterm decision tree (depth 9)": logged(DecisionTreeRegressor(max_depth=9, min_samples_leaf=5, random_state=SEED), scaled=False),
}
for n, m in midterm_models.items(): cv_eval(n, m)
display(table())

# %% [markdown]
# # **Reading the table.** The picture is the Midterm's: the realtor's rule (district median price per m² × area) already halves the dummy's error (10.45 against 20.81 M KZT), the linear model on log price is the best Week 3–4 model at **9.10 ± 0.61 M KZT, MdAPE 12.1 %**, and KNN and the single tree stay behind it (10.18 and 10.19 M). The fold SD of the linear model, 0.61 M, is the yardstick for everything that follows: a model has to beat 9.10 M by clearly more than that to count as an improvement. The fold SDs of 0.4–0.75 M for all five models come from the folds themselves: the per-fold table in Section 3.1 shows that the same fold is the hardest for every tree model, because it holds more of the expensive ads.

# %% [markdown]
# ## 3. Ensembles and hyperparameter tuning (Week 5)
#
# **Why ensembles.** The Midterm found that the data are visibly non-linear (a U-shaped year effect, area × district interactions, a price band three times wide at a given area) and that a single tree already had the best R² but a staircase fit and the largest fold-to-fold spread. Ensembles are the Week 5 answer to exactly that: a **random forest** averages many deep trees and removes their variance; **gradient boosting** adds many shallow trees one after another, each fitted to the residuals of the previous ones, and removes the bias of a shallow tree. Both learn interactions without feature engineering and need no scaling.
#
# ### 3.1 One tree, a forest, boosting: bias and variance
# Four tree models with default settings on the same five folds, with the training MAE next to the validation MAE. The gap between the two measures how much a model memorises; the fold SD measures how much its validation error depends on which rows it saw.

# %%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
bv_models = {
    "Decision tree, fully grown": logged(DecisionTreeRegressor(random_state=SEED), scaled=False),
    "Decision tree, depth 9 (Midterm)": midterm_models["Midterm decision tree (depth 9)"],
    "Random forest, 300 trees (default)": logged(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1), scaled=False),
    "Gradient boosting (default: 100 trees, depth 3)": logged(GradientBoostingRegressor(random_state=SEED), scaled=False),
    "Hist. gradient boosting (default)": logged(HistGradientBoostingRegressor(random_state=SEED), scaled=False),
}
bv_rows = {n: cv_eval(n, m, train_score=True, register=(n != "Decision tree, depth 9 (Midterm)")) for n, m in bv_models.items()}
bv = pd.DataFrame(bv_rows).T
bv["gap train→val"] = bv["CV MAE (M KZT)"] - bv["train MAE (M KZT)"]
display(bv[["train MAE (M KZT)", "CV MAE (M KZT)", "gap train→val", "CV MAE SD", "CV MdAPE %", "fit time s"]].astype(float).round(2))
print("validation MAE of every fold (M KZT):"); display(pd.DataFrame({n: r["folds MAE"] for n, r in bv_rows.items()}, index=[f"fold {i+1}" for i in range(5)]).T)

# %% [markdown]
# # **Bias and variance, in our numbers.** The fully grown tree has a training MAE of **0.00** M and a validation MAE of **11.25** M, a gap of 11.25 M and the largest fold SD (0.73, folds from 9.89 to 12.06): it memorises every training ad and generalises worst of all, which is variance. Limiting it to depth 9 (the Midterm choice) raises the training MAE to 6.69 M and cuts the gap to 3.50 M and the SD to 0.50, but the validation MAE is still 10.19 M: a shallow tree has bias, it can only draw a staircase with 2⁹ steps over 24 fields.
#
# The **random forest** keeps the deep trees (training MAE 2.98 M, so it still memorises) but averages 300 of them: the validation MAE drops to **8.00 M** and the SD to **0.37**, half the single tree's. Averaging cancels the variance of each tree and keeps their low bias; that is what bagging is for. **Gradient boosting** does the opposite: each of its 100 trees of depth 3 is a weak, biased learner (the training MAE is 6.52 M, the smallest gap of all, 1.34 M), but adding them one after another on the residuals removes the bias, and the sum reaches **7.87 M** with the smallest SD (0.33). Histogram boosting with its default of 31 leaves and 100 iterations gets to **7.01 ± 0.38 M**, 2.1 M below the linear model, three to four times the linear model's fold SD, so the gain is not noise. Both ensembles fix the single tree's problems, the forest its variance and boosting its bias, and histogram boosting's worst fold (7.65 M) is still better than the linear model's mean (9.10 M).

# %% [markdown]
# ### 3.2 Learning rate and number of trees
# For gradient boosting the two settings that matter most are tied together: a smaller learning rate needs more trees. We refit `GradientBoostingRegressor(max_depth=3)` on each training fold once with 600 trees and read the validation MAE after 50, 100, 200, 300, 450 and 600 trees (`staged_predict`), for learning rates 0.3, 0.1 and 0.03.

# %%
LRS, STAGES = [0.3, 0.1, 0.03], [50, 100, 200, 300, 450, 600]
curve = {lr: {s: [] for s in STAGES} for lr in LRS}
t0 = time.time()
for tr, va in cv.split(X_train):
    pre = make_pre(False).fit(X_train.iloc[tr]); Xtr, Xva = pre.transform(X_train.iloc[tr]), pre.transform(X_train.iloc[va])
    for lr in LRS:
        g = GradientBoostingRegressor(n_estimators=600, learning_rate=lr, max_depth=3, random_state=SEED).fit(Xtr, np.log(y_train[tr]))
        for i, p in enumerate(g.staged_predict(Xva), 1):
            if i in STAGES: curve[lr][i].append(mean_absolute_error(y_train[va], np.exp(p)) / 1e6)
lr_table = pd.DataFrame({lr: {s: np.mean(v) for s, v in d.items()} for lr, d in curve.items()}).T
lr_table.index.name = "learning rate"; lr_table.columns.name = "trees"
display(lr_table.round(2)); print(f"({time.time()-t0:.0f} s)")
fig, ax = plt.subplots(figsize=(7, 3.6))
for lr in LRS: ax.plot(STAGES, lr_table.loc[lr], marker="o", label=f"learning rate {lr}")
ax.set_xlabel("number of trees"); ax.set_ylabel("5-fold CV MAE, M KZT"); ax.set_title("Gradient boosting (depth 3): learning rate × number of trees"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "e01_boosting_lr_trees.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# # **Learning rate × trees.** The smaller the learning rate, the more trees are needed: with 0.3 the curve flattens at **6.99 M after 450 trees** and creeps back up to 7.00 at 600, with 0.1 it is still improving at 600 trees (**6.74 M**), and with 0.03 the model is plainly underfitting even at 600 trees (7.44 M, and 11.76 M after 50). Each tree moves the prediction by learning rate × its own fit, so a rate of 0.03 needs roughly ten times the trees of 0.3 to travel the same distance. Too many trees hurt only at the largest rate, and only a little, because with 3,852 rows and trees of depth 3 there is still signal left in the residuals after 600 steps; the lecture's warning about fitting noise in the residuals would show up later on this curve. The whole table from 300 trees on lies between 6.74 and 7.44 M, and the fold SD is about 0.35, so 0.1 × 600 and 0.3 × 450 are not distinguishable; we would pick 0.1 with at least 600 trees, which is exactly what the randomised search below settles on.

