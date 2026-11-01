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
# **Reading the table.** The picture is the Midterm's: the realtor's rule (district median price per m² × area) already halves the dummy's error (10.45 against 20.81 M KZT), the linear model on log price is the best Week 3–4 model at **9.10 ± 0.61 M KZT, MdAPE 12.1 %**, and KNN and the single tree stay behind it (10.18 and 10.19 M). The fold SD of the linear model, 0.61 M, is the yardstick for everything that follows: a model has to beat 9.10 M by clearly more than that to count as an improvement. The fold SDs of 0.4–0.75 M for all five models come from the folds themselves: the per-fold table in Section 3.1 shows that the same fold is the hardest for every tree model, because it holds more of the expensive ads.

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
# **Bias and variance, in our numbers.** The fully grown tree has a training MAE of **0.00** M and a validation MAE of **11.25** M, a gap of 11.25 M and the largest fold SD (0.73, folds from 9.89 to 12.06): it memorises every training ad and generalises worst of all, which is variance. Limiting it to depth 9 (the Midterm choice) raises the training MAE to 6.69 M and cuts the gap to 3.50 M and the SD to 0.50, but the validation MAE is still 10.19 M: a shallow tree has bias, it can only draw a staircase with 2⁹ steps over 24 fields.
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
# **Learning rate × trees.** The smaller the learning rate, the more trees are needed: with 0.3 the curve flattens at **6.99 M after 450 trees** and creeps back up to 7.00 at 600, with 0.1 it is still improving at 600 trees (**6.74 M**), and with 0.03 the model is plainly underfitting even at 600 trees (7.44 M, and 11.76 M after 50). Each tree moves the prediction by learning rate × its own fit, so a rate of 0.03 needs roughly ten times the trees of 0.3 to travel the same distance. Too many trees hurt only at the largest rate, and only a little, because with 3,852 rows and trees of depth 3 there is still signal left in the residuals after 600 steps; the lecture's warning about fitting noise in the residuals would show up later on this curve. The whole table from 300 trees on lies between 6.74 and 7.44 M, and the fold SD is about 0.35, so 0.1 × 600 and 0.3 × 450 are not distinguishable; we would pick 0.1 with at least 600 trees, which is exactly what the randomised search below settles on.

# %% [markdown]
# ### 3.3 Randomised search for the forest and for histogram boosting
# Both ensembles are tuned with `RandomizedSearchCV` on the **same five folds**, scored by MAE in KZT (MdAPE and R² are recorded as well). Every candidate is a full pipeline, so imputation and encoding are refitted inside each fold. The forest search draws 25 settings, the boosting search 40; each search also records the training-fold score so that we can see whether a setting memorises.

# %%
from sklearn.model_selection import RandomizedSearchCV
def search(name, est, space, n_iter):
    t0 = time.time()
    s = RandomizedSearchCV(est, space, n_iter=n_iter, cv=cv, scoring=SCORING, refit="MAE", random_state=SEED, n_jobs=-1, return_train_score=True)
    s.fit(X_train, y_train)
    res = pd.DataFrame(s.cv_results_)
    res = res[[c for c in res.columns if c.startswith("param_")] + ["mean_train_MAE", "mean_test_MAE", "std_test_MAE", "mean_test_MdAPE", "mean_test_R2", "rank_test_MAE"]].copy()
    res.columns = [c.replace("param_regressor__model__", "") for c in res.columns]
    res["mean_train_MAE"] = -res["mean_train_MAE"] / 1e6; res["mean_test_MAE"] = -res["mean_test_MAE"] / 1e6
    res["std_test_MAE"] = res["std_test_MAE"] / 1e6; res["mean_test_MdAPE"] = -res["mean_test_MdAPE"]
    res = res.sort_values("rank_test_MAE"); res.insert(0, "model", name); res.to_csv(OUT / f"search_{name.split()[0].lower()}.csv", index=False)
    b = s.best_index_
    row = {"CV MAE (M KZT)": -s.cv_results_["mean_test_MAE"][b] / 1e6, "CV MAE SD": s.cv_results_["std_test_MAE"][b] / 1e6,
           "CV MdAPE %": -s.cv_results_["mean_test_MdAPE"][b], "CV R²": s.cv_results_["mean_test_R2"][b], "fit time s": s.cv_results_["mean_fit_time"][b],
           "folds MAE": np.round(-np.array([s.cv_results_[f"split{i}_test_MAE"][b] for i in range(5)]) / 1e6, 2)}
    registry[name] = (s.best_estimator_, row)
    print(f"{name}: {n_iter} candidates × 5 folds in {time.time()-t0:.0f} s -> best CV MAE {row['CV MAE (M KZT)']:.2f} ± {row['CV MAE SD']:.2f} M, MdAPE {row['CV MdAPE %']:.1f} %")
    print("   best settings:", {k.replace("regressor__model__", ""): v for k, v in s.best_params_.items()})
    return s, res

rf_space = {"regressor__model__n_estimators": [200, 400], "regressor__model__max_depth": [None, 8, 12, 16, 24],
            "regressor__model__min_samples_leaf": [1, 2, 4, 8], "regressor__model__max_features": ["sqrt", 0.3, 0.5, 0.75, 1.0]}
hgb_space = {"regressor__model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12, 0.2], "regressor__model__max_iter": [150, 250, 400, 600, 800],
             "regressor__model__max_leaf_nodes": [7, 15, 31, 63], "regressor__model__min_samples_leaf": [5, 10, 20, 35, 50],
             "regressor__model__l2_regularization": [0.0, 0.1, 1.0, 5.0], "regressor__model__max_depth": [None, 6]}
rf_search, rf_res = search("Random forest (tuned)", logged(RandomForestRegressor(random_state=SEED, n_jobs=1), scaled=False), rf_space, 25)
hgb_search, hgb_res = search("Gradient boosting (tuned)", logged(HistGradientBoostingRegressor(random_state=SEED, early_stopping=False), scaled=False), hgb_space, 40)

# %% [markdown]
# **XGBoost and LightGBM.** The lecture's third ensemble is XGBoost, and the syllabus names XGBoost and LightGBM as the course's tools, so both are tuned the same way (20 random settings each, same folds, same metric). They implement the same idea as scikit-learn's histogram boosting with three additions: an L2 penalty on the leaf values inside the objective, second-order gradients, and row/column subsampling of the kind the forest uses.

# %%
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
xgb_space = {"regressor__model__n_estimators": [300, 600, 900], "regressor__model__learning_rate": [0.03, 0.05, 0.08, 0.12], "regressor__model__max_depth": [3, 4, 6, 8],
             "regressor__model__min_child_weight": [1, 5, 10], "regressor__model__subsample": [0.7, 0.85, 1.0], "regressor__model__colsample_bytree": [0.6, 0.8, 1.0], "regressor__model__reg_lambda": [0.0, 1.0, 5.0]}
lgb_space = {"regressor__model__n_estimators": [300, 600, 900], "regressor__model__learning_rate": [0.03, 0.05, 0.08, 0.12], "regressor__model__num_leaves": [7, 15, 31, 63],
             "regressor__model__min_child_samples": [5, 10, 20, 35], "regressor__model__subsample": [0.7, 0.85, 1.0], "regressor__model__subsample_freq": [1], "regressor__model__colsample_bytree": [0.6, 0.8, 1.0], "regressor__model__reg_lambda": [0.0, 1.0, 5.0]}
xgb_search, xgb_res = search("XGBoost (tuned)", logged(XGBRegressor(random_state=SEED, n_jobs=1, tree_method="hist"), scaled=False), xgb_space, 20)
lgb_search, lgb_res = search("LightGBM (tuned)", logged(LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1), scaled=False), lgb_space, 20)

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for a, (res, title) in zip(ax, [(rf_res, "Random forest: 25 settings"), (hgb_res, "Hist. gradient boosting: 40 settings")]):
    r = res.sort_values("mean_test_MAE").reset_index(drop=True)
    a.errorbar(range(len(r)), r["mean_test_MAE"], yerr=r["std_test_MAE"], fmt="o", ms=3, capsize=2, label="validation MAE ± fold SD")
    a.plot(range(len(r)), r["mean_train_MAE"], "s", ms=3, color="C3", alpha=.6, label="training MAE")
    a.axhline(bv.loc["Random forest, 300 trees (default)" if "forest" in title else "Hist. gradient boosting (default)", "CV MAE (M KZT)"], color="grey", ls="--", lw=1, label="default setting")
    a.set_xlabel("candidate, sorted by validation MAE"); a.set_ylabel("MAE, M KZT"); a.set_title(title); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "e02_random_search.png", bbox_inches="tight"); plt.show()
print("five best forest settings:"); display(rf_res.head(5).drop(columns=["model", "rank_test_MAE"]).round(3))
print("five best boosting settings:"); display(hgb_res.head(5).drop(columns=["model", "rank_test_MAE"]).round(3))

# %% [markdown]
# **What the searches found.** *Forest:* the 25 settings run from 7.99 to 9.9 M, and the best (200 trees, depth 16, one row per leaf, all features at each split) scores **7.99 ± 0.38 M**, against **8.00 M** for the default 300 trees. Tuning bought 0.01 M, a fortieth of the fold SD: the forest is insensitive to its settings on these data, and the only thing that hurts it is starving the splits (`max_features="sqrt"` with 8 rows per leaf gives the worst candidates). The training MAE of 3 M next to a validation MAE of 8 M shows that each forest memorises, and that it does not matter, which is bagging again. *Boosting:* the 40 settings run from 6.62 to 8.1 M. The best (learning rate 0.12, 7 leaves, 600 iterations, 5 rows per leaf, no L2 penalty) scores **6.62 ± 0.35 M**, against 7.01 M for the default, a gain of 0.39 M, about one fold SD, and the five best settings lie within 0.11 M of each other with quite different shapes (7 to 31 leaves, learning rates from 0.03 to 0.12, 600 to 800 iterations). What they share is **many iterations and small leaves**: the opposite of the bank data in Practice 7B, where the best boosting was slow and heavily regularised, because here the signal is strong (R² ≈ 0.85) and lives in continuous inputs, so the model can afford to fit fine structure. The training MAE of the best boosting setting is 3.4 M, half its validation MAE, which means that even the chosen model memorises part of the training ads; the L2 penalty did nothing to stop that (the best setting has none), and the validation score did not punish it. We keep the tuned boosting as the leading candidate and note that its advantage over the default is one SD, not ten.

# %% [markdown]
# **Three implementations, one answer.** XGBoost's best setting (depth 4, 600 trees, learning rate 0.12, 5 rows per leaf, 70 % of the rows and 60 % of the columns per tree, L2 penalty 5) scores **6.64 ± 0.32 M**, LightGBM's best (31 leaves, 900 trees, learning rate 0.08, 20 rows per leaf, 70 % rows and 80 % columns, L2 penalty 5) **6.75 ± 0.28 M**, and scikit-learn's histogram boosting stays first at **6.62 ± 0.35 M**. The three differ by 0.13 M, a third of one fold SD, so the choice of library is not where the error comes from: all three fit the same additive model of shallow trees on the same histogram bins, and the subsampling and the L2 penalty that XGBoost and LightGBM add bring no measurable gain on 3,852 rows (the best XGBoost settings have the penalty, the best scikit-learn settings have none, and both land at the same place). The agreement is useful as a check: the 6.6 M floor of Section 3 is a property of the data and the fields, not of one implementation, which is why Section 3.4 asks what the fields do not contain. The training MAEs of the top settings (0.1–4.3 M against 6.6–6.8 M on validation) show the same memorising as before; LightGBM's best setting with 31 leaves and 900 trees fits its training ads to 1.0 M and generalises no worse than the slower ones, which is the "many small steps" result again. The tuned scikit-learn boosting remains the candidate for Section 6, with XGBoost and LightGBM in the final table as the two independent confirmations of its score.

# %% [markdown]
# ### 3.4 What do the ensembles rely on?
# Two views. The forest is refitted on all training rows with `oob_score=True`: every tree leaves out about a third of the rows, so each row can be predicted by the trees that never saw it, which gives a validation estimate without a split. Then impurity importance of that forest (summed over the one-hot columns of each original field) is shown next to **permutation importance** of the tuned boosting pipeline on the training rows (`n_repeats=10`): how much the MAE in KZT grows when one column is shuffled.

# %%
from sklearn.inspection import permutation_importance
rf_params = {k.replace("regressor__model__", ""): v for k, v in rf_search.best_params_.items()}
pre_tree = make_pre(False).fit(X_train)
rf_oob = RandomForestRegressor(**rf_params, oob_score=True, random_state=SEED, n_jobs=-1).fit(pre_tree.transform(X_train), np.log(y_train))
oob_pred = np.exp(rf_oob.oob_prediction_)
print(f"forest OOB MAE {mean_absolute_error(y_train, oob_pred)/1e6:.2f} M KZT, OOB MdAPE {mdape(y_train, oob_pred):.1f} %, OOB R² {r2_score(y_train, oob_pred):.3f}"
      f"   (5-fold CV of the same setting: MAE {registry['Random forest (tuned)'][1]['CV MAE (M KZT)']:.2f} M)")
names = pre_tree.get_feature_names_out()
def origin(n):
    n = n.split("__", 1)[1]
    if n.startswith("missingindicator_"): return n.replace("missingindicator_", "") + " (missing?)"
    for c in CAT:
        if n.startswith(c + "_"): return c
    return n
imp = pd.Series(rf_oob.feature_importances_, index=[origin(n) for n in names]).groupby(level=0).sum().sort_values(ascending=False)
t0 = time.time()
perm = permutation_importance(registry["Gradient boosting (tuned)"][0], X_train, y_train, scoring="neg_mean_absolute_error", n_repeats=10, random_state=SEED, n_jobs=-1)
perm_s = pd.Series(perm.importances_mean / 1e6, index=FEATURES).sort_values(ascending=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
imp.head(12).iloc[::-1].plot.barh(ax=ax[0], color="C0"); ax[0].set_title("Forest: impurity importance (summed per field)"); ax[0].set_xlabel("share of impurity decrease")
perm_s.head(12).iloc[::-1].plot.barh(ax=ax[1], color="C1"); ax[1].set_title("Boosting: permutation importance on training rows"); ax[1].set_xlabel("increase of MAE when shuffled, M KZT")
plt.tight_layout(); plt.savefig(FIG / "e03_importance.png", bbox_inches="tight"); plt.show()
print(f"({time.time()-t0:.0f} s)  permutation importance, top 8 (M KZT):", perm_s.head(8).round(2).to_dict())

# %% [markdown]
# **OOB and importance.** The forest's OOB MAE is **7.66 M** against 7.99 M for the 5-fold estimate of the same setting, OOB R² 0.80: the two estimates agree within one SD, and OOB is the cheaper one because it needs no refits (each tree already leaves out about a third of the rows). Both importance views put **size first and location second**: the forest's impurity importance gives 0.35 each to `area` and `log_area` (the same information in two columns, which the forest splits between them; together 0.70), then `lat` and `lon` (0.08 and 0.07), ceiling height (0.04, a proxy for building class) and the year (0.02). Permutation importance on the tuned boosting model says the same with a scale that means something: shuffling `area` raises the MAE by **17.5 M KZT**, shuffling `lon` or `lat` by 7.2 and 6.4 M, the year by 2.8 M, ceiling height by 2.1 M, and `district` only by 1.3 M. The coordinates beat the district labels by a factor of five: 13 district names are a coarse map, and the ensembles use the fine one, which is the structure the Midterm's KNN was trying to use and could not. The ranking agrees between the two methods and with the Midterm's linear coefficients (log area and the Almaty centre districts were the largest), and it tells us where the remaining error cannot come from: not from a missing size or location signal, but from what the fields do not contain.

# %% [markdown]
# ### 3.5 Extra trees and a blend of the two tuned ensembles
# Two cheap additions that the lecture mentions: **extremely randomised trees** (random split thresholds, which reduces variance further at the price of some bias) and a **blend** that averages the log-price predictions of the tuned forest and the tuned boosting model. A blend helps when the two models make *different* mistakes.

# %%
class GeoMeanBlend(BaseEstimator, RegressorMixin):
    """fits each estimator and returns the geometric mean of their KZT predictions (= mean on the log scale)"""
    def __init__(self, estimators): self.estimators = estimators
    def fit(self, X, y): self.fitted_ = [clone(e).fit(X, y) for e in self.estimators]; return self
    def predict(self, X): return np.exp(np.mean([np.log(e.predict(X)) for e in self.fitted_], axis=0))
cv_eval("Extra trees (500, leaf 2)", logged(ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.75, random_state=SEED, n_jobs=1), scaled=False))
cv_eval("Blend: tuned forest + tuned boosting", GeoMeanBlend([registry["Random forest (tuned)"][0], registry["Gradient boosting (tuned)"][0]]))
display(table(["Midterm linear (log price)", "Midterm decision tree (depth 9)", "Random forest, 300 trees (default)", "Random forest (tuned)",
               "Hist. gradient boosting (default)", "Gradient boosting (tuned)", "XGBoost (tuned)", "LightGBM (tuned)", "Extra trees (500, leaf 2)", "Blend: tuned forest + tuned boosting"]))

# %% [markdown]
# **Extra trees and the blend.** Extra trees score **8.69 ± 0.43 M**, worse than the forest (8.00): random split thresholds add bias, and this market has sharp thresholds (a building from 2018 onwards, the top floor, a district border) that a random cut misses. The blend of the tuned forest and the tuned boosting scores **6.95 ± 0.33 M**, between its two parents (6.62 and 7.99) and worse than boosting alone: averaging only helps when both models are about equally good and wrong on different ads, and here the forest is 1.4 M weaker and its errors are mostly the same ads (both models fail on the same luxury and text-only cases of Section 7). The tuned boosting therefore stays the single ensemble candidate for Section 6.

# %% [markdown]
# ## 4. Unsupervised analysis: market segments, unusual ads and PCA (Week 6)
#
# **Why.** The Midterm error grew with price and with size: one global model treated the luxury segment like the mass market, and we never defined what the "segments" of this market are. Clustering answers that without using the price: it groups apartments by what they *are* (size, age, height, location), and only afterwards we look at what they *cost*. DBSCAN finds the ads that sit in no dense group at all. PCA tells how many independent dimensions the structured fields really carry, which is the question behind the Midterm's weak KNN.
#
# ### 4.1 K-Means segments within each city
# **Inputs:** `log_area`, `rooms`, `floor`, `total_floors`, `year_built`, `lat`, `lon` (median-imputed, standardised; the target and `price_m2` are not inputs). Clustering is run **within each city**: with both cities together the first thing any distance sees is the 1,000 km between them, and the two clusters that come out are just "Astana" and "Almaty", which we already know. The number of segments is chosen per city with three measures on the training rows: inertia (elbow), silhouette, and **stability** (adjusted Rand index between the full fit and fits on ten random 80 % subsamples). We take the k with the best silhouette among the values that are stable (mean ARI ≥ 0.9, i.e. the partition barely moves when a fifth of the ads is dropped) and whose smallest cluster has at least 30 ads, so that a "segment" is never a handful of outliers.

# %%
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
SEG = ["log_area", "rooms", "floor", "total_floors", "year_built", "lat", "lon"]
train_df = clean.loc[~test_mask].copy(); test_df = clean.loc[test_mask].copy()
seg_prep, seg_model, seg_scores, seg_k = {}, {}, {}, {}
rng = np.random.default_rng(SEED)
for city in ["Astana", "Almaty"]:
    d = train_df[train_df["city"] == city]
    prep = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit(d[SEG])
    M = prep.transform(d[SEG]); rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(M)
        aris = []
        for _ in range(10):
            sub = rng.choice(len(M), size=int(0.8 * len(M)), replace=False)
            aris.append(adjusted_rand_score(km.labels_, KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(M[sub]).predict(M)))
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(M, km.labels_, sample_size=2000, random_state=SEED),
                     "stability (mean ARI)": np.mean(aris), "stability (min ARI)": np.min(aris), "smallest cluster": np.bincount(km.labels_).min()})
    sc = pd.DataFrame(rows).set_index("k"); seg_scores[city] = sc
    ok = sc[(sc["stability (mean ARI)"] >= 0.9) & (sc["smallest cluster"] >= 30)]   # stable, and no cluster made of a handful of outliers
    k_best = int(ok["silhouette"].idxmax()) if len(ok) else int(sc["silhouette"].idxmax())
    seg_k[city] = k_best; seg_prep[city] = prep; seg_model[city] = KMeans(n_clusters=k_best, n_init=20, random_state=SEED).fit(M)
    print(f"{city}: {len(d):,} training ads -> chosen k = {k_best}"); display(sc.round(3))
fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
for i, city in enumerate(["Astana", "Almaty"]):
    sc = seg_scores[city]
    for j, col in enumerate(["inertia", "silhouette", "stability (mean ARI)"]):
        ax[i, j].plot(sc.index, sc[col], marker="o"); ax[i, j].axvline(seg_k[city], color="C3", ls="--", lw=1); ax[i, j].set_title(f"{city}: {col}"); ax[i, j].set_xlabel("k")
plt.tight_layout(); plt.savefig(FIG / "e04_kmeans_selection.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Choosing k.** *Astana:* the silhouette is flat between k = 3 and 5 (0.205–0.207) and falls from k = 6 on; the inertia bends nowhere sharply; stability decides. k = 3 is the least stable choice of all (mean ARI 0.81, worst subsample 0.55: drop a fifth of the ads and one of the three clusters moves), k = 2 is the most stable (0.99) but only says "old versus new", and **k = 4** combines a silhouette of 0.206 with a stability of 0.97 (worst 0.95) and no cluster under 180 ads. *Almaty:* the silhouette rises to a maximum of **0.250 at k = 5**, which is also the most stable partition (0.97, worst 0.92); k = 3 and k = 6 are unstable (0.64 and 0.66), a sign that at those k the algorithm has two equally good ways to cut the cloud. A silhouette of 0.21–0.25 is low: these are cuts through a continuous market, not separated groups, and the stability measure is what tells us that the cuts are at least reproducible. Before the coordinate fix of Section 2.2 the same procedure produced clusters of one or two ads sitting hundreds of kilometres away, which is how that data error was found.

# %% [markdown]
# **The segments in original units.** Medians of the inputs per segment, then, and only then, the asking price per m², the median price and the share of ads above 80 M KZT (which were never inputs).

# %%
def assign_segment(df):
    lab = pd.Series(index=df.index, dtype=object)
    for city in ["Astana", "Almaty"]:
        m = df["city"] == city
        if m.any(): lab[m] = [f"{city[:3]}-{l}" for l in seg_model[city].predict(seg_prep[city].transform(df.loc[m, SEG]))]
    return lab
train_df["segment"] = assign_segment(train_df); test_df["segment"] = assign_segment(test_df)
prof = train_df.groupby("segment").agg(ads=("id", "size"), area_m2=("area", "median"), rooms=("rooms", "median"), floor=("floor", "median"),
                                       floors_in_building=("total_floors", "median"), year_built=("year_built", "median"),
                                       lat=("lat", "median"), lon=("lon", "median"),
                                       fresh_renovation_share=("condition", lambda s: (s == "fresh_renovation").mean()),
                                       price_m2_kKZT=("price_m2", lambda s: s.median() / 1e3), median_price_M=("price", lambda s: s.median() / 1e6),
                                       share_above_80M=("price", lambda s: (s >= 80e6).mean())).round(2)
prof.to_csv(OUT / "segment_profiles.csv"); display(prof)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, city in zip(ax, ["Astana", "Almaty"]):
    d = train_df[train_df["city"] == city]
    for s, g in d.groupby("segment"): a.scatter(g["lon"], g["lat"], s=5, alpha=.5, label=f"{s} (n={len(g)}, {g['price_m2'].median()/1e3:.0f} k/m²)")
    a.set_title(f"{city}: K-Means segments on the map"); a.set_xlabel("longitude"); a.set_ylabel("latitude"); a.legend(fontsize=8, markerscale=3)
plt.tight_layout(); plt.savefig(FIG / "e05_segments_map.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **The segments in words.** *Astana (4):* **Ast-3**, the mass market, 912 ads of 45 m² in buildings from 2024, 659 k KZT/m², median 31 M; **Ast-1**, large new family flats, 584 ads of 92 m² and 3 rooms, 667 k/m², median 60 M, 27 % of them above 80 M; **Ast-2**, high floors in tall new towers, 325 ads on the 13th floor of 18, 70 m², 650 k/m²; **Ast-0**, the old low-rise stock, 182 ads of 56 m² in 6-storey buildings from 2006 (median), by far the cheapest per m² (477 k) and in money (26 M), concentrated north-east of the centre on the map. *Almaty (5):* **Alm-4**, the premium high-rise, 283 ads on the 12th floor of 16, 80 m², the dearest per m² (848 k), 32 % above 80 M; **Alm-0**, large flats in new low-rise club houses, 385 ads of 87 m² in 8-storey buildings, 780 k/m², median 67.5 M, 36 % above 80 M; **Alm-3**, the Soviet-era centre, 319 ads of 51 m² in 5-storey buildings from 1976, still 766 k/m² because of where they stand; **Alm-2**, small new flats in the west, 531 ads of 47.5 m², 722 k/m²; **Alm-1**, the new north-eastern outskirts (lat 43.31, lon 76.99 on the map), 331 ads of 57 m², the cheapest per m² in the city (632 k).
#
# Price per m², which was never an input, differs by 1.4× between segments in Astana (477 to 667 k) and 1.3× in Almaty (632 to 848 k), so the segments do carry price information, but it is information that the fields already contain (age, height, location), which is what Section 4.4 tests. What the segmentation tells us that a single field does not: the cheap segment of Astana is defined by age *and* place together (old low-rise in the north-east), and in Almaty two segments of the same size and age (Alm-2 and Alm-1) differ by 90 k/m² only through where they are. The maps show the same: Astana's segments overlap everywhere except the old north-east, Almaty's are geographic (west, centre, north-east).

# %% [markdown]
# ### 4.2 DBSCAN: dense groups and unusual ads
# DBSCAN on the same seven standardised inputs, all training rows together (the two cities are far apart in `lat`/`lon`, so they can never be joined). `min_samples = 14`, twice the number of inputs; `eps` is read from the knee of the sorted distance to the 14th neighbour (the point farthest from the straight line between the ends of the curve), with two other values as a check. The noise points are the ads that have fewer than 14 similar apartments within `eps`: candidates for data errors or for genuinely unusual objects.

# %%
all_prep = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit(train_df[SEG])
M_all = all_prep.transform(train_df[SEG])
kd = np.sort(NearestNeighbors(n_neighbors=14).fit(M_all).kneighbors(M_all)[0][:, -1])
p1, p2 = np.array([0, kd[0]]), np.array([len(kd) - 1, kd[-1]]); line = p2 - p1; line /= np.linalg.norm(line)
pts = np.column_stack([np.arange(len(kd)), kd]) - p1
dist = np.abs(pts[:, 0] * line[1] - pts[:, 1] * line[0]); knee = int(np.argmax(dist)); eps = float(kd[knee])
db_rows = []
for e in [eps * 0.75, eps, eps * 1.25]:
    lab = DBSCAN(eps=e, min_samples=14).fit_predict(M_all)
    sizes = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.array([])
    db_rows.append({"eps": round(e, 3), "clusters": int(lab.max() + 1), "noise ads": int((lab == -1).sum()), "noise %": round(100 * (lab == -1).mean(), 1),
                    "largest cluster": int(sizes.max()) if len(sizes) else 0, "clusters ≥ 100 ads": int((sizes >= 100).sum())})
display(pd.DataFrame(db_rows).set_index("eps"))
db_labels = DBSCAN(eps=eps, min_samples=14).fit_predict(M_all); train_df["dbscan"] = db_labels
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(kd); ax[0].axhline(eps, color="C3", ls="--", lw=1); ax[0].set_title(f"distance to the 14th neighbour, sorted (knee at eps = {eps:.2f})"); ax[0].set_xlabel("training ads, sorted"); ax[0].set_ylabel("distance (standardised units)")
d = train_df[train_df["city"] == "Almaty"]
for l, g in d.groupby("dbscan"):
    ax[1].scatter(g["lon"], g["lat"], s=5, alpha=.5, color="black" if l == -1 else None, label="noise" if l == -1 else f"cluster {l} (n={len(g)})")
ax[1].set_title("DBSCAN groups, Almaty (black = noise)"); ax[1].legend(fontsize=7, markerscale=3, ncol=2)
plt.tight_layout(); plt.savefig(FIG / "e06_dbscan.png", bbox_inches="tight"); plt.show()
print("cluster sizes by city:"); display(pd.crosstab(train_df["dbscan"], train_df["city"]).T)
kd_all = NearestNeighbors(n_neighbors=14).fit(M_all).kneighbors(M_all)[0][:, -1]
unusual = train_df.assign(k_distance=kd_all)[train_df["dbscan"] == -1].sort_values("k_distance", ascending=False).head(6)
pd.set_option("display.max_colwidth", 110)
display(unusual[["city", "district", "rooms", "area", "floor", "total_floors", "year_built", "price", "segment", "k_distance", "description"]]
        .assign(price=lambda t: (t["price"] / 1e6).round(1), description=lambda t: t["description"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 110)).round(2))

# %% [markdown]
# **What DBSCAN finds.** At the knee, eps = 1.40, DBSCAN returns exactly **two clusters, which are the two cities** (1,984 Astana ads and 1,834 Almaty ads; 5 Almaty ads without coordinates land in the Astana cluster because the imputed median latitude lies in Astana, a per-city imputation would fix that and is noted for the Final), plus **29 noise ads (0.8 %)**; a smaller eps (1.05) breaks the clouds into 10 groups but declares 131 ads noise, a larger one (1.75) leaves 13 noise ads and still two clusters. In seven standardised dimensions each city's market is **one dense cloud that thins out towards its edge**: the sorted 14th-neighbour distance rises slowly from 0.3 to 1.4 over 3,800 ads and only then shoots up, so there is no empty space between "segments" for a density method to find, exactly as in the wholesale data of Practice 6A. That is also the honest reading of the K-Means segments above: a silhouette of 0.21–0.25 says they are cuts through a continuum, useful as a description, not islands.
#
# What DBSCAN is good for here is the **noise**: the 29 ads that have fewer than 14 similar apartments within eps. The six most isolated ones are not typos but unusual objects: a 140 m² flat on the 42nd floor of a 43-storey tower, a 148 m² penthouse on floor 32 of 33, a 6-room 187 m² penthouse on floors 19–20 of a 2025 building (listed twice, ads 222 and 2738, a repost that the Midterm's near-duplicate rule missed because the street field differs), a 1-room flat on the 11th floor of a 34-storey building. We keep them in the training data, since they are real, but Section 7.4 uses "no dense group of similar ads" as a warning that the model's estimate for such an ad rests on few comparables.

# %% [markdown]
# ### 4.3 PCA: how many dimensions do the structured fields carry?
# PCA on the standardised numeric fields of the training rows (`area`, `rooms`, `kitchen_area`, `floor`, `total_floors`, `year_built`, `ceiling_height`, `n_photos`, `lat`, `lon`; `log_area` is left out because it is a function of `area`). Then two questions: what the first components mean (loadings), and whether feeding components instead of raw columns helps a model (KNN and Ridge on the full encoded matrix, with PCA refitted inside every fold).

# %%
from sklearn.decomposition import PCA
PCA_COLS = ["area", "rooms", "kitchen_area", "floor", "total_floors", "year_built", "ceiling_height", "n_photos", "lat", "lon"]
pca_prep = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit(train_df[PCA_COLS])
Z = pca_prep.transform(train_df[PCA_COLS]); pca = PCA(random_state=SEED).fit(Z); S = pca.transform(Z)
cum = np.cumsum(pca.explained_variance_ratio_)
print("explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
print("components for 80 / 90 / 95 %:", [int(np.searchsorted(cum, q) + 1) for q in (0.8, 0.9, 0.95)], "of", len(PCA_COLS))
load = pd.DataFrame(pca.components_[:3].T, index=PCA_COLS, columns=["PC1", "PC2", "PC3"]).round(3)
display(load)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(range(1, len(cum) + 1), cum * 100, marker="o"); ax[0].axhline(90, color="grey", ls="--", lw=1); ax[0].set_xlabel("components"); ax[0].set_ylabel("cumulative explained variance, %"); ax[0].set_title("Numeric fields: cumulative variance")
for a, (col, title, kw) in zip(ax[1:], [(train_df["city"].map({"Astana": 0, "Almaty": 1}), "PC1 × PC2 coloured by city", dict(cmap="coolwarm")),
                                          (np.log10(train_df["price_m2"]), "PC1 × PC2 coloured by log10(price per m²)", dict(cmap="viridis"))]):
    s = a.scatter(S[:, 0], S[:, 1], c=col, s=4, alpha=.5, **kw); a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_title(title)
plt.colorbar(s, ax=ax[2], fraction=.04)
plt.tight_layout(); plt.savefig(FIG / "e07_pca.png", bbox_inches="tight"); plt.show()

# %%
pca_rows = []
for model_name, model in [("KNN (k=10, distance)", KNeighborsRegressor(10, weights="distance")), ("Ridge (alpha=10)", Ridge(alpha=10))]:
    for k in [None, 0.9, 20, 10, 5, 2]:
        steps = [("pre", make_pre(True))] + ([("pca", PCA(n_components=k, random_state=SEED))] if k is not None else []) + [("model", model)]
        est = TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log, inverse_func=np.exp, check_inverse=False)
        r = cv_eval(f"{model_name} + PCA({k})" if k is not None else f"{model_name}, no PCA", est, register=(k in (None, 0.9) and "KNN" in model_name), verbose=False)
        pca_rows.append({"model": model_name, "PCA components": "none" if k is None else k, "CV MAE (M KZT)": r["CV MAE (M KZT)"], "SD": r["CV MAE SD"], "CV MdAPE %": r["CV MdAPE %"]})
n_enc = make_pre(True).fit(X_train).transform(X_train.head()).shape[1]
print(f"encoded matrix has {n_enc} columns")
display(pd.DataFrame(pca_rows).pivot(index="PCA components", columns="model", values="CV MAE (M KZT)").round(2).reindex(["none", 0.9, 20, 10, 5, 2]))

# %% [markdown]
# **How many dimensions, and what they are.** Ten numeric fields compress to **5 components for 80 % and 7 for 90 %** of the variance, so there is some redundancy but not much: PC1 carries 27.8 %, PC2 21.8 %, and the last component carries 0.0 %, because across two cities latitude and longitude are perfectly anti-correlated (loadings +0.40 and −0.40 on every component: all of Astana is north-west of all of Almaty), so one coordinate is redundant. The loadings read naturally. **PC1** has positive weights on total floors (0.41), latitude (0.40), year built (0.34) and floor (0.34) and a negative one on longitude (−0.40): it is "new high-rise in Astana" against "older low-rise in Almaty", which is why the middle map separates the cities along PC1. **PC2** is size: area (0.50), rooms (0.52), kitchen area (0.23) and the number of photos (0.35; big flats get more photos), opposed to the Astana direction. **PC3** is the floor and the height of the building against area and rooms: a small flat high up versus a large flat in a low building. Price per m² does not follow either axis cleanly in the right-hand map, which already suggests that the directions of largest variance are not the directions of price.
#
# **Does PCA help a model?** No, at any number of components. KNN on the raw 71-column matrix scores 11.13 M; on the components that keep 90 % of the variance 11.48 M, on 10 components 11.33, on 2 components 12.44. Ridge loses even more: 9.10 M without PCA, 10.61 M with 90 % of the variance kept, 11.32 M with 10 components. PCA keeps the variance of the inputs, which in this matrix is dominated by the one-hot district and attribute columns, and throws away low-variance directions that carry price; a model wants the second kind. The Midterm's hand-made KNN on size and location only (10.18 M) beats both KNN on everything (11.13) and KNN on principal components (11.48): choosing columns by what they mean beats choosing directions by how much they vary. PCA stays in this notebook as a description of the table, not as a preprocessing step, which is the same conclusion as Practice 6B and Assignment 2.

# %% [markdown]
# ### 4.4 Does a segment feature help the supervised model?
# The Midterm plan promised to try the cluster id as a feature. To keep it leakage-free, K-Means is fitted **inside each CV fold** on the seven segmentation inputs of that fold's training rows (both cities together with k = 6, since a per-city fit inside a pipeline would need the city label; the one-hot city column is already there), and its one-hot segment id is appended to the tuned boosting pipeline.

# %%
class KMeansSegment(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=6, random_state=SEED): self.n_clusters, self.random_state = n_clusters, random_state
    def fit(self, X, y=None): self.km_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(X); return self
    def transform(self, X): return np.eye(self.n_clusters)[self.km_.predict(X)]
    def get_feature_names_out(self, input_features=None): return np.array([f"segment_{i}" for i in range(self.n_clusters)])
hgb_params = {k.replace("regressor__model__", ""): v for k, v in hgb_search.best_params_.items()}
pre_seg = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))]), NUM + BOOL),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), CAT),
                             ("seg", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("km", KMeansSegment(6))]), SEG)])
seg_est = TransformedTargetRegressor(regressor=Pipeline([("pre", pre_seg), ("model", HistGradientBoostingRegressor(**hgb_params, random_state=SEED))]), func=np.log, inverse_func=np.exp, check_inverse=False)
cv_eval("Gradient boosting (tuned) + K-Means segment", seg_est)
display(table(["Gradient boosting (tuned)", "Gradient boosting (tuned) + K-Means segment"]))

# %% [markdown]
# **The segment feature changes nothing:** 6.65 ± 0.31 M with it against 6.62 ± 0.35 M without, a difference of 0.03 M, a tenth of the fold SD, and the MdAPE moves by 0.05 points. The boosting model already receives the seven inputs the segments are built from and carves the same regions itself with its own splits, so a cluster id tells it nothing new. The promise of the Midterm plan is kept and the answer is negative; the segment column is left out of the final model, and the segments are used where they are useful, in the error analysis of Section 7.2.

# %% [markdown]
# ## 5. A neural network in PyTorch (Weeks 7–8)
#
# **Why.** The Midterm's linear model cannot bend and its tree bends only in steps. A multilayer perceptron learns smooth interactions between area, age, district and coordinates; it is the Week 7–8 tool and the last model class promised in the plan. We build it in PyTorch the way Practice 8 did: the same preprocessing as the linear models (median imputation, standardisation, one-hot encoding, refitted on each training fold), the target is **standardised log(price)**, the loss is **L1** (the absolute error on the log scale is a relative error, which matches MdAPE), the optimiser is Adam with weight decay, mini-batches of 128 rows, and **early stopping** on an inner validation split (15 % of the training fold) with a patience of 30 epochs. Each configuration is cross-validated on the same five folds as every other model.
#
# The estimator below wraps all of that in the scikit-learn interface, so that `cross_validate` and the final comparison treat it like any other model.

# %%
import torch, torch.nn as nn
torch.set_num_threads(1)  # one thread: torch ships its own OpenMP runtime and xgboost/lightgbm (imported above) use the system one; two runtimes in one process deadlock torch's thread pool on macOS. The networks are small enough that one thread costs nothing.

class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """MLP on log(price): hidden layers `hidden` (empty tuple = linear model), ReLU (or sigmoid), dropout, L1 loss, Adam, early stopping."""
    def __init__(self, hidden=(128, 64), dropout=0.1, lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=300, patience=30, val_fraction=0.15, activation="relu", seed=SEED):
        self.hidden, self.dropout, self.lr, self.weight_decay, self.batch_size, self.max_epochs, self.patience, self.val_fraction, self.activation, self.seed = hidden, dropout, lr, weight_decay, batch_size, max_epochs, patience, val_fraction, activation, seed
    def _build(self, n_in):
        layers, d = [], n_in
        act = nn.Sigmoid if self.activation == "sigmoid" else nn.ReLU
        for h in self.hidden: layers += [nn.Linear(d, h), act(), nn.Dropout(self.dropout)]; d = h
        layers.append(nn.Linear(d, 1)); return nn.Sequential(*layers)
    def fit(self, X, y):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        self.pre_ = make_pre(True).fit(X); Xm = self.pre_.transform(X).astype(np.float32)
        ly = np.log(y); self.mu_, self.sd_ = ly.mean(), ly.std(); t = ((ly - self.mu_) / self.sd_).astype(np.float32)
        idx = rng.permutation(len(Xm)); n_val = int(self.val_fraction * len(Xm)); va, tr = idx[:n_val], idx[n_val:]
        Xtr, ttr = torch.tensor(Xm[tr]), torch.tensor(t[tr]); Xva, yva = torch.tensor(Xm[va]), y[va]
        self.model_ = self._build(Xm.shape[1]); opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.L1Loss(); best, best_state, wait, self.history_ = np.inf, None, 0, []
        for epoch in range(self.max_epochs):
            self.model_.train(); perm = torch.randperm(len(Xtr)); tl = 0.0
            for i in range(0, len(Xtr), self.batch_size):
                b = perm[i:i + self.batch_size]; opt.zero_grad(); loss = loss_fn(self.model_(Xtr[b]).squeeze(1), ttr[b]); loss.backward(); opt.step(); tl += loss.item() * len(b)
            self.model_.eval()
            with torch.no_grad(): pv = self.model_(Xva).squeeze(1).numpy()
            val_mae = mean_absolute_error(yva, np.exp(pv * self.sd_ + self.mu_)) / 1e6
            self.history_.append({"epoch": epoch + 1, "train L1 (log units)": tl / len(Xtr) * self.sd_, "val MAE (M KZT)": val_mae})
            if val_mae < best - 1e-3: best, wait, best_state = val_mae, 0, {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience: break
        self.model_.load_state_dict(best_state); self.best_epoch_ = int(np.argmin([h["val MAE (M KZT)"] for h in self.history_])) + 1
        self.n_params_ = sum(p.numel() for p in self.model_.parameters()); return self
    def predict(self, X):
        self.model_.eval()
        with torch.no_grad(): p = self.model_(torch.tensor(self.pre_.transform(X).astype(np.float32))).squeeze(1).numpy()
        return np.exp(p * self.sd_ + self.mu_)

nn_configs = {"NN: no hidden layer (= linear on log price)": dict(hidden=()), "NN: 64": dict(hidden=(64,)), "NN: 128-64": dict(hidden=(128, 64)),
              "NN: 256-128-64": dict(hidden=(256, 128, 64)), "NN: 128-64, lr = 0.05 (too high)": dict(hidden=(128, 64), lr=0.05)}
nn_rows, nn_hist = {}, {}
tr0, va0 = next(cv.split(X_train))
for name, cfg in nn_configs.items():
    t0 = time.time(); est = TorchMLPRegressor(**cfg)
    r = cv_eval(name, est, n_jobs=1, register=("too high" not in name), verbose=False)
    one = TorchMLPRegressor(**cfg).fit(X_train.iloc[tr0], y_train[tr0]); nn_hist[name] = pd.DataFrame(one.history_)
    nn_rows[name] = {**r, "parameters": one.n_params_, "best epoch (fold 1)": one.best_epoch_, "epochs run (fold 1)": len(one.history_), "time per fit s": (time.time() - t0) / 6}
    print(f"{name:46s} CV MAE {r['CV MAE (M KZT)']:.2f} ± {r['CV MAE SD']:.2f} M  MdAPE {r['CV MdAPE %']:.1f} %  params {one.n_params_:,}  best epoch {one.best_epoch_}  ({time.time()-t0:.0f} s)")
nn_t = pd.DataFrame(nn_rows).T.drop(columns=["folds MAE", "fit time s"]).astype(float).round(2)
display(nn_t)
fig, ax = plt.subplots(1, len(nn_configs), figsize=(17, 3.6), sharey=True)
for a, (name, h) in zip(ax, nn_hist.items()):
    a.plot(h["epoch"], h["val MAE (M KZT)"], color="C0", label="inner validation MAE, M KZT (left axis)")
    a2 = a.twinx(); a2.plot(h["epoch"], h["train L1 (log units)"], color="C1", alpha=.7, label="training L1 loss, log units (right axis)"); a2.set_ylim(0.05, 0.5)
    if a is not ax[-1]: a2.set_yticklabels([])
    a.axvline(nn_rows[name]["best epoch (fold 1)"], color="C3", ls="--", lw=1); a.set_title(name.replace("NN: ", ""), fontsize=9); a.set_xlabel("epoch"); a.set_ylim(5, 16)
ax[0].set_ylabel("validation MAE, M KZT"); ax[0].legend(fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig(FIG / "e08_nn_curves.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **What the networks do.** The network with **no hidden layer** is the linear model on log price trained by Adam instead of least squares, and it lands at **9.26 ± 0.57 M** against 9.10 M for the Midterm's linear model: the same model within a quarter of a fold SD, which is the check that the training loop, the loss and the early stopping are right. Adding hidden layers bends the function a little: **64 units give 8.82 ± 0.60 M**, 128-64 gives 8.90 M and 256-128-64 gives 8.86 ± 0.41 M, all within 0.1 M of each other and about 0.3 M (half a fold SD) below the linear network, with 4,700 to 60,000 parameters for 3,270 inner-training rows. The curves show why there is no more to gain: the training loss keeps falling (to 0.13 log units for the deepest network, which is a 13 % error on its training ads), while the inner-validation MAE stops improving after 10–30 epochs and then wanders by ±0.3 M; the networks memorise, and early stopping, dropout and weight decay are what keep them honest. With a **learning rate of 0.05** the validation MAE jumps between 8.5 and 10.5 M from one epoch to the next and the training loss never settles: Adam's steps overshoot the minimum, early stopping fires on a lucky epoch, and the CV score is the worst of the five (9.43 M). Every fit takes 0.3–0.7 s on one CPU thread of the laptop, so a GPU was never a question on these data.

# %% [markdown]
# **Gradient flow with depth (the Practice 8A measurement).** Our networks use ReLU and at most three hidden layers, so the vanishing gradient of Lecture 8 should not touch them; the measurement below shows that this is a choice and not luck. For networks of 1, 3 and 6 hidden layers of 64 units, once with ReLU and once with sigmoid, we take the untrained network (PyTorch's default initialisation, the same seed), compute the L1 loss on all inner-training rows of fold 1, call `backward()` once and compare the gradient norm of the first hidden layer's weights with that of the last. We then train each of the six networks on fold 1 with the usual early stopping and read the best inner-validation MAE and the epoch at which it was reached.

# %%
pre0 = make_pre(True).fit(X_train.iloc[tr0]); X0 = torch.tensor(pre0.transform(X_train.iloc[tr0]).astype(np.float32))
ly0 = np.log(y_train[tr0]); t0_ = torch.tensor(((ly0 - ly0.mean()) / ly0.std()).astype(np.float32))
grad_rows = []
for depth in (1, 3, 6):
    for act in ("relu", "sigmoid"):
        torch.manual_seed(SEED); net = TorchMLPRegressor(hidden=(64,) * depth, activation=act, dropout=0.0)._build(X0.shape[1])
        nn.L1Loss()(net(X0).squeeze(1), t0_).backward()
        lin = [m for m in net if isinstance(m, nn.Linear)]
        g_first, g_last = lin[0].weight.grad.norm().item(), lin[-2].weight.grad.norm().item()
        fitted = TorchMLPRegressor(hidden=(64,) * depth, activation=act).fit(X_train.iloc[tr0], y_train[tr0])
        grad_rows.append({"hidden layers": depth, "activation": act, "grad norm, first hidden layer": g_first, "grad norm, last hidden layer": g_last,
                          "ratio first / last": g_first / g_last, "best inner-val MAE (M KZT), fold 1": min(h["val MAE (M KZT)"] for h in fitted.history_),
                          "best epoch": fitted.best_epoch_, "epochs run": len(fitted.history_)})
grad_t = pd.DataFrame(grad_rows).set_index(["hidden layers", "activation"])
pd.set_option("display.float_format", lambda v: f"{v:.4g}"); display(grad_t); pd.reset_option("display.float_format")
grad_t.to_csv(OUT / "nn_gradient_flow.csv")

# %% [markdown]
# **What the gradients say.** With one hidden layer the ratio is 1 by construction (the first and the last hidden layer are the same layer). With three hidden layers the ReLU network passes the gradient through almost unchanged (ratio **1.10**) while the sigmoid network loses a factor of 30 (**0.031**); with six layers ReLU keeps **0.12** and sigmoid is down to **5.5 × 10⁻⁵**, four orders of magnitude less. This is the Lecture 8 argument in our numbers: the slope of the sigmoid is at most 0.25, so every sigmoid layer multiplies the gradient by a quarter times its weights, a factor of 5–7 per layer here, whereas an active ReLU unit passes it on unchanged (the mild loss of the deep ReLU network comes from PyTorch's default initialisation, which is tuned a little small for ReLU). Training tells the same story: the ReLU networks reach their best inner-validation MAE after **25–30 epochs** (8.63–8.68 M on fold 1); the one- and three-layer sigmoid networks get to a similar place (8.32 and 8.35 M) but need **264 and 285 epochs**, ten times more, and the three-layer one was still improving when it hit the 300-epoch cap; the six-layer sigmoid network **does not learn at all**: its best inner-validation MAE is 20.3 M, the level of the dummy model, because its first layers receive no gradient and the output stays at the mean. These are fold-1 inner-validation numbers, a diagnosis and not a model choice: the 0.3 M by which the shallow sigmoid networks beat ReLU on this fold is within one fold SD, and none of the six comes near the boosting models. The design we started with is the one the measurement supports: ReLU, few layers, early stopping.

# %% [markdown]
# **Cross-check with scikit-learn's MLPRegressor.** The same idea with another implementation (two hidden layers, tanh, L2 penalty 0.1, early stopping), to make sure the PyTorch result is not an artefact of our training loop.

# %%
from sklearn.neural_network import MLPRegressor
cv_eval("sklearn MLP 128-64 (tanh, alpha 0.1)", logged(MLPRegressor(hidden_layer_sizes=(128, 64), activation="tanh", alpha=0.1, learning_rate_init=1e-3, max_iter=700, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED), scaled=True))
display(table(["Midterm linear (log price)", "NN: no hidden layer (= linear on log price)", "NN: 64", "NN: 128-64", "NN: 256-128-64", "sklearn MLP 128-64 (tanh, alpha 0.1)", "Gradient boosting (tuned)"]))

# %% [markdown]
# **An honest comparison.** scikit-learn's MLP lands at **8.86 ± 0.53 M**, the same as our PyTorch networks (8.82–8.90), so the result is not an artefact of our loop. Against the tuned boosting model (**6.62 ± 0.35 M**) every network is 2.2 M behind, six fold SDs, and the gap is the same on every fold. The reason is the shape of the data, not the training: 3,852 rows with 24 fields, where the strongest effects are thresholds and interactions (a building from 2018 onwards, the top floor, a district border, area × district) that a tree finds with one split, while a network has to approximate each of them with smooth functions learned from a few hundred examples, and the one-hot districts give it sparse inputs it cannot interpolate between. The lecture's example of a network with 1,345 parameters that did no better than a linear model is our result too: on these data a neural network is not worth it. It will become worth it at the Final, when the inputs are text and photo embeddings, which is where networks earn their keep.

# %% [markdown]
# ## 6. Choosing the model and one evaluation on the saved test split
#
# The lecture's checklist for choosing an algorithm fits what the table shows: we have many rows relative to features (3,852 × 24), the relations are non-linear with interactions, the target has outliers (the luxury tail) that pull a linear fit, and the user needs accuracy in money more than coefficients to read. That points to tree ensembles, and the only cost is interpretability, which Section 3.4 recovers through permutation importance.
#
# Every model of this notebook on the same five folds, sorted by CV MAE. **The choice is made on the CV column** (the Midterm chose on the test column, which was a slip). Then each model is fitted on all 3,852 training rows and evaluated **once** on the 963 saved test ads, so that the Endterm table is directly comparable with the Midterm table and the Final can be compared with both.

# %%
order = sorted(registry, key=lambda n: registry[n][1]["CV MAE (M KZT)"])
best_name = order[0]
rows = []
t0 = time.time()
test_pred = {}
for n in order:
    est = clone(registry[n][0]) if not isinstance(registry[n][0], (RandomizedSearchCV,)) else registry[n][0]
    est.fit(X_train, y_train); p = est.predict(X_test); test_pred[n] = p; registry[n] = (est, registry[n][1])
    r = registry[n][1]
    rows.append({"model": n, "CV MAE (M KZT)": f"{r['CV MAE (M KZT)']:.2f} ± {r['CV MAE SD']:.2f}", "CV MdAPE %": round(r["CV MdAPE %"], 1), "CV R²": round(r["CV R²"], 3),
                 "Test MAE (M KZT)": round(mean_absolute_error(y_test, p) / 1e6, 3), "Test RMSE (M KZT)": round(root_mean_squared_error(y_test, p) / 1e6, 2),
                 "Test MdAPE %": round(mdape(y_test, p), 2), "Test R²": round(r2_score(y_test, p), 3), "within ±10 %": f"{(np.abs(p - y_test) / y_test <= 0.10).mean():.0%}", "within ±20 %": f"{(np.abs(p - y_test) / y_test <= 0.20).mean():.0%}"})
results = pd.DataFrame(rows).set_index("model"); results.to_csv(OUT / "results_table.csv")
print(f"chosen by CV MAE: {best_name}   ({time.time()-t0:.0f} s for the {len(order)} final fits)")
display(results)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
cvm = pd.Series({n: registry[n][1]["CV MAE (M KZT)"] for n in order}); sd = pd.Series({n: registry[n][1]["CV MAE SD"] for n in order})
colors = ["C3" if n == best_name else ("grey" if n.startswith(("Dummy", "District", "Midterm")) else "C0") for n in order]
ax[0].barh(order, cvm, xerr=sd, color=colors); ax[0].invert_yaxis(); ax[0].set_xlabel("5-fold CV MAE, M KZT (± fold SD)"); ax[0].set_title("Model selection: cross-validation on the training rows")
ax[1].barh(order, results["Test MdAPE %"], color=colors); ax[1].invert_yaxis(); ax[1].axvline(10, color="black", ls="--", lw=1, label="registration goal: 10 %"); ax[1].set_xlabel("test MdAPE, %"); ax[1].set_title("The same models on the saved test split (used once)"); ax[1].legend()
ax[1].set_yticklabels([]); ax[0].tick_params(axis="y", labelsize=8)
plt.tight_layout(); plt.savefig(FIG / "e09_model_comparison.png", bbox_inches="tight"); plt.show()

# %%
best_est = registry[best_name][0]; p_best = test_pred[best_name]
rng = np.random.default_rng(SEED); boot = []
for _ in range(2000):
    i = rng.integers(0, len(y_test), len(y_test)); boot.append((mean_absolute_error(y_test[i], p_best[i]) / 1e6, mdape(y_test[i], p_best[i])))
ci = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
print(f"{best_name}: test MAE {mean_absolute_error(y_test, p_best)/1e6:.2f} M KZT (95 % bootstrap interval {ci[0,0]:.2f}–{ci[1,0]:.2f}), test MdAPE {mdape(y_test, p_best):.2f} % ({ci[0,1]:.2f}–{ci[1,1]:.2f})")
import joblib; joblib.dump(best_est, "models/endterm_best_model.joblib"); print("saved to models/endterm_best_model.joblib")

# %% [markdown]
# **The choice and the one test.** Cross-validation ranks the tuned histogram boosting first (6.62 M), XGBoost second (6.64), the boosting with the segment feature third (6.65), LightGBM fourth (6.75), the blend and the default histogram boosting next (6.95 and 7.01), then the forests (8.0), the extra trees and the networks (8.7–8.9), the Midterm linear model (9.10), and the Week 3–4 models below it. The chosen model, **tuned histogram gradient boosting on log price**, reaches on the 963 saved test ads a **MAE of 6.29 M KZT** (95 % bootstrap interval 5.56–7.14), a **MdAPE of 9.13 %** (8.31–9.88) and an **R² of 0.855**; 54 % of the test ads are within ±10 % of the estimate and 84 % within ±20 %, against 8.66 M, 11.27 %, 0.712, 45 % and 73 % for the Midterm's best model on the same ads. The error in money is down by 27 %, and the registration goal of a median error under 10 % is **reached**, with the whole bootstrap interval below 10 %. The test MAE (6.29) sits below the CV estimate (6.62) by less than one fold SD, so the CV estimate was honest.
#
# On the test column LightGBM (MAE 6.19 M, R² 0.877), XGBoost (6.27 M, MdAPE 8.94 %), the blend (MdAPE 8.68 %) and the default boosting (9.06 %) look as good as or slightly better than the chosen model on one metric or another, and the six best models are inside each other's bootstrap intervals (5.56–7.14 M). We do **not** switch to LightGBM or the blend: that would be choosing on the test set, the Midterm's slip, and a 0.1 M MAE or 0.45-point MdAPE difference on 963 ads is far inside the interval. The order of the six is a draw; the choice stands with the CV column, and the fact that three boosting libraries and the blend all land at 6.2–6.5 M on the test ads is the best evidence that the number is real.

# %% [markdown]
# ## 7. Error analysis of the chosen model
# ### 7.1 Residuals

# %%
resid = p_best - y_test; rel = resid / y_test * 100
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(y_test / 1e6, p_best / 1e6, s=8, alpha=.4); lim = [min(y_test.min(), p_best.min()) / 1e6, max(y_test.max(), p_best.max()) / 1e6]
ax[0].plot(lim, lim, "r--"); ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("actual asking price, M KZT"); ax[0].set_ylabel("predicted, M KZT"); ax[0].set_title(f"{best_name}: predicted vs actual")
ax[1].scatter(X_test["area"], rel, s=8, alpha=.4); ax[1].axhline(0, color="r", ls="--"); ax[1].set_xlabel("area, m²"); ax[1].set_ylabel("error, % of actual"); ax[1].set_title("Relative error vs area"); ax[1].set_ylim(-80, 120)
sns.histplot(rel.clip(-80, 120), bins=60, ax=ax[2]); ax[2].axvline(0, color="r", ls="--"); ax[2].set_xlabel("error, % of actual"); ax[2].set_title("Distribution of relative errors")
plt.tight_layout(); plt.savefig(FIG / "e10_residuals.png", bbox_inches="tight"); plt.show()
print(f"median signed error {np.median(rel):+.1f} %  |  within ±10 %: {(np.abs(rel) <= 10).mean():.1%}  within ±20 %: {(np.abs(rel) <= 20).mean():.1%}  |  Midterm linear: {(np.abs(test_pred['Midterm linear (log price)'] - y_test) / y_test <= 0.10).mean():.1%} / {(np.abs(test_pred['Midterm linear (log price)'] - y_test) / y_test <= 0.20).mean():.1%}")

# %% [markdown]
# ### 7.2 Where the error lives: price band, city, size, segment, and the Midterm comparison

# %%
err = test_df[["id", "city", "district", "rooms", "area", "price", "year_built", "condition", "seller_type", "segment", "description"]].copy()
err["pred"] = p_best; err["abs_err_M"] = np.abs(resid) / 1e6; err["ape_%"] = np.abs(rel); err["signed_%"] = rel
err["midterm_ape_%"] = np.abs(test_pred["Midterm linear (log price)"] - y_test) / y_test * 100
err["price_band"] = pd.cut(err["price"] / 1e6, [0, 20, 35, 50, 80, 150, 1e4], labels=["<20M", "20–35M", "35–50M", "50–80M", "80–150M", ">150M"])
tabs = []
for col in ["price_band", "city", "rooms", "segment"]:
    t = err.groupby(col, observed=True).agg(n=("pred", "size"), MAE_M=("abs_err_M", "mean"), MdAPE=("ape_%", "median"), median_signed=("signed_%", "median"), Midterm_MdAPE=("midterm_ape_%", "median")).round(1)
    t.index = [f"{col}={i}" for i in t.index]; tabs.append(t)
seg_err = pd.concat(tabs); seg_err.to_csv(OUT / "error_by_segment.csv"); display(seg_err)

# %% [markdown]
# **Where the error lives.** *By price band:* the mass market (20–50 M KZT, 595 of the 963 test ads) has a MdAPE of **7.8–8.1 %**, the band 50–80 M 10.7 %, the band 80–150 M **12.9 %** against 21.9 % for the Midterm model, and the 15 ads above 150 M **16.6 %** against 31.8 %. The ensembles fixed most of what the Midterm diagnosed as "one global model treats the luxury segment like the mass market": the expensive bands improved twice as much as the cheap ones. The *median signed error* shows the remaining bias: cheap ads (<20 M) are over-estimated by 9.4 % (a flat is cheap for a reason that is often not in the fields: rough finish, cash-only sale), and the luxury ads are under-estimated by 7 % (80–150 M) and 14 % (>150 M): with 15 such ads the model shrinks them towards the mass market. *By city:* Almaty 8.7 %, Astana 9.4 %, the same within noise, while the MAE is higher in Almaty (7.0 against 5.6 M) simply because prices there are higher. *By rooms:* 1–3 rooms 8–10 %, 4 rooms 12.4 %, 5 rooms 21.8 % (10 ads), the same pattern as the price bands. *By segment:* the hardest segments are Astana's old low-rise stock (Ast-0, MdAPE 11.1 %, over-estimated by 4.8 %, and the only segment that did not improve on the Midterm's 11.0 %) and its tall-tower flats (Ast-2, 10.9 %), while the mass-market segments of both cities are at 8.1–8.4 %; the largest errors in money are in Almaty's large club-house flats (Alm-0, MAE 14.8 M) and premium high-rise (Alm-4, 9.4 M, under-estimated by 4.6 %). Every row of the table is better than the Midterm's, the smallest gain in the cheapest band and the largest in the most expensive ones.

# %% [markdown]
# ### 7.3 The worst cases, and what happened to the Midterm's worst cases

# %%
worst = err.sort_values("ape_%", ascending=False).head(8)
show = lambda t: display(t[["city", "district", "rooms", "area", "year_built", "condition", "price", "pred", "ape_%", "midterm_ape_%", "description"]]
                         .assign(price=lambda d: (d["price"] / 1e6).round(1), pred=lambda d: (d["pred"] / 1e6).round(1),
                                 description=lambda d: d["description"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 120)).round(1))
print("8 worst test errors of the chosen model:"); show(worst)
print("\nthe 8 worst test errors of the Midterm linear model, with the Endterm error on the same ads:"); show(err.sort_values("midterm_ape_%", ascending=False).head(8))
better = (err["ape_%"] < err["midterm_ape_%"]).mean()
print(f"\nthe Endterm model is closer than the Midterm model on {better:.0%} of the 963 test ads; its MdAPE on the Midterm's 50 worst ads is {err.nlargest(50, 'midterm_ape_%')['ape_%'].median():.1f} % against {err.nlargest(50, 'midterm_ape_%')['midterm_ape_%'].median():.1f} %")

# %% [markdown]
# **The worst cases, and what the Endterm fixed.** The eight worst test errors of the chosen model fall into two groups. Half are **ads that are cheap for a reason that only the text gives**: ad 811 is a 1-room flat at 13.9 M that says "cash only, sale by assignment of rights" (predicted 20.8 M), ad 3316 is a 99 m² flat in rough state at 34 M (predicted 58 M), ad 1730 a 55 m² flat from 1974 that "needs repair" in the centre of Almaty (40.5 against 62.8 M). The other half are **unique objects with no comparables**: a 228 m² penthouse (110 against 171 M), a 276 m² designer duplex at 525 M (predicted 245 M, the one large under-estimate), a 107 m² flat with a mountain view at 85 M (predicted 134 M). The fields describe an ordinary new apartment in each case; the explanation sits in the description, which is the Final's input.
#
# The Midterm's eight worst ads tell the Endterm's story: the 364 m² penthouse went from a 133 % error to 33 %, the 56 m² flat in Nura from 68 % to 3 %, the 71 m² flat in Almalinsky from 70 % to 8 %, the 235 m² Medeu apartment from 69 % to 26 %, while four stay above 40 %: the townhouse listed as a flat (1392), the Primera complex flat (1588), a 49 m² flat from 1970 in Turksib (4490, from 75 % to 48 %) and a 156 m² flat in Alatau (2004, from 66 % to 48 %). Over all 963 test ads the Endterm model is closer than the Midterm model on 60 %, and on the Midterm's 50 worst ads its MdAPE is **23.8 % against 46.8 %**: the hard cases were halved, not solved. The ensembles fixed the errors that came from the shape of the price surface (interactions between size, age and location that a linear model could not bend to) and left the errors that come from information the fields do not contain.

# %% [markdown]
# ### 7.4 Back to the question: which ads are over-priced?
# The user's rule from Section 1: an ad is flagged when its asking price is more than 20 % above the model estimate for comparable apartments. On the test ads this counts how many sellers ask well above the typical price, and the DBSCAN noise flag from 4.2 tells whether the model's estimate is even trustworthy for that ad (an ad with no similar apartments in the data gets a wide-error warning instead of a flag).

# %%
err["deviation_%"] = (err["price"] - err["pred"]) / err["pred"] * 100
near = NearestNeighbors(radius=eps).fit(M_all).radius_neighbors(all_prep.transform(test_df[SEG]), return_distance=False)
err["no_similar_ads"] = np.array([len(n) < 14 for n in near])   # fewer than min_samples training ads within eps = DBSCAN would call it noise
flag = err["deviation_%"] > 20
print(f"test ads asked > 20 % above the estimate: {flag.sum()} of {len(err)} ({flag.mean():.0%}); asked > 20 % below: {(err['deviation_%'] < -20).sum()} ({(err['deviation_%'] < -20).mean():.0%})")
print(f"ads with no dense group of similar apartments (DBSCAN noise): {err['no_similar_ads'].sum()} — their MdAPE is {err.loc[err['no_similar_ads'], 'ape_%'].median():.1f} % against {err.loc[~err['no_similar_ads'], 'ape_%'].median():.1f} % for the rest")
print("flag rate by city:", err.groupby("city")["deviation_%"].apply(lambda s: f"{(s > 20).mean():.0%}").to_dict(), " by seller type:", err.groupby("seller_type")["deviation_%"].apply(lambda s: f"{(s > 20).mean():.0%}").to_dict())
fig, ax = plt.subplots(figsize=(7, 3.4))
sns.histplot(err["deviation_%"].clip(-60, 100), bins=60, ax=ax); ax.axvline(20, color="C3", ls="--", label="+20 %: over-priced flag"); ax.axvline(-20, color="C2", ls="--", label="−20 %: possible bargain or hidden problem"); ax.set_xlabel("asking price relative to the model estimate, %"); ax.legend(); ax.set_title("Test ads: how far the asking price is from the estimate")
plt.tight_layout(); plt.savefig(FIG / "e11_overpricing_flags.png", bbox_inches="tight"); plt.show()
summary = {"best_model": best_name, "test_mae_M": float(mean_absolute_error(y_test, p_best) / 1e6), "test_mdape": float(mdape(y_test, p_best)), "test_r2": float(r2_score(y_test, p_best)),
           "bootstrap95_mae": ci[:, 0].tolist(), "bootstrap95_mdape": ci[:, 1].tolist(), "segments_k": seg_k, "dbscan_eps": eps, "dbscan_noise_train": int((db_labels == -1).sum()),
           "pca_components_90": int(np.searchsorted(cum, 0.9) + 1), "flagged_overpriced_share": float(flag.mean()), "rf_params": rf_params, "hgb_params": hgb_params,
           "n_train": int(len(X_train)), "n_test": int(len(X_test)), "features": FEATURES}
_ = (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

# %% [markdown]
# **The buyer's flag.** The asking prices of the test ads are centred on the model estimate (median deviation 0 %) with a longer right tail: **93 ads of 963 (10 %) ask more than 20 % above the estimate** for comparable apartments, and 50 (5 %) ask more than 20 % below it. The flag rate is higher in Almaty (11 %) than in Astana (8 %), and, against the folk wisdom about agents, owners and agents over-ask equally often (10 % each). The **28 test ads with no dense group of similar apartments** (the DBSCAN criterion of Section 4.2) have a MdAPE of 12.9 % against 9.0 % for the rest, so the warning is justified: for a penthouse or a flat on the 40th floor the estimate rests on a handful of comparables and a 20 % deviation is not yet evidence of over-pricing. For a buyer the rule is therefore: an ordinary ad more than 20 % above the estimate is in the top tenth of asking prices for its kind and is worth questioning; an ad more than 20 % below it is either a bargain or carries a condition stated in the text (cash only, no title yet), which is exactly what the Final's text features will read.

# %% [markdown]
# ## 8. Conclusions, limitations and the plan for the Final
#
# **Answer to the question.** With the structured fields of a krisha.kz ad, tuned gradient boosting estimates the typical asking price of a comparable apartment with a **median error of 9.1 %** and a **MAE of 6.3 M KZT** on the saved test ads (bootstrap interval 8.3–9.9 % and 5.6–7.1 M), against 11.3 % and 8.7 M for the Midterm's linear model and 14.3 % and 9.9 M for the realtor's district rule. The registration goal of a median error under 10 % is reached. Measured against that estimate, **one test ad in ten asks more than 20 % above** what comparable apartments go for, equally often from owners and agents and a little more often in Almaty; one in twenty asks more than 20 % below. For a buyer, a +20 % deviation on an ordinary ad is a reason to negotiate; for a seller, the estimate is a defensible starting price; for a realtor, it is a reference that is wrong by less than 10 % for more than half of all ads.
#
# **What the Endterm added, in one line each.** *Ensembles* cut the error by 27 % and did so mostly in the expensive bands (MdAPE above 80 M KZT from 22–32 % to 13–17 %), the weakness the Midterm diagnosed. *Tuning* bought one fold SD for boosting and nothing for the forest; the data are not sensitive to the settings, many iterations with small leaves is all that matters. *Clustering* found that each city's market is one dense cloud, not islands: K-Means segments are a useful description (4 in Astana, 5 in Almaty, stable under resampling, differing by 1.3–1.4× in price per m² through age, height and place), DBSCAN's noise points are real unusual objects, and a segment feature adds nothing to the model. *PCA* showed that the structured fields carry about seven independent dimensions and that none of them is "price": every reduction costs accuracy. *The neural network* ties the linear model and loses to boosting by six fold SDs; on 3,852 tabular rows it is not worth it. *The importance analysis* says the model is not short of size or location signal; what is left is in the text.
#
# **Limitations.**
# 1. **Asking price ≠ sale price.** We model what sellers ask, not what buyers pay; the flag finds ads that are high *relative to other asking prices*, which is the right reference for negotiating but not a valuation.
# 2. **One snapshot (4–8 October 2026), one site.** Prices move with mortgage programmes and the exchange rate; a model from October is not valid in spring without re-scraping, and krisha.kz's audience (mostly agents) may differ from olx.kz's.
# 3. **A thin luxury tail.** 15 test ads above 150 M KZT; the model shrinks them towards the mass market by 14 %, and no amount of tuning changes that without more such ads.
# 4. **The test set has now been looked at three times** (Midterm, this notebook, and it will be at the Final). We chose every setting by CV and used it once per stage, but it is no longer a sealed benchmark; a fresh temporal hold-out (ads scraped a month later) is the honest next check.
# 5. **Tuning on the same folds as the comparison** makes the tuned models' CV scores slightly optimistic; the test column, where the tuned and the default boosting are 0.3 M apart, shows the size of that effect.
# 6. **The worst errors are text-only.** "Cash only", "assignment of rights", "needs repair", "designer duplex" are in the description, not in the fields; no structured model will fix them.
#
# **Plan for the Final.** (1) **Text:** TF-IDF of the sanitised description (prices and digits removed) fitted on the residuals of the boosting model, cross-fitted so that the text model never sees its own training residuals; the worst cases above are the test of whether it works. (2) **Photos:** embeddings of up to three listing photos from a pretrained CNN (the photo URLs are already collected), as a second residual correction, with the same honest check. (3) **A local app** that takes the fields, the text and the photos and returns the estimate, the ±interval from the segment's error, and the over-pricing flag with the "few comparables" warning. (4) **A temporal hold-out**: re-scrape in November and report the error on ads the model has never seen, which answers limitation 4. The split in `data/clean/split_ids.csv` stays unchanged, so the Final table will sit next to this one and the Midterm's.

# %% [markdown]
# ## 9. Contributions
# **Rakhat Balgabekov** — the randomised-search pipeline for the forest and the boosting model, the first version of the clustering (K-Means and DBSCAN) and the PCA diagnostics, the segment feature, the Endterm report tables.
# **Aitbek Kaiyrbekuly** — the Midterm fixes (phone-number scrubbing, feature list, CV-only model choice), the bias/variance and learning-rate experiments, the PyTorch network and its training loop, the per-city segmentation and the error analysis, the slides.
# Both: the readings under every table and plot, the conclusions and the plan for the Final were discussed and written together; both can explain every part of the notebook.
#
# AI assistance was limited to a final review of the completed work and presentation for correctness and academic English.
