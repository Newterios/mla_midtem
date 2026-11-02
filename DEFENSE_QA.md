# Endterm defense: likely questions and answers

Criterion 7 (15 points, per student) is scored on whether **each** partner can explain **any** part of the project, including the partner's part, and answer method questions correctly. Both of us read all of this. All numbers are from the executed notebook (`endterm/results_table.csv`, `endterm/summary.json`).

## A. The project in two sentences
We predict the asking price of an apartment in Astana or Almaty from the fields of its krisha.kz ad, on 4,815 ads we scraped ourselves, with a fixed test split of 963 ads saved at the Midterm. The Endterm adds tuned tree ensembles, a clustering of the market, PCA and a PyTorch network; the best model is tuned histogram gradient boosting on log(price) with a test MAE of 6.29 M KZT and a median error of 9.1 %, against 8.65 M / 11.3 % for the Midterm linear model.

## B. Questions about the workflow

**What did the Midterm feedback say, and what did you change because of it?**
82/100 and no written comments. We reviewed the Midterm against its rubric ourselves and took the weaknesses from its own error analysis; the change log in Section 2 lists six fixes with before/after evidence: the sharpened question, the contact-data scrub of the free text (7 handles masked), `has_detail` dropped (8.65 → 8.66 M, nothing lost), 10 wrong coordinates set to missing, model choice by CV only, and ensembles for the expensive bands (MdAPE above 80 M from 22–32 % to 13–17 %).


**Why MAE in KZT as the main metric, and why MdAPE too?**
The user thinks in money, so the error has to be in money; MAE is robust to the few luxury ads that dominate RMSE. MdAPE is there because our registration goal was a *relative* median error under 10 %, and because a 5 M error means something different on a 20 M flat and on a 200 M one.

**Why is the target modelled as log(price)?**
Prices are right-skewed (median 40 M, top 1 % above 190 M). On the log scale the relation with area is close to linear, errors become relative, and a model does not spend its capacity on a few expensive ads. Predictions are transformed back with exp before any metric is computed, so every number is in KZT.

**What does cross-validation estimate, and why 5 folds on the training rows only?**
It estimates how the model would do on ads it has not seen, using five different held-out fifths of the training set; the mean is the estimate, the fold SD is its uncertainty. The test rows stay untouched so that one final number is an honest check. Every choice (hyperparameters, k for K-Means, the number of PCA components, the network size) was made on CV, never on the test set. The Midterm picked the "best" model by test MAE; that was a slip and is fixed.

**What does the pipeline protect against?**
Leakage from preprocessing. Imputation medians, the scaler, the one-hot vocabulary, PCA and the K-Means segment feature are all fitted on the training part of each fold only, so no information from the validation fold or the test set reaches the model. Fitting the scaler on all rows before splitting would make the CV score slightly optimistic in a way we could not measure.

**Why the same split as at the Midterm?**
So that the three stages are comparable: the Endterm numbers can be set next to the Midterm table ad by ad (Section 7.3 does exactly that), and the Final will be compared with both.

**What was wrong with `has_detail`?**
It records whether our scraper managed to open the ad page or only the search card. That is a property of our data collection, not of the apartment; a new ad would always have it equal to True. The Midterm excluded `is_hot` for the same reason and should have excluded this one too.

**The Midterm notebook says 8.65 M for the linear model, your table says 8.66. Which is right?**
Both. 8.65 M (MdAPE 11.3 %) is what the Midterm notebook reported. In the Endterm the same linear model on log price is re-run on the same split with the Endterm feature list: `has_detail` dropped and 10 ads with coordinates in the wrong city set to missing. That moves it to 8.66 M and 11.27 %, a hundredth of a million, and this re-run is the Midterm reference in every Endterm table, so that all models are compared on identical inputs.

## C. Ensembles and tuning

**What is the difference between bagging (random forest) and boosting?**
A forest trains many deep trees on bootstrap samples with random feature subsets and averages them: each tree has low bias and high variance, the average keeps the low bias and cancels the variance. Boosting trains shallow trees one after another, each on the residuals of the sum so far, scaled by the learning rate: each tree has high bias, the sum removes it. In our table the fully grown tree has training MAE near zero and the largest fold SD (variance); the forest cuts the SD, boosting cuts the error further.

**Why does a smaller learning rate need more trees? Why can too many trees hurt?**
Each tree moves the prediction by learning_rate × its own fit, so a step of 0.03 needs roughly ten times the trees of a step of 0.3 to cover the same distance. With a large rate the model overshoots and starts fitting noise in the residuals, and the validation MAE rises again; with a small rate and early stopping it does not.

**What is histogram gradient boosting?**
The same algorithm with the continuous inputs binned into at most 255 bins, which makes finding a split much faster and acts as a mild regulariser. It also handles missing values natively (we still impute, so that the forest and the linear models see the same matrix).

**How did you tune, and how do you know the gain is real?**
`RandomizedSearchCV` with 25 (forest), 40 (histogram boosting), 20 (XGBoost) and 20 (LightGBM) random settings on the same five folds, scored by MAE. The gain over the default setting is compared with the fold SD: a gain smaller than the SD is inside the noise. Forest: best 7.99 ± 0.38 M against 8.00 M for the default, nothing. Boosting: best 6.62 ± 0.35 M against 7.01 M for the default, a gain of 0.39 M, about one fold SD; the five best settings are within 0.11 M of each other and all use many iterations (600–800) and small leaves (5–10 rows).

**What is OOB error?**
Every tree of the forest is trained on a bootstrap sample that leaves out about a third of the rows; predicting each row only with the trees that never saw it gives a validation estimate without a split. Our OOB MAE agrees with the 5-fold CV MAE.

**Impurity importance vs permutation importance?**
Impurity importance adds up how much each split on a feature reduced the squared error in the forest; it favours features with many possible split points (continuous ones, high-cardinality one-hot groups). Permutation importance shuffles one column and measures how much the MAE of the fitted model grows; it is model-agnostic and in the units of the metric. Both rank area and location first.

**Why does the blend (forest + boosting) help or not?**
A blend helps only when the models make different mistakes. The blend scores 6.95 ± 0.33 M, between the forest (7.99) and the boosting (6.62) and worse than boosting alone: the forest is 1.4 M weaker and fails on the same ads, so the average is pulled down.

**Why did you keep scikit-learn's boosting when XGBoost is the lecture's model?**
Because the choice is made by CV MAE and scikit-learn's tuned histogram boosting is first (6.62 M) by 0.02 M over XGBoost (6.64) and 0.13 M over LightGBM (6.75), all inside one fold SD. XGBoost and LightGBM were tuned the same way (20 random settings each, same folds) and are in the final table; the point of running three implementations is that they agree, so the 6.6 M floor is set by the data, not by the library. On the test column LightGBM is even a little better (MAE 6.19 against 6.29 M, R² 0.877 against 0.855); switching to it would be choosing on the test set, the Midterm's slip, and the difference is far inside the bootstrap interval (5.56–7.14 M).

## D. Clustering and PCA

**Why cluster within each city, and what were the inputs?**
With both cities in one run the distance sees the 1,000 km between them first and returns "Astana" and "Almaty". Inputs: log area, rooms, floor, floors in the building, year built, latitude, longitude, standardised. The price is not an input; it is looked at afterwards to see what the segments cost.

**How did you choose k?**
Three measures: inertia (the elbow is where adding a cluster stops reducing the within-cluster sum of squares much), silhouette (how much closer each ad is to its own cluster than to the nearest other one, from −1 to 1), and stability (adjusted Rand index between the full fit and fits on 80 % subsamples: if the clusters change when a fifth of the rows is removed, they are not real). We took the k with the best silhouette among the stable ones. Astana k = 4 (silhouette 0.206, stability 0.97; k = 3 had a slightly higher silhouette but a stability of 0.81 with a worst subsample of 0.55), Almaty k = 5 (silhouette 0.250, stability 0.97). Segments: Astana mass market (45 m², 2024, 659 k/m²), large family flats (92 m², 667 k), tall-tower flats (floor 13 of 18), old low-rise (2006, 477 k/m², the cheapest); Almaty premium high-rise (848 k/m²), club-house flats (87 m², 780 k), Soviet centre (1976, 766 k), small new west (722 k), cheap north-east (632 k).

**What is a silhouette of 0.3? Is that good?**
It says the clusters overlap: for a typical ad the nearest other cluster is only moderately farther than its own. Apartment markets are continuous, so cuts through a cloud are expected; the segments are still useful as a description because they are stable and differ in the medians.

**What does DBSCAN add? What is eps, what is min_samples?**
DBSCAN groups points that have at least `min_samples` neighbours within radius `eps` and marks the rest as noise; it does not need k and finds dense groups of any shape. We set min_samples = 14 (twice the number of inputs) and read eps from the knee of the sorted 14th-neighbour distance. Its noise points are the ads with no dense group of similar apartments: a townhouse listed as a flat, a 1-room flat on the 30th floor of a building from 1965, and so on. We use them as a "no similar ads, wide error" warning in the buyer's flag.

**Why not K-Means on price per m²?**
Because price per m² is derived from the target. Clusters built with it would describe the price, not the apartments, and a segment feature built that way would leak the target into the model.

**What do PC1 and PC2 mean?**
PC1 (27.8 % of the variance): total floors 0.41, latitude 0.40, year 0.34, floor 0.34, longitude −0.40, i.e. new high-rise in Astana against older low-rise in Almaty. PC2 (21.8 %): area 0.50, rooms 0.52, photos 0.35, i.e. size. Five components keep 80 %, seven keep 90 %; the tenth carries 0 % because latitude and longitude are anti-correlated across the two cities. The first components are size (area, rooms, kitchen) and the building (year, floors, ceiling), and location comes as its own direction because latitude and longitude are not correlated with size.

**Why does PCA not help the model?**
PCA keeps the directions of largest variance in the inputs, not the directions that predict the price. KNN on the components is as bad as KNN on all one-hot columns; the trees split on raw columns and do not care about rotations. PCA is a good description of the table and a poor preprocessing step here. KNN: 11.13 M on the raw 71 columns, 11.48 M on the components that keep 90 % of the variance, 12.44 M on two components. Ridge: 9.10 M raw, 10.61 M with 90 % kept, 11.32 M with 10 components.

**Did the segment feature help?**
6.65 ± 0.31 M with the in-fold K-Means segment against 6.62 ± 0.35 M without: a tenth of the fold SD, nothing. The boosting model already sees the same seven inputs and can carve the same regions itself, so a cluster id adds nothing it did not know.

## E. The neural network

**Describe the network and the training.**
Inputs: the same standardised and one-hot encoded matrix as the linear models (71 columns). Layers: 64 units in one hidden layer (best), also 128-64 and 256-128-64 with ReLU and dropout, one linear output. Target: standardised log(price). Loss: L1 (absolute error on the log scale ≈ relative error). Optimiser: Adam, learning rate 1e-3, weight decay 1e-4, mini-batches of 128, up to 300 epochs, early stopping with patience 30 on 15 % of the training fold. Cross-validated on the same five folds as everything else.

**What is backpropagation / what does `loss.backward()` do?**
It computes the gradient of the loss with respect to every weight by the chain rule, layer by layer from the output back to the input; `optimizer.step()` then moves each weight against its gradient. Adam scales each weight's step by a running estimate of the gradient's mean and variance, so the learning rate does not have to be tuned per layer.

**Why early stopping and dropout?**
With 3,000 training rows and tens of thousands of weights the network can memorise. Early stopping stops when the inner validation error stops improving; dropout randomly switches off units during training so that no unit can rely on another; weight decay shrinks the weights. Without them the training loss keeps falling while the validation MAE rises (visible in the curves).

**What happened with learning rate 0.05?**
CV MAE 9.43 ± 0.55 M, worse than the network with no hidden layer (9.26): the loss jumps, the validation curve is noisy and worse, and early stopping fires on a bad epoch. The learning rate is the one setting that can make a network not learn at all.

**Why does the network lose to boosting?**
Tabular data with 3,852 rows, 24 columns, a few strong non-linear effects and interactions: trees find axis-aligned splits (area > 80, year > 2018, district = X) directly, a network has to approximate them with smooth functions from few examples. The lecture's lesson holds: a network with many parameters can do no better than a simpler model when the data are small and tabular. Best network 8.82 ± 0.60 M (64 units), scikit-learn MLP 8.86 M, Midterm linear 9.10 M, tuned boosting 6.62 ± 0.35 M: the networks are 2.2 M behind, about six fold SDs, on every fold.

**Why ReLU and not sigmoid? Did you check the vanishing gradient?**
Yes, with the Practice 8A measurement: untrained networks of 1, 3 and 6 hidden layers of 64 units, one `backward()` on the inner-training rows of fold 1, gradient norm of the first hidden layer divided by that of the last. ReLU: 1.10 for three layers, 0.12 for six. Sigmoid: 0.031 for three layers, 5.5 × 10⁻⁵ for six. The sigmoid's slope is at most 0.25, so each layer multiplies the gradient by about a fifth; an active ReLU unit passes it on unchanged. Trained on fold 1, the ReLU networks peak after 25–30 epochs, the shallow sigmoid networks need 264–285 epochs for a similar MAE, and the six-layer sigmoid network never leaves the mean (inner-validation MAE 20.3 M, the dummy's level). That is why our networks use ReLU and few layers. Table: `endterm/nn_gradient_flow.csv`.

**Why not a deeper network?**
Three reasons from the evidence: 64, 128-64 and 256-128-64 units land within 0.1 M of each other (8.82–8.90 M), so depth buys nothing here; the training loss keeps falling while the validation MAE stalls after 10–30 epochs, so a deeper network only memorises faster; and 3,270 inner-training rows cannot support 60,000 parameters. The gradient-flow table adds that depth is also where training itself gets fragile (the six-layer sigmoid network does not learn).

**Which parts of Practice 8 did you reuse, and which not?**
Reused: the PyTorch training loop (`nn.Sequential`, mini-batches, a fixed seed, a validation curve per epoch), the too-high learning rate as a failure case, the network with no hidden layer as the linear check, the gradient-flow measurement with depth and the two activations, cross-validated classical baselines and one test at the end. Not reused: backpropagation in NumPy with a gradient check, the zero/constant initialisations, weight images and ablation of hidden units, and everything from Practice 8B (convolutions, CNNs, MobileNetV2). Those are exercises on digit and clothing images; our inputs are 24 tabular fields and there is nothing to convolve. The Final, with photos and text, is where that part comes in.

**Why a linear network with no hidden layer?**
As a check: it is the Midterm linear model trained by gradient descent, and it reaches about the same MAE, which shows the training loop is correct before we trust the deeper networks.

## F. Results and errors

**Where do the text and the photos for the Final come from, and may you use them?**
From the same ad pages we already scraped in October: the description and the photo URLs are stored in the raw snapshot under the same ad id as the structured fields, so they attach to the same 4,815 objects and the same saved split without new collection. The ad pages are public, need no login and are allowed by krisha.kz's `robots.txt` (checked 4 Oct 2026, see the data card). The description goes through the contact-data scrub first; photos are downloaded only to compute embeddings and are not stored or shared. A source stays in the Final model only if it lowers the CV MAE by more than one fold SD, about 0.35 M.


**What is the answer to the question?**
Structured fields give a median error of 9.1 % and a MAE of 6.3 M KZT on the saved test ads (Midterm: 11.3 %, 8.7 M; realtor rule: 14.3 %, 9.9 M), so the 10 % goal is reached. Measured against that estimate, 93 of the 963 test ads (10 %) ask more than 20 % above what comparable apartments go for, owners and agents equally often, a little more often in Almaty (11 %) than in Astana (8 %); 5 % ask more than 20 % below. The error grew with price at the Midterm and still does, but far less: MdAPE 12.9 % at 80–150 M (was 21.9 %) and 16.6 % above 150 M (was 31.8 %).

**Where does the model fail, and why?**
Expensive and large apartments (few training ads, quality not in the fields), and ads whose price is explained only in the text: "cash only", "documents at the end of the year", assignment sales, a townhouse listed as a flat. The ensembles halve the error of the mass market and reduce the luxury error, but the text-explained cases remain, which is the Final's job (TF-IDF on the description, photos).

**How sure are you of the test number?**
A 2,000-sample bootstrap of the 963 test ads gives a 95 % interval of 5.56–7.14 M KZT for the MAE and 8.31–9.88 % for the MdAPE for the MAE. The interval covers the test draw only; retraining on a new snapshot or a different month would move the number more.

**What would you do with more time?**
Text features from the description (planned for the Final), a temporal hold-out (re-scrape a month later and test on it), quantile regression to give the buyer an interval instead of a point, and a separate luxury model or sample weights for the thin tail.

## Defense plan: 7 minutes of slides, 5 of questions
Both partners speak; either may be asked about any part. Roughly 40 seconds per slide.

| Slides | Speaker | Time |
|---|---|---|
| 1–3 question, what changed, bias/variance of tree / forest / boosting | Aitbek | 2:00 |
| 4–6 tuning and importance, K-Means segments, DBSCAN and PCA | Rakhat | 2:30 |
| 7–8 the network, the results table and the one test | Aitbek | 1:30 |
| 9 error analysis and the over-pricing flag | Rakhat | 0:40 |
| 10 conclusions, limitations, plan for the Final | Aitbek | 0:20 |

Hand-over lines: after slide 3 "Rakhat tuned these and looked for structure in the data", after slide 6 "Aitbek checked whether a network does better", after slide 8 "Rakhat looked at where the model is wrong", after slide 9 "one slide to close".

## G. Partner's part, in one breath each
- *Rakhat on the network:* "Same encoded inputs as the linear model, standardised log price, L1 loss, Adam, early stopping on an inner split, 5-fold CV; the one-hidden-layer net ties the linear model, deeper nets are a little better but all lose to tuned boosting because the data are small and tabular."
- *Aitbek on the search:* "RandomizedSearchCV with 25 forest, 40 boosting, 20 XGBoost and 20 LightGBM settings on the same five folds, MAE as the refit metric, train scores recorded to see memorising; best boosting setting learning rate 0.12, 7 leaves, 600 iterations, 5 rows per leaf, no L2 penalty; the gain over default is compared with the fold SD."

## H. Terms from the lectures the examiner may use

**Bagging vs boosting in one sentence each (Lecture 5).** Bagging = bootstrap samples + random feature subsets at each split, deep trees, predictions averaged (regression) or voted (classification); boosting = an additive model where each new weak learner is fitted to the negative gradient of the loss (the residuals for squared error), scaled by the learning rate.

**What does XGBoost add to plain gradient boosting?** A regularised objective (L2 penalty on the leaf values, a penalty per leaf), second-order gradients in the split criterion, tree pruning, sparsity-aware split finding (a default direction for missing values), column and row subsampling, and a fast histogram implementation. scikit-learn's HistGradientBoosting has the histogram part, native missing values and an L2 penalty; LightGBM grows trees leaf-wise instead of level-wise. In our table all three land within a few tenths of a million of each other, which is what one expects: same algorithm family, same data. Our numbers: scikit-learn 6.62 ± 0.35, XGBoost 6.64 ± 0.32, LightGBM 6.75 ± 0.28 M KZT, a spread of 0.13 M, a third of a fold SD.

**Grid search vs random search.** Grid search tries every combination of a parameter dictionary; random search draws settings at random for a fixed budget. We used random search because the boosting space has six dimensions (a full grid would be 6 × 5 × 4 × 5 × 4 × 2 = 4,800 fits × 5 folds) and because random search covers each dimension with more distinct values for the same budget.

**Why not accuracy or ROC-AUC?** Those are classification metrics; our task is regression. The regression metrics are MAE, RMSE, MdAPE and R²; MAE is the main one because the user thinks in money and the luxury ads dominate RMSE.

**K-Means in the lecture's words (Lecture 6).** A centroid-based algorithm: pick k, place k centroids, assign each ad to its nearest centroid (Euclidean distance on the standardised inputs), move each centroid to the mean of its ads, repeat until the assignments stop changing. It minimises the within-cluster sum of squares (WCSS, inertia). Scaling is essential because the distance would otherwise be dominated by the field with the largest units (year built), and outliers pull the centroids, which is why the ads with coordinates in the wrong city had to be fixed first.

**Elbow vs silhouette vs stability.** The elbow (WCSS vs k) looks for the k after which inertia stops falling fast; in our data it falls smoothly with no elbow. The silhouette compares each ad's mean distance to its own cluster with the distance to the nearest other cluster, from −1 to 1; ours are 0.21–0.25, "clusters are not far apart". Stability (adjusted Rand index under resampling) is not in the lecture; it comes from Practice 6A and is what separated k = 4 from k = 3 in Astana.

**DBSCAN's three kinds of points.** Core: at least minPts neighbours within eps. Border: within eps of a core point but not core itself. Noise: neither. Density reachability chains core points into one cluster, which is why DBSCAN finds clusters of any shape and does not need k; its weakness is one eps for the whole data set, which is why a continuous cloud with a dense centre and sparse edges becomes one cluster plus noise.

**PCA in the lecture's steps (Lecture 7).** Standardise, compute the covariance matrix, take its eigenvectors (the components, orthogonal, therefore uncorrelated, which removes multicollinearity) and eigenvalues (the variance along each), keep the components that reach the chosen share of variance (we report 80, 90 and 95 %: 5, 7 and 8 of 10), project the data on them. Limitations the lecture names and we saw: the components are not interpretable by themselves (we interpret them through loadings) and PCA keeps the global shape, not the directions that matter for a target.

**Vanishing gradient (Lecture 8).** In a deep network the gradient reaching the first layers is a product of one factor per layer; with sigmoid each factor is at most 0.25 times the weights, so after a few layers the gradient is near zero and the first layers stop learning. Ours: first-to-last gradient-norm ratio 5.5 × 10⁻⁵ for six sigmoid layers against 0.12 for six ReLU layers; the six-layer sigmoid network does not learn.

**Forward pass, backward pass (Lecture 8).** Forward: each layer computes z = W·a + b and a = f(z) with a non-linear activation (ReLU in our network); the output layer gives the prediction, the loss compares it with the target. Backward: the chain rule gives ∂L/∂W for every layer from the output back to the input; the optimiser updates W ← W − η·∂L/∂W. Vanishing gradients: with sigmoid or tanh the derivative is at most 0.25, so in deep networks the product of many such factors makes the early layers' gradients tiny; ReLU's derivative is 1 for active units, which is why we use it. Overfitting is handled by early stopping, dropout and weight decay.

**What is the learning rate, and what happened at 0.05?** The step size of each weight update. Too small: slow; too large: the loss oscillates and never settles, which is exactly the fifth training curve (validation MAE jumping between 8.5 and 10.5 M) and the worst CV score of the five networks.
