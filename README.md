# US Accidents — Accident Distance Regression, Severity Classification & Clustering
**23CSE301 Machine Learning Capstone · Amrita Vishwa Vidyapeetham, Coimbatore · Review 2**
Team: Vibin Ragav S (CB.SC.U4CSE24556) · Shre Raam P J (CB.SC.U4CSE24548) · Nitin Pranav (CB.SC.U4CSE24539)

## 1. Overview and problem statement
Using nationwide US traffic-accident records we study three problems on the same dataset:

| Track | Task | Target | Algorithms | Review |
|---|---|---|---|---|
| Regression | Predict the road length affected by an accident | `Distance(mi)` | 10 | 1 |
| Classification | Predict the severity tier of an accident | `Severity` ∈ {1,2,3,4} | 10 (Part A: 5 in Review 1, Part B: 5 in Review 2) | 1 + 2 |
| Clustering | Discover groups of similar accident records **without labels** | none (unsupervised) | K-Means, Agglomerative | 2 |

All predictors are *pre-incident* attributes (time, location, weather, road infrastructure). Post-event fields are excluded to prevent leakage (Section 5).

## 2. Dataset
* File: `data/US_Accidents_Dec20_Updated.csv` — 42,768 rows × 47 columns (the *US Accidents* dataset as supplied by the course/team; we have no independent provenance beyond the file itself and its column names, which cover 2016–2020 records from 49 states).
* **Data-integrity note:** the dataset was supplied to us twice (an earlier copy of exactly 16 MiB and a later copy of 16,819,984 bytes). We compared them: both have the same 47 columns and 42,768 rows, the first 42,767 rows are identical, and the byte-size difference is only Windows (CRLF) versus Unix line endings. **In both copies the final record (`A-42768`) is cut off mid-row** (missing `Start_Lng`, all road flags, all weather fields, `Distance(mi)`). So the file is a 42,768-row extract with one incomplete last record; no complete version was available to us and we did not invent the missing values. The regression and classification notebooks keep the row and impute it (regression drops it because its target is missing); the clustering notebook removes it (42,767 records) because it needs complete location/time fields.
* Severity distribution: 1 → 1.0 %, 2 → 73.1 %, 3 → 21.8 %, 4 → 4.1 % (a majority-class predictor scores 73.1 % accuracy).

## 3. Repository layout
```
README.md   requirements.txt   .gitignore
data/US_Accidents_Dec20_Updated.csv
notebooks/regression.ipynb        # Review 1  – 10 regression algorithms
notebooks/classification.ipynb    # Review 1 (Part A) + Review 2 (Part B, 10-model table, tuning)
notebooks/clustering.ipynb        # Review 2  – K-Means + Agglomerative
models/                           # severity_classifier.joblib (tuned RF + fitted preprocessor, ~21 MB) is WRITTEN by classification.ipynb; not shipped in the archive (git-ignored)
results/*.csv|json, results/figures/*.png   # result tables + figures (CSV/JSON/clustering PNGs are written by the notebooks; the other PNGs are exported from the executed notebook outputs)
US_Accidents_ML_Review2.pptx      # Review 2 presentation (numbers cross-checked against results/)
Review2_Viva_Preparation.md       # viva study guide (items marked TEAM TO VALIDATE need your own check)
```

## 4. Setup and how to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace regression.ipynb
jupyter nbconvert --to notebook --execute --inplace classification.ipynb   # ≈ 15 min on 2 CPU cores (CV of 10 models + tuning)
jupyter nbconvert --to notebook --execute --inplace clustering.ipynb       # ≈ 5 min
```
Run notebooks from the `notebooks/` folder (they use the relative paths `../data`, `../results`, `../models`; nothing outside the project folder is used). Notebooks already contain their executed outputs. Run time on the 2-CPU machine used here: clustering ≈ 2 min, regression ≈ 4 min, classification ≈ 11 min.

## 5. Preprocessing and leakage prevention
| Excluded from features | Reason |
|---|---|
| `Severity`, `Distance(mi)` | targets (each excluded from the other track's features too) |
| `End_Lat`, `End_Lng`, `End_Time` | describe the end of the event — `Distance(mi)` is derived from the coordinates; end time is future information |
| `ID`, `Description` | identifier / post-incident free text |
| `Street`, `City`, `Zipcode`, `Airport_Code` | very high cardinality |
| `Country`, `Turning_Loop` | constant |
| `Number` | 65 % missing |

* Supervised tracks: 80/20 split (`stratify=y`, `random_state=42`) **before** any fitting; `SimpleImputer`, `StandardScaler`, `OneHotEncoder` are fitted on the training rows only (asserted in the classification notebook: the scaler's `n_samples_seen_` equals the training size). Cross-validation uses `Pipeline`s so preprocessing is re-fitted inside every fold. Final feature matrix: 67 columns.
* Clustering uses its own, separately justified 20-feature matrix (cyclic time encodings, standardised continuous features, binary road flags ≥ 1 % prevalence); labels are asserted absent from it and used only for post-hoc interpretation.
* Known, minor: in the Review 1 code the "top-15 weather conditions / top-20 states" lists are computed on all rows before the split (feature frequencies only, no labels). It was left unchanged to keep Review 1 results reproducible.

## 6. Results
### 6.0 Regression (Review 1 — unchanged; re-executed for Review 2)
`notebooks/regression.ipynb` was re-run top-to-bottom in the Review 2 environment with **no errors and identical metrics** to the stored Review 1 outputs (fit times differ by machine). Test set: 8,554 rows (dropping the one record with a missing target gives 42,767 rows).

| Rank | Model | R² | RMSE (mi) | MAE (mi) |
|---|---|---|---|---|
| 1 | Random Forest | 0.0485 | 1.5075 | 0.5322 |
| 2 | Gradient Boosting | 0.0417 | 1.5129 | 0.5290 |
| 3 | Polynomial (deg 2) | 0.0381 | 1.5157 | 0.5548 |
| 4 | Linear | 0.0370 | 1.5166 | 0.5521 |
| 5 | Ridge | 0.0370 | 1.5166 | 0.5520 |
| 6 | ElasticNet | 0.0326 | 1.5200 | 0.5438 |
| 7 | Lasso | 0.0292 | 1.5227 | 0.5444 |
| 8 | SVR | −0.0141 | 1.5563 | 0.4730 |
| 9 | KNN | −0.1277 | 1.6411 | 0.5761 |
| 10 | Decision Tree | −0.1703 | 1.6718 | 0.5405 |

Explained variance is very low for every model, and the 5-fold CV R² of the top two models is negative (RF −0.043, GB −0.104) — pre-incident features carry little signal about `Distance(mi)`.


### 6.1 Classification — all 10 algorithms (held-out test set, 8,554 rows)
Ranked by weighted F1 (tie-break: ROC-AUC OvR macro). Precision/Recall/F1 are support-weighted; ROC-AUC is One-vs-Rest, macro-averaged (SVC uses `decision_function` scores because it was trained without probabilities; every model has a real AUC).

| Rank | Model | Part | Accuracy | Precision (w) | Recall (w) | F1 (w) | ROC-AUC (OvR) | Fit time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | Random Forest Classifier | B | 0.7490 | 0.7160 | 0.7490 | 0.6953 | 0.8170 | 5.35 |
| 2 | Gradient Boosting Classifier | B | 0.7495 | 0.7198 | 0.7495 | 0.6807 | 0.8181 | 35.70 |
| 3 | MLP Classifier (Neural Net) | B | 0.7430 | 0.6997 | 0.7430 | 0.6790 | 0.7997 | 3.52 |
| 4 | Decision Tree Classifier | A | 0.7371 | 0.6791 | 0.7371 | 0.6747 | 0.7154 | 0.23 |
| 5 | KNN Classifier | A | 0.7180 | 0.6622 | 0.7180 | 0.6725 | 0.6889 | 0.01 |
| 6 | Bagging Classifier (DT base) | B | 0.7473 | 0.7184 | 0.7473 | 0.6707 | 0.7848 | 6.64 |
| 7 | Logistic Regression | A | 0.7336 | 0.6579 | 0.7336 | 0.6597 | 0.7745 | 4.22 |
| 8 | AdaBoost Classifier | B | 0.7312 | 0.6327 | 0.7312 | 0.6252 | 0.6981 | 4.47 |
| 9 | Support Vector Machine (SVC) | A | 0.6243 | 0.6156 | 0.6243 | 0.6171 | 0.6856 | 22.41 |
| 10 | Gaussian Naive Bayes | A | 0.1417 | 0.7257 | 0.1417 | 0.1608 | 0.6667 | 0.14 |

*Part A = Review 1 algorithms, Part B = Review 2 algorithms. Part-A numbers reproduce the Review 1 slides to the displayed precision; Logistic Regression differs in the 4th decimal from the Review 1 README (library-version effect).* 

**Model selection.** 5-fold stratified CV on the training set (preprocessing refitted per fold):

| CV rank | Model | CV weighted F1 (mean) | CV weighted F1 (std) | CV accuracy (mean) | CV time (s) |
|---|---|---|---|---|---|
| 1 | Random Forest Classifier | 0.6938 | 0.003 | 0.7474 | 22 |
| 2 | MLP Classifier (Neural Net) | 0.6825 | 0.0122 | 0.7376 | 13 |
| 3 | Gradient Boosting Classifier | 0.6766 | 0.0023 | 0.7453 | 142 |
| 4 | KNN Classifier | 0.6747 | 0.003 | 0.7191 | 3 |
| 5 | Decision Tree Classifier | 0.6724 | 0.0064 | 0.7289 | 1 |
| 6 | Bagging Classifier (DT base) | 0.6693 | 0.004 | 0.743 | 19 |
| 7 | Logistic Regression | 0.6587 | 0.003 | 0.7325 | 20 |
| 8 | AdaBoost Classifier | 0.6312 | 0.0107 | 0.7302 | 19 |
| 9 | Support Vector Machine (SVC) | 0.5799 | 0.0477 | 0.5655 | 121 |
| 10 | Gaussian Naive Bayes | 0.1418 | 0.0104 | 0.133 | 1 |

Random Forest ranks first in CV, so it was tuned with `RandomizedSearchCV` (12 of 240 sampled combinations, stratified 3-fold CV, scoring weighted F1, `random_state=42`). Best: `n_estimators=200, max_depth=30, min_samples_leaf=5, max_features=0.5`.

| Metric | Baseline RF | Tuned RF | Absolute change |
|---|---|---|---|
| Accuracy | 0.7490 | 0.7547 | +0.0057 |
| Precision (w) | 0.7160 | 0.7272 | +0.0112 |
| Recall (w) | 0.7490 | 0.7547 | +0.0057 |
| F1 (w) | 0.6953 | 0.7067 | +0.0114 |
| ROC-AUC (OvR) | 0.8170 | 0.8170 | -0.0000 |

* Weighted F1 improved by **+0.0114** (accuracy +0.0057) on the test set, evaluated once. The 95 % bootstrap interval for the F1 change over test rows is [+0.0049, +0.0179] (excludes 0; it reflects test-sample noise, not re-training variance). ROC-AUC did not change.
* Over-fitting: train–test weighted-F1 gap shrank from 0.305 (baseline) to 0.156 (tuned) — still over-fitted.
* **Weakness:** tuned recall per class is 0.105 / 0.952 / 0.256 / 0.040 (Severity 1–4); macro F1 = 0.367. Rare severities are barely detected.
* **SVC caveat:** the Review 1 SVC uses `max_iter=3000` and does not converge (verified: scikit-learn reports early termination), which partly explains its rank. Gaussian NB collapses (14 % accuracy) because its conditional-independence/Gaussian assumptions are violated.

### 6.2 Clustering (internal metrics only; labels unused in fitting)
K-Means: all 42,767 records, k = 6 (elbow by chord rule). Agglomerative (Ward): a fixed random sample of 8,000 records (seed 42, chosen without labels; the O(n²) pairwise matrix for all records is infeasible), k = 4 (largest dendrogram merge-height gap, k ≥ 3). "Control" rows refit at the other algorithm's k on the same 8,000 records.

| Algorithm | k | Fitted on | Evaluated on | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|---|---|
| K-Means | 6 | all 42,767 | all 42,767 | 0.1195 | 1.9289 | 4,345.0 |
| K-Means | 6 | all 42,767 | 8,000 subset | 0.1202 | 1.9183 | 835.9 |
| Agglomerative (Ward) | 4 | 8,000 subset | 8,000 subset | 0.1157 | 2.1792 | 755.6 |
| K-Means (control, matched k) | 4 | 8,000 subset | 8,000 subset | 0.1156 | 2.0979 | 911.0 |
| Agglomerative (control, matched k) | 6 | 8,000 subset | 8,000 subset | 0.0797 | 2.3037 | 661.6 |

Silhouette ↑ · Davies–Bouldin ↓ · Calinski–Harabasz ↑. Calinski–Harabasz grows with n, so full-data and 8,000-record values are not comparable.

Linkage comparison at k = 6 on the sample — single/average/complete score a high Silhouette only because they are degenerate (one cluster holds 95–99.8 % of points); Ward is balanced:

| Linkage | Silhouette | Davies-Bouldin | Calinski-Harabasz | Largest cluster % | Smallest cluster % | fit_s |
|---|---|---|---|---|---|---|
| ward | 0.0797 | 2.3037 | 661.6 | 29.5 | 4.15 | 4.3 |
| complete | 0.2362 | 1.5022 | 226.4 | 94.9 | 0.04 | 5.4 |
| average | 0.3514 | 0.9452 | 67.9 | 99.5 | 0.03 | 5.2 |
| single | 0.4684 | 0.6699 | 53.8 | 99.8 | 0.01 | 1.7 |

* Overall structure is **weak** (Silhouette ≈ 0.12): clusters overlap; single-initialisation K-Means runs agree with the final model only moderately (ARI 0.50–0.99).
* The K-Means clusters are mainly geo-climatic regimes (e.g. high-altitude Mountain West, California, cold Ohio-Valley, humid low-visibility central, warm Southern Plains, humid Southeast). Post-hoc, they are unrelated to `Severity` (ARI -0.0002).

### 6.3 Required visualisations
Elbow curve and Silhouette/Davies–Bouldin vs k; Ward dendrogram (truncated to the last 30 merges, with the chosen cut); PCA-2D scatter for each algorithm; t-SNE of 5,000 records coloured by each algorithm; classification: grouped metric bar chart, 10 confusion matrices, Random Forest importances, decision-tree plot, OvR ROC curves. Figures are embedded in the notebooks; clustering PNGs are also in `results/figures/`.

## 7. Reproducibility
`random_state=42` everywhere (splits, models, KMeans, sampling, t-SNE, bootstrap). Results above were produced with Python 3.13, scikit-learn 1.9.1, pandas 3.0, numpy 2.5 and are deterministic on the same versions; fit times vary by machine. Other scikit-learn versions can change results in the 3rd–4th decimal (Review 1 outputs were produced with a different version).

## 8. Limitations and known constraints
* Minority severity classes (1 and 4) are poorly detected; no class-weighting or threshold tuning was attempted.
* Tuning is deliberately small (12 random draws) to run on 2 CPU cores; a larger search might find more.
* Agglomerative clustering and the dendrogram use an 8,000-record sample; t-SNE uses 5,000. K-Means is full-data.
* Clustering features are a justified but not unique choice; t-SNE is qualitative only.
* The regression track is unchanged from Review 1 (R² is low for all models, cross-validated R² of the top two is negative — the signal for `Distance(mi)` in pre-incident features is weak).
* The earlier README's "25 / 25 marks" self-assessment table was removed: marks are awarded by the instructor, not verified by this repository.

## 9. Use of AI assistance (academic-integrity statement)
Generative-AI assistance (Anthropic's Claude) was used for **code scaffolding and drafting**: structuring the Part-B and clustering notebooks, helper functions, the README and slide layout. Per guideline 7.5, the analysis, interpretation and feature-engineering decisions must be the team's own: **team members must review, run and be able to explain every section before the viva, and should revise the written interpretations in their own words.** No code from external sources was copied; standard scikit-learn/SciPy APIs were used.

## 10. Validation record (what was actually executed)
All three notebooks were re-executed top-to-bottom (`jupyter nbconvert --execute --inplace`) from the packaged folder against the supplied `data/US_Accidents_Dec20_Updated.csv`:

| Notebook | Code cells executed | Errors | Figures | Notes |
|---|---|---|---|---|
| `regression.ipynb` | 20 / 20 | 0 | 6 | 10 models, tuning of 2 models, 5-fold CV of top 2, diagnostics |
| `classification.ipynb` | 23 / 23 | 0 | 9 (new Review 2 section) | 10 models, all 6 metrics, CV selection, tuning, saved model |
| `clustering.ipynb` | 21 / 21 | 0 | 7 | K-Means + Ward, 3 indices each, elbow, dendrogram, PCA ×2, t-SNE |

Every number in `results/` reproduced the previously stored values exactly (maximum absolute difference 0.0; only fit-time columns vary). The saved model was reloaded and re-scored on the held-out split (weighted F1 0.7067, accuracy 0.7547 — identical to the notebook). **Not validated:** other Python / scikit-learn versions, Windows/macOS, and the unknown full original dataset.
