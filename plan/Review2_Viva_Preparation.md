# Review 2 — Viva Preparation Guide
*23CSE301 Machine Learning Capstone · US Accidents*

**How to use this file.** It lists questions an instructor is likely to ask, with a *short answer framework* (what to say, in what order) and the numbers from **our own executed notebooks** (`results/`). It is a study aid, **not a script to recite**. The course guidelines (7.5) say the analysis and interpretation must be the team's own, so every item marked **⚠ TEAM TO VALIDATE** is a claim we derived from the results that *each of you must check against the notebook output and be able to defend or revise in your own words*. Where a number is quoted, open the notebook and find it before the viva.

---
## 0. Team speaking guide
| Part | Suggested time | Owner (assign in your team) | Must be able to explain line-by-line |
|---|---|---|---|
| Problem, data, leakage, preprocessing (slides 2–3) | 2 min | ______ | split-before-fit, what is excluded and why |
| Regression recap (Review 1) | 1 min | ______ | R², RMSE, MAE, why R² is low |
| Classification Part B + 10-model table + metrics (slides 4–8) | 4 min | ______ | each of 10 models in one sentence; weighted F1; OvR AUC |
| Model selection + tuning (slides 9–11) | 3 min | ______ | CV vs test, RandomizedSearchCV, bootstrap CI, over-fitting |
| Clustering (slides 12–18) | 4 min | ______ | elbow rule, linkage, 3 indices, PCA, t-SNE, sampling |
| Limitations + conclusions (slides 19–20) | 1 min | ______ | what does *not* work and why |

Guideline 7.4: **any** member can be asked about **any** part. Everyone should read every section of all three notebooks at least once and *run* them.

Delivery tips: state the question first, then the method, then the number, then the caveat. Say "in our experiment" rather than "in general". If you do not know, say what you would check (e.g., "I would look at the confusion matrix").

---
## 1. Data, leakage and preprocessing
**Q1. Why did you remove `End_Lat`, `End_Lng`, `End_Time`?**
Framework: define leakage → they are only known *after* the accident → `Distance(mi)` is computed from start/end coordinates, so using them would give the answer away → a real deployment could not use them.

**Q2. How do you prevent leakage in preprocessing?**
Split first (80/20, stratified, `random_state=42`) → fit imputer/scaler/one-hot encoder on the training rows only → transform the test rows → inside cross-validation use a `Pipeline` so each fold re-fits preprocessing. In the notebook we *assert* the scaler saw exactly the training rows (34,214).

**Q3. Why stratified split? Why 80/20?**
Severity is imbalanced (1.0 / 73.1 / 21.8 / 4.1 %); stratification keeps these shares in both partitions so rare classes appear in the test set (86 Severity-1 rows). 80/20 is the guideline's typical split.

**Q4. The dataset file has a cut-off last record — what did you do?**
The final row (`A-42768`) is incomplete in the supplied CSV (same in both copies we received). Classification/regression keep it and impute (1 row of 42,768, negligible); clustering removes it (needs complete location/time). We did not invent the missing values.

**Q5. What is an engineered feature and why?** `Start_Hour`, `Day_of_Week`, `Month` from `Start_Time` (rush hours, weekday vs weekend, seasons). For clustering we used cyclic sin/cos encodings so hour 23 and 0 are neighbours.

---
## 2. Classification algorithms (be ready for one-sentence answers on all 10)
| # | Model | One-sentence answer | Hyper-parameter / note |
|---|---|---|---|
| 1 | Logistic Regression | Linear model; softmax over classes; coefficients → log-odds | baseline; `max_iter=1000` |
| 2 | KNN | Majority vote of the *k* nearest points; needs scaling | k = 7; distance metric = Euclidean |
| 3 | Gaussian NB | Bayes rule + *conditional independence* + Gaussian likelihoods | worst: 14.2 % accuracy; assumptions violated (correlated coordinates/weather, boolean features) |
| 4 | Decision Tree | Recursive axis-aligned splits minimising impurity | `max_depth=10` |
| 5 | SVC (RBF) | Max-margin boundary in kernel space; needs scaling | **capped at `max_iter=3000` → not converged** (be upfront) |
| 6 | Random Forest | Many de-correlated trees on bootstrap samples + random feature subsets; average votes | 200 trees; best test F1 |
| 7 | AdaBoost | Sequential weak learners (stumps); re-weights mistakes | 100 stumps |
| 8 | Gradient Boosting | Sequential trees fitted to the gradient (residual) of the loss | 100 trees, depth 3, lr 0.1 |
| 9 | Bagging (DT base) | Bootstrap-aggregate depth-10 trees; reduces variance | 50 trees |
| 10 | MLP | Feed-forward net (64, 32), ReLU, Adam, early stopping | scaling essential |

**Q6. Bagging vs Random Forest?** Both bootstrap rows; RF *also* samples a random subset of features at every split, which de-correlates trees. **Q7. Bagging vs Boosting?** Bagging = parallel, variance reduction; boosting = sequential, bias reduction, can over-fit noise.
**Q8. Why does Naive Bayes fail here?** Independence assumption violated; Gaussian assumption on 0/1 and one-hot columns; it also over-predicts minority classes. ⚠ TEAM TO VALIDATE: confirm this explanation from the confusion matrix (it predicts Severity 1 for 3397 of the Severity-2 test rows).
**Q9. Why is SVC so low?** Check `n_iter_`: 4 of 6 internal solvers hit the 3,000-iteration cap, so it did not converge. ⚠ TEAM TO VALIDATE: re-fit with a higher `max_iter` (or on a subsample) and see whether the rank changes — we did not run that experiment.

---
## 3. Metrics
**Q10. Why not just accuracy?** A model that always predicts Severity 2 scores 73.1 %. Our best accuracy is 74.9 %, i.e. only ~2 points above that, so we rank on **weighted F1** and show confusion matrices.

**Q11. Define precision, recall, F1.** Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = harmonic mean. *Weighted* average = mean over classes weighted by class support (so it is dominated by Severity 2). Macro would treat the four classes equally.

**Q12. How is multiclass ROC-AUC computed?** One-vs-Rest: for each severity class *k*, treat it as positive and the rest as negative, compute a binary AUC from that class's score, then average (we report macro; weighted is in `classification_per_class_auc.csv`). AUC = probability a random positive scores higher than a random negative; it is threshold-free and rank-based.

**Q13. SVC has no `predict_proba` — how did you get AUC?** Used `decision_function` scores; AUC only needs ranking. We compute per-class binary AUCs ourselves because `roc_auc_score(multi_class='ovr')` insists on probabilities summing to 1, and checked our helper equals scikit-learn's for the 9 probabilistic models.

**Q14. Why can accuracy be high but macro F1 low?** Tuned RF: accuracy 75.5 %, per-class recall 0.105 / 0.952 / 0.256 / 0.040; macro F1 0.367. The model is good at the majority class and poor on rare severities. ⚠ TEAM TO VALIDATE against the `classification_report` cell.

**Q15. Read a confusion matrix.** Rows = true class, columns = predicted; diagonal = correct; off-diagonal = confusions. Nearly every model sends most Severity 3 and 4 cases to Severity 2.

---
## 4. Model selection and tuning
**Q16. How did you pick the final model?** 5-fold stratified CV on the **training** set for all 10 models (preprocessing re-fitted per fold). Highest CV weighted F1: **Random Forest Classifier** (1.0000 ± 0.6938). The test set was not used to choose it. It is also first on the test table (F1 0.6953); Gradient Boosting has a slightly higher ROC-AUC (0.8181 vs 0.8170).

**Q17. What did tuning do?** RandomizedSearchCV, 12 of 240 combinations, 3-fold CV, scoring weighted F1. Best: n_estimators 200, max_depth 30, min_samples_leaf 5, max_features 0.5. Test weighted F1 0.6953 → 0.7067 (**+0.0114**), accuracy +0.0057, ROC-AUC unchanged.

**Q18. Is the improvement real?** Bootstrap 95 % CI of the F1 change over test rows = [+0.0049, +0.0179], excluding 0. Caveat: it only reflects test-sample noise (the model is fixed), not training randomness; the gain is small (~1 point). ⚠ TEAM TO VALIDATE how you would phrase this.

**Q19. Why RandomizedSearch instead of GridSearch?** 240 combinations × 3 folds of a 200-tree forest is expensive on 2 CPU cores; random search samples the space and usually finds a near-best setting with far fewer fits. A bigger budget might do better.

**Q20. Over-fitting?** Baseline RF: train F1 1.000 vs test 0.695; tuned: gap 0.305 → 0.156 — still over-fits. Regularising via `min_samples_leaf` and `max_features` shrank it.

**Q21. What would improve minority-class detection?** Ideas (not implemented): `class_weight='balanced'`, threshold tuning on predicted probabilities (per-class ROC-AUC is 0.76–0.94, so the scores carry signal), resampling/SMOTE inside CV. ⚠ TEAM TO VALIDATE by trying at least one.

---
## 5. Clustering
**Q22. Supervised vs unsupervised?** Clustering never sees a target; it groups by feature similarity. We excluded `Severity` and `Distance(mi)` from features, from choosing *k*, and from sampling; they are used only after fitting for interpretation (guidelines 3.3).

**Q23. How did you choose k for K-Means?** Elbow: inertia vs k (2–12) on all 42,767 records; a smooth curve with no sharp bend, so we used an objective rule (farthest point from the curve's chord) → k = 6. Cross-check: silhouette at k=6 = 0.120 (max 0.121). ⚠ TEAM TO VALIDATE: be honest that the elbow is weak.

**Q24. What do the three indices mean?** Silhouette (−1..1, higher better): closeness to own cluster vs nearest other. Davies–Bouldin (≥0, lower better): average worst-case ratio of within-cluster scatter to between-centre distance. Calinski–Harabasz (higher better): between- over within-cluster dispersion; grows with *n* and *k*. Do not compare magnitudes across indices.

**Q25. Your results?** K-Means (k=6, all records): Silhouette 0.1195, DB 1.929, CH 4,345. Ward (k=4, 8,000-record sample): 0.1157, 2.179, 756. A silhouette near 0.12 means **weak, overlapping structure**. CH values on 42,767 vs 8,000 rows are not comparable.

**Q26. Why did single/average linkage have a higher silhouette but you chose Ward?** They are degenerate: one cluster holds 95–99.8 % of points and the rest are outliers (chaining), which inflates silhouette. Ward minimises within-cluster variance and gives balanced clusters (4.2 % smallest).

**Q27. How did you read the dendrogram / choose 4 clusters?** Look for the largest vertical gap between successive merge heights (excluding the trivial k=2); cutting there gives k = 4. ⚠ TEAM TO VALIDATE: the gap at k=4 (18.9) is close to the gap at k=2 (19.0) — the choice is not decisive.

**Q28. Why sample for hierarchical clustering?** Agglomerative needs the full pairwise distance matrix: O(n²) memory and O(n²)–O(n³) time; for 42,767 points that is ≈ 7 GB. We used a fixed random sample of 8,000 (seed 42, drawn without labels), and evaluated K-Means on the same 8,000 for a like-for-like comparison. K-Means itself is ~O(n·k·d·iterations) and ran on all records. Silhouette is also O(n²), so on the sweep it used a 10,000-row subsample; the final K-Means silhouette is exact.

**Q29. What is PCA doing in your plots?** Projects the scaled 20-D matrix onto the two orthogonal directions of maximum variance (fitted on all records). PC1+PC2 explain only 32.0 % of variance, so overlap in the 2-D picture partly reflects information lost by the projection.

**Q30. t-SNE vs PCA?** PCA is linear and preserves global variance; t-SNE is non-linear and preserves local neighbourhoods (perplexity 30, seed 42, 5,000 points). Distances/cluster sizes in t-SNE are not meaningful; use it only to see local coherence. It is stochastic and expensive (O(n²) naïve).

**Q31. What do your clusters mean?** From the profile table they look like geographic/climatic regimes (e.g. high-altitude Mountain West with pressure ≈ 25.6 inHg). ARI vs Severity ≈ -0.0002, i.e. no relationship. ⚠ TEAM TO VALIDATE the region labels by plotting each cluster's latitude/longitude on a map — the names in the slides are our reading of the mean values, not a tested fact.

**Q32. Are the K-Means clusters stable?** Single-initialisation runs agree with the final model only moderately (ARI 0.50–0.99); the final fit uses `n_init=20`. This supports the "weak structure" conclusion.

---
## 6. Limitations (say them before you are asked)
* Minority severities (1 and 4) are rarely detected; weighted F1 hides this.
* SVC (Review 1 setting) did not converge; its rank is not a fair verdict on RBF-SVMs.
* Tuning budget small (12 random draws).
* Hierarchical clustering and its metrics use an 8,000-record sample; t-SNE uses 5,000.
* Weak cluster structure (silhouette ≈ 0.12); features are one justified choice among many.
* Dataset: an extract with 42,768 rows whose final record is truncated; we have no information on how it was sampled from the original *US Accidents* data, so conclusions apply to this extract only.
* Regression: R² ≤ 0.05 for all models; CV R² of the best two is negative.
* No GUI/deployment (bonus not attempted).

## 7. Quick-fire checks (be able to answer without notes)
1. Which five algorithms are Part A, which five Part B? 2. What does `stratify=y` do? 3. Why `random_state=42`? 4. What would happen if the scaler were fit on all data? (test information leaks into training statistics — guideline deduction.) 5. What is the difference between `predict` and `predict_proba`? 6. What does `n_init` do in K-Means? 7. Why scale before KNN/SVM/K-Means/MLP but not trees? 8. What is inertia? 9. Why is accuracy 0.14 for Naive Bayes yet its precision (weighted) 0.73? (it rarely predicts Severity 2 but is precise when it does — check the matrix.)
