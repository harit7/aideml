# Digit Classification from Pixel Features: Technical Report

## Introduction

The task is to predict digit labels (0-9) from 64 pixel-intensity features (the sklearn 8x8 digits format), training on `train.csv` (1,257 rows) and evaluating on a held-out test set. Classification accuracy is the target metric. Four modeling approaches were tried: SVM (RBF kernel), Random Forest, k-Nearest Neighbors, and a second Random Forest run motivated by a LightGBM timeout issue.

## Preprocessing

- Features used as-is: 64 pixel-intensity columns, no engineering.
- Standardization (`StandardScaler`) applied only for distance/margin-based models (SVM, kNN); tree-based Random Forest models used raw features since they are scale-invariant.
- No missing value handling or outlier treatment was needed or applied.
- Train/test split was provided (`train.csv` / `test.csv`), with true test labels held in a separate hidden file for evaluation purposes only.

## Modelling Methods

Four models were evaluated, all with fixed hyperparameters (no tuning) and validated via 5-fold (stratified) cross-validation on the training set before scoring on the held-out test set:

1. **SVM (RBF kernel)** — `C=10`, `gamma='scale'`, features standardized in a pipeline.
2. **Random Forest (v1)** — 500 trees, no max depth, default settings otherwise.
3. **k-Nearest Neighbors** — `k=5`, features standardized.
4. **Random Forest (v2)** — 300 trees, substituted for a LightGBM approach that was timing out. On a dataset this small (1,257 rows, 64 features), the timeout was attributed to LightGBM's process/threading overhead rather than genuine compute cost, and RandomForest was chosen as a fast, dependency-light replacement that runs in seconds.

## Results Discussion

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| SVM (RBF) | 97.53% ± 0.68% | **98.52%** |
| Random Forest (500 trees) | 97.21% ± 1.13% | 97.41% |
| kNN (k=5) | 96.74% ± 1.22% | 97.04% |
| Random Forest (300 trees) | 97.45% ± 0.82% | 97.78% |

The SVM with an RBF kernel was the clear winner, at 98.52% test accuracy, outperforming both Random Forest variants and kNN by roughly 0.7-1.5 points. All four models showed CV and test accuracy tracking closely, indicating no overfitting across the board — expected given the small, clean, well-separated nature of the digits dataset. The gap between the two Random Forest runs (97.41% vs 97.78%) is within the noise of CV fold variance and tree count differences (500 vs 300), not a meaningful methodological improvement. kNN performed respectably but was the weakest of the four, consistent with pixel-space Euclidean distance being a slightly less effective similarity measure than the SVM's learned margin.

The LightGBM timeout was a practical constraint rather than a modeling finding — it forced a pivot to Random Forest, which trained in 6 seconds and gave solid (if not best-in-class) results.

## Future Work

- Tune SVM hyperparameters (`C`, `gamma`) via grid/random search — the untuned model already leads, so gains are plausible.
- Investigate the LightGBM timeout root cause (e.g., disable verbosity, cap thread count, or use a minimal `num_leaves`/`n_estimators` config) so gradient boosting can be fairly compared.
- Try dimensionality reduction (PCA) ahead of kNN/SVM, given the 64 features likely contain redundant/correlated pixels.
- Ensemble the SVM and Random Forest predictions (e.g., soft voting) to see if their error patterns are complementary.