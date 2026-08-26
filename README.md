# ML Experiments

Small, focused notebooks testing specific ML and feature-engineering techniques in depth — separate from my main end-to-end checkpoint projects. Each notebook picks one concept and explores it properly: comparing approaches, checking assumptions, and confirming what actually works with real evaluation, not just running code once and moving on.

## Notebooks

### 1. `binning_strategies_comparison.ipynb`
Compares three `KBinsDiscretizer` strategies (k-means, quantile, uniform) for discretizing continuous features (age, annual income, spending score) on a customer transactions dataset. Uses a `ColumnTransformer` to apply binning across multiple columns, and evaluates each strategy's effect on Decision Tree classification accuracy via 10-fold cross-validation, with before/after distribution histograms for each feature.

### 2. `knn_vs_simple_imputation.ipynb`
Compares `KNNImputer` against `SimpleImputer` (mean) for handling missing values in the WeatherAUS dataset, predicting `RainTomorrow` with Logistic Regression.
- **Finding:** KNN imputation performed almost identically to simple mean imputation (83% accuracy either way, ~42% recall on the "Yes/Rain" class in both cases) — despite being significantly more computationally expensive. On this dataset, the added complexity of KNN imputation wasn't worth its cost.
- Also surfaced an important class-imbalance issue: `RainTomorrow` is ~78% "No" / 22% "Yes", so accuracy alone is a misleading metric — a model that always predicts "No" would already score ~78%.
- **Follow-up experiment:** retrained with `class_weight='balanced'` — rain-day recall nearly doubled (42% → 74%), at the cost of more false alarms and a drop in overall accuracy (83% → 75%). A genuine precision/recall trade-off, not a free improvement — which metric matters more depends on whether missing a rain warning or a false alarm is costlier in the real use case.

## Tools Used
Python · Pandas · NumPy · Scikit-learn · Matplotlib

---
*More notebooks will be added here as I explore other techniques (outlier handling, encoding strategies, etc.) alongside my main project work.*

## Dataset
Download `WeatherAUS.csv` from [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) 
and place it in the project folder before running the notebook.
