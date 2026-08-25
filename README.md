# ML Experiments

Small, focused notebooks testing specific ML and feature-engineering techniques in depth — separate from my main end-to-end checkpoint projects. Each notebook picks one concept and explores it properly: comparing approaches, checking assumptions, and confirming what actually works with real evaluation (not just running code once and moving on).

## Notebooks

- **binning_strategies_comparison.ipynb** — Compares three `KBinsDiscretizer` strategies (k-means, quantile, uniform) for discretizing continuous features (age, annual income, spending score) on a customer transactions dataset. Uses a `ColumnTransformer` to apply binning across multiple columns, and evaluates each strategy's effect on Decision Tree classification accuracy via 10-fold cross-validation, with before/after distribution histograms for each feature.

## Tools Used
Python · Pandas · NumPy · Scikit-learn · Matplotlib

---
*More notebooks will be added here as I explore other techniques (imputation methods, outlier handling, encoding strategies, etc.) alongside my main project work.*
