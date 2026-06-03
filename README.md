# XGBoost Classification Ensemble

This repository contains a machine learning classification project built around a high-dimensional tabular dataset. The goal is to predict the binary target `Y` for each `RecordId` and generate a competition-ready submission file.

## Project Highlights

- Binary classification using XGBoost
- Mean imputation for missing feature values
- Correlation-based feature reduction
- Robust scaling for numerical stability
- Weighted ensemble of 200 XGBoost models
- Validation scoring with ROC AUC and accuracy
- Final probability predictions exported as a CSV submission

## Model Approach

The notebook trains an ensemble of XGBoost binary classifiers. Each model uses a different random seed and slightly varied learning parameters. Validation ROC AUC scores are used as ensemble weights, so stronger models contribute more to the final prediction.

Core steps:

1. Load train and test datasets.
2. Impute missing values using mean imputation.
3. Drop highly correlated features using a correlation threshold of `0.8`.
4. Scale features with `RobustScaler`.
5. Train 200 XGBoost models.
6. Weight each model by validation ROC AUC.
7. Generate probability predictions for the test set.
8. Export `RecordId` and predicted `Y` values to `submission.csv`.

## Final Output

The final prediction file is available at:

```text
outputs/submission.csv
```

It contains 105,482 prediction rows with:

- `RecordId`
- `Y`

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```text
notebooks/final-notebook.ipynb
```

Or adapt the script in:

```text
src/train_xgboost_ensemble.py
```

The original notebook was configured for the Kaggle dataset path:

```text
/kaggle/input/iml-fall-2024-challenge-1/
```

If running locally, update the train and test file paths before execution.
