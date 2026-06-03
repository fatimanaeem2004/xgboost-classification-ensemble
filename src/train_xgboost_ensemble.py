import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


TRAIN_PATH = "/kaggle/input/iml-fall-2024-challenge-1/train_set.csv"
TEST_PATH = "/kaggle/input/iml-fall-2024-challenge-1/test_set.csv"
OUTPUT_PATH = "submission.csv"


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Imputing missing values...")
    imputer = SimpleImputer(strategy="mean")
    train_df.loc[:, train_df.columns != "Y"] = imputer.fit_transform(
        train_df.loc[:, train_df.columns != "Y"]
    )

    print("Reducing highly correlated features...")
    correlation_matrix = train_df.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)
    ]
    train_df_reduced = train_df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} highly correlated features.")

    features_reduced = train_df_reduced.drop(columns=["RecordId", "Y"])
    target = train_df_reduced["Y"]

    print("Scaling features...")
    scaler = RobustScaler()
    features = scaler.fit_transform(features_reduced)

    train_f, val_f, train_t, val_t = train_test_split(
        features, target, test_size=0.30, random_state=42
    )
    train_data = xgb.DMatrix(train_f, label=train_t)
    val_data = xgb.DMatrix(val_f, label=val_t)

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.4,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.5,
        "colsample_bynode": 0.6,
        "scale_pos_weight": 8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "alpha": 3,
        "lambda": 3,
        "max_delta_step": 2,
        "tree_method": "hist",
        "verbosity": 1,
    }

    print("Training weighted XGBoost ensemble...")
    num_models = 200
    num_rounds = 500
    models = []
    weights = []

    for i in range(num_models):
        if i % 10 == 0:
            print(f"Training model {i + 1}/{num_models}...")

        params = base_params.copy()
        params["seed"] = i * 10
        params["eta"] = 0.05 + i * 0.001
        params["colsample_bytree"] = 0.8 - i * 0.001

        model = xgb.train(params, train_data, num_boost_round=num_rounds, verbose_eval=False)
        models.append(model)

        pred_prob = model.predict(val_data)
        weights.append(roc_auc_score(val_t, pred_prob))

    weights = np.array(weights)
    weights /= weights.sum()

    val_preds_weighted = np.zeros(len(val_t))
    for model, weight in zip(models, weights):
        val_preds_weighted += weight * model.predict(val_data)

    val_predictions = [1 if prob > 0.5 else 0 for prob in val_preds_weighted]
    print(f"Validation Accuracy: {accuracy_score(val_t, val_predictions)}")

    print("Preparing test set...")
    test_df_imputed = imputer.transform(test_df)
    test_features = pd.DataFrame(test_df_imputed, columns=test_df.columns)
    test_features = test_features.drop(columns=["RecordId"] + to_drop)
    test_data = xgb.DMatrix(test_features)

    print("Generating test predictions...")
    test_preds_weighted = np.zeros(len(test_features))
    for model, weight in zip(models, weights):
        test_preds_weighted += weight * model.predict(test_data)

    submission = pd.DataFrame({"RecordId": test_df["RecordId"], "Y": test_preds_weighted})
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Submission file created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
