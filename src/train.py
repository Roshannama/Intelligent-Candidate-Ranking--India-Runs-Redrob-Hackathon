from pathlib import Path
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
FEATURE_FILE = ARTIFACT_DIR / "candidates.parquet"
LABEL_FILE = DATA_DIR / "labels.csv"
MODEL_FILE = ARTIFACT_DIR / "ranker.pkl"
def main():
    print("Loading candidate features...")
    features = pd.read_parquet(
        FEATURE_FILE
    )
    print(
        f"Candidate records: {len(features):,}"
    )
    print("Loading human relevance labels...")
    labels = pd.read_csv(
        LABEL_FILE
    )
    print(
        f"Labeled candidates: {len(labels):,}"
    )
    if "candidate_id" not in features.columns:
        raise ValueError(
            "candidate_id missing from candidates.parquet"
        )
    if "candidate_id" not in labels.columns:
        raise ValueError(
            "candidate_id missing from labels.csv"
        )
    if "relevance" not in labels.columns:
        raise ValueError(
            "relevance column missing from labels.csv"
        )
    data = features.merge(
        labels[
            [
                "candidate_id",
                "relevance"
            ]
        ],
        on="candidate_id",
        how="inner"
    )
    print(
        f"Training records after merge: {len(data):,}"
    )
    if len(data) < 50:
        raise ValueError(
            "Too few labeled candidates. "
            "Create at least ~50 labels."
        )
    data["relevance"] = (
        pd.to_numeric(
            data["relevance"],
            errors="coerce"
        )
        .fillna(0)
        .clip(0, 4)
    )
    exclude_columns = {
        "candidate_id",
        "relevance"
    }
    feature_columns = []
    for column in data.columns:
        if column in exclude_columns:
            continue
        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            feature_columns.append(column)
    if not feature_columns:
        raise ValueError(
            "No numerical features found."
        )
    print()
    print("Features used:")
    print(feature_columns)
    X = (
        data[feature_columns]
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
    )
    y = data["relevance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )
    print()
    print(
        f"Training samples : {len(X_train)}"
    )
    print(
        f"Validation samples : {len(X_test)}"
    )
    print()
    print("Training LightGBM...")
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    model.fit(
        X_train,
        y_train
    )
    predictions = model.predict(
        X_test
    )
    mae = mean_absolute_error(
        y_test,
        predictions
    )
    print()
    print("=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)
    print(
        f"Validation MAE: {mae:.4f}"
    )
    importance = pd.DataFrame({
        "feature":
            feature_columns,
        "importance":
            model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )
    print()
    print("Feature importance:")
    print(
        importance.head(20).to_string(
            index=False
        )
    )
    model_package = {
        "model": model,
        "features": feature_columns
    }
    joblib.dump(
        model_package,
        MODEL_FILE
    )
    print()
    print(
        f"Model saved to: {MODEL_FILE}"
    )
if __name__ == "__main__":
    main()