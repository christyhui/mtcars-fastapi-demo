"""
train_model.py

Trains a linear regression model on the MTCARS dataset.
Response variable : mpg
Predictors        : wt (weight, 1000 lbs), hp (gross horsepower)

Usage:
    python train_model.py

Output:
    models/model.pkl  — serialized scikit-learn Pipeline (scaler + regressor)
    models/model_metadata.json — feature names, coefficients, and eval metrics
"""

import json
import pathlib
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = pathlib.Path("mtcars.csv")
MODEL_DIR = pathlib.Path("models")
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

PREDICTORS = ["wt", "hp"]
RESPONSE = "mpg"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Load data ─────────────────────────────────────────────────────────────────


def load_data(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    print(df[[RESPONSE] + PREDICTORS].describe().round(2))
    return df


# ── Train ─────────────────────────────────────────────────────────────────────


def train(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X = df[PREDICTORS]
    y = df[RESPONSE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    metrics = {
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(mean_squared_error(y_test, y_pred) ** 0.5, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    regressor: LinearRegression = pipeline.named_steps["regressor"]
    metadata = {
        "predictors": PREDICTORS,
        "response": RESPONSE,
        "coefficients": dict(zip(PREDICTORS, regressor.coef_.tolist())),
        "intercept": round(float(regressor.intercept_), 4),
        "metrics": metrics,
    }

    print("\n── Model summary ────────────────────────────────")
    for k, v in metadata["coefficients"].items():
        print(f"  {k:>4} coefficient : {v:.4f}")
    print(f"  intercept         : {metadata['intercept']:.4f}")
    print(f"\n── Test-set metrics ─────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")

    return pipeline, metadata


# ── Save ──────────────────────────────────────────────────────────────────────


def save(pipeline: Pipeline, metadata: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved  → {MODEL_PATH}")

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved → {METADATA_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    df = load_data(DATA_PATH)
    pipeline, metadata = train(df)
    save(pipeline, metadata)
    print("\nDone. Ready to serve predictions.")


if __name__ == "__main__":
    main()
