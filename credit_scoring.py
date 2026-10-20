"""Compare baseline classifiers for the Kaggle credit-risk dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


TARGET = "loan_status"
DEFAULT_DATA = Path(__file__).parent / "data" / "credit_risk_dataset.csv"


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_models(random_state: int) -> dict[str, object]:
    return {
        "Dummy baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "Decision tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=random_state
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Hist. gradient boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to CSV dataset")
    parser.add_argument("--output", type=Path, default=Path("model_metrics.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")

    data = data.dropna(subset=[TARGET])
    X = data.drop(columns=TARGET)
    y = data[TARGET].astype(int)
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    results = []
    fitted_models = {}
    for name, estimator in build_models(args.random_state).items():
        model = Pipeline(
            [
                ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
                ("classifier", estimator),
            ]
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "pr_auc": average_precision_score(y_test, probabilities),
            }
        )
        fitted_models[name] = (model, predictions)

    metrics = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)

    best_name = metrics.iloc[0]["model"]
    best_predictions = fitted_models[best_name][1]
    print(f"Rows: {len(data)} | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Positive class rate: {y.mean():.1%}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nBest by ROC-AUC: {best_name}")
    print("Confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(y_test, best_predictions))
    print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()