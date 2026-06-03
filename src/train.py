from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocess import build_preprocessor, split_features_target


MODELS = {
    "knn": KNeighborsClassifier(n_neighbors=8),
    "svc_linear": SVC(kernel="linear", random_state=0),
    "decision_tree": DecisionTreeClassifier(max_features=18, random_state=0),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=0),
}


def evaluate_model(model, x_train, x_valid, y_train, y_valid):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_valid)

    metrics = {
        "accuracy": accuracy_score(y_valid, predictions),
        "precision": precision_score(y_valid, predictions, zero_division=0),
        "recall": recall_score(y_valid, predictions, zero_division=0),
        "f1": f1_score(y_valid, predictions, zero_division=0),
    }

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probabilities = pipeline.predict_proba(x_valid)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_valid, probabilities)
    elif hasattr(pipeline.named_steps["model"], "decision_function"):
        scores = pipeline.decision_function(x_valid)
        metrics["roc_auc"] = roc_auc_score(y_valid, scores)
    else:
        metrics["roc_auc"] = None

    return pipeline, metrics


def train(data_path: Path, model_output: Path, metrics_output: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    x, y = split_features_target(frame)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )

    rows = []
    best_pipeline = None
    best_name = None
    best_accuracy = -1.0

    for name, model in MODELS.items():
        pipeline, metrics = evaluate_model(model, x_train, x_valid, y_train, y_valid)
        rows.append({"model": name, **metrics})
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_name = name
            best_pipeline = pipeline

    results = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_output)
    results.to_csv(metrics_output, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved best model: {best_name} -> {model_output}")
    print(f"Saved metrics: {metrics_output}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heart disease classification models.")
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("models/heart_disease_pipeline.joblib"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/model_metrics.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model_output, args.metrics_output)
