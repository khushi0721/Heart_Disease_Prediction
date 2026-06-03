from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from preprocess import FEATURE_COLUMNS, validate_feature_columns


def load_input(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        return pd.DataFrame(payload)
    return pd.read_csv(input_path)


def predict(model_path: Path, input_path: Path) -> pd.DataFrame:
    pipeline = joblib.load(model_path)
    frame = load_input(input_path)
    validate_feature_columns(frame)

    features = frame[FEATURE_COLUMNS]
    predictions = pipeline.predict(features)
    output = frame.copy()
    output["prediction"] = predictions

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        output["heart_disease_probability"] = pipeline.predict_proba(features)[:, 1]

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heart disease predictions with a saved pipeline.")
    parser.add_argument("--model", type=Path, default=Path("models/heart_disease_pipeline.joblib"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictions = predict(args.model, args.input)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
    else:
        print(predictions.to_string(index=False))
