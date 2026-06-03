import pandas as pd
import pytest

from src.preprocess import FEATURE_COLUMNS, split_features_target, validate_feature_columns


def make_frame():
    row = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
        "target": 0,
    }
    return pd.DataFrame([row])


def test_validate_feature_columns_accepts_expected_schema():
    validate_feature_columns(make_frame())


def test_validate_feature_columns_reports_missing_columns():
    frame = make_frame().drop(columns=["age"])
    with pytest.raises(ValueError, match="age"):
        validate_feature_columns(frame)


def test_split_features_target_returns_expected_columns():
    features, target = split_features_target(make_frame())
    assert list(features.columns) == FEATURE_COLUMNS
    assert target.iloc[0] == 0
