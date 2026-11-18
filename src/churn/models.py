"""Model factory: every model is a full Pipeline (features -> preprocessing -> classifier)."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from churn.features import FeatureBuilder, build_preprocessor


def build_classifier(name: str, params: dict[str, Any], seed: int):
    params = dict(params)
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "svm":
        # probability=True fits Platt scaling so we get calibrated-ish scores for ROC-AUC/thresholds.
        return SVC(probability=True, random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str, params: dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            # pandas output keeps readable column names for SHAP plots.
            ("preprocess", build_preprocessor().set_output(transform="pandas")),
            ("model", build_classifier(name, params, seed)),
        ]
    )
