"""Feature engineering + preprocessing, packaged as sklearn transformers.

Everything lives inside the model Pipeline so the exact same transformations
run in cross-validation, on the test set, and in the API — no training/serving skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]  # fmt: skip
ENGINEERED_NUMERIC = ["num_services", "avg_monthly_spend", "charge_increase"]
ENGINEERED_CATEGORICAL = ["tenure_band"]
TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven features:

    - num_services: more products = higher switching cost.
    - avg_monthly_spend: lifetime spend / months; compares to current bill.
    - charge_increase: current bill minus historic average; price hikes drive churn.
    - tenure_band: churn risk is highly non-linear in tenure (most churn in year 1).
    """
    df = df.copy()
    df["num_services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    if "InternetService" in df:
        df["num_services"] += (df["InternetService"] != "No").astype(int)
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"])
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    df["tenure_band"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    return df


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless sklearn wrapper around add_features."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_features(X)


def build_preprocessor() -> ColumnTransformer:
    numeric = NUMERIC_COLUMNS + ENGINEERED_NUMERIC
    categorical = CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )
