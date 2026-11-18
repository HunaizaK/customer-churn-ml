"""Download, clean, validate and split the Telco churn data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from churn.config import DataConfig
from churn.utils import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]  # fmt: skip
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


class DataValidationError(ValueError):
    """Raised when the raw data does not match the expected schema."""


def download_data(cfg: DataConfig, force: bool = False) -> Path:
    """Fetch the CSV once and verify its checksum so every run sees identical data."""
    path = Path(cfg.raw_path)
    if path.exists() and not force:
        logger.info("Data already at %s", path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s", cfg.url)
        response = requests.get(cfg.url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)

    if cfg.sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != cfg.sha256:
            raise DataValidationError(f"Checksum mismatch for {path}: got {digest}, expected {cfg.sha256}")
    return path


def clean(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """Fix the known quirks of this dataset.

    TotalCharges is stored as text and is blank for brand-new customers
    (tenure == 0) who have not been billed yet, so 0 is the correct value,
    not an imputed mean.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", None, regex=True), errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    if target in df.columns and df[target].dtype == object:
        df[target] = df[target].map({"Yes": 1, "No": 0})
    return df


def validate(df: pd.DataFrame, target: str = "Churn") -> None:
    """Fail loudly on schema drift instead of training on bad data."""
    missing = set(FEATURE_COLUMNS + [target]) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")
    if df[NUMERIC_COLUMNS].isna().any().any():
        bad = df[NUMERIC_COLUMNS].isna().sum()
        raise DataValidationError(f"Null numeric values after cleaning:\n{bad[bad > 0]}")
    if not set(df[target].unique()) <= {0, 1}:
        raise DataValidationError(f"Target must be binary 0/1, got {df[target].unique()}")
    if (df[["tenure", "MonthlyCharges", "TotalCharges"]] < 0).any().any():
        raise DataValidationError("Negative values in tenure/charges")


def load_dataset(cfg: DataConfig) -> pd.DataFrame:
    df = clean(pd.read_csv(cfg.raw_path), cfg.target)
    validate(df, cfg.target)
    logger.info("Loaded %d rows, churn rate %.1f%%", len(df), 100 * df[cfg.target].mean())
    return df


def split(df: pd.DataFrame, cfg: DataConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split keeps the ~26% churn rate identical in train and test."""
    X = df[FEATURE_COLUMNS]
    y = df[cfg.target]
    return train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=seed)
