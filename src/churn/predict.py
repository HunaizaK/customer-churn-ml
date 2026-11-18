"""Load a trained artifact and score new customers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn.data import clean


def load_artifact(path: str | Path = "artifacts/model.joblib") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `make train` first.")
    return joblib.load(path)


def predict(artifact: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = clean(pd.DataFrame(records))[artifact["feature_columns"]]
    proba = artifact["pipeline"].predict_proba(df)[:, 1]
    return [
        {"churn_probability": round(float(p), 4), "will_churn": bool(p >= artifact["threshold"]), "threshold": artifact["threshold"]} for p in proba
    ]
