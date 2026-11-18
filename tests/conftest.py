"""Shared fixtures: a small synthetic dataset with the real Telco schema.

Tests never touch the network or the real CSV, so they run in CI in seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def make_telco_like(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yes_no = lambda: rng.choice(["Yes", "No"], n)  # noqa: E731
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    addon = lambda: np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))  # noqa: E731
    phone = yes_no()
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 118, n).round(2)
    # Churn depends on contract + tenure so models have real signal to learn.
    logit = -1.0 + 1.5 * (contract == "Month-to-month") - 0.04 * tenure + 0.6 * (internet == "Fiber optic")
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    total = (monthly * tenure).round(2).astype(str)
    total[tenure == 0] = " "  # mimic the real dataset's blank TotalCharges for new customers
    return pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": yes_no(),
            "Dependents": yes_no(),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
            "InternetService": internet,
            "OnlineSecurity": addon(),
            "OnlineBackup": addon(),
            "DeviceProtection": addon(),
            "TechSupport": addon(),
            "StreamingTV": addon(),
            "StreamingMovies": addon(),
            "Contract": contract,
            "PaperlessBilling": yes_no(),
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_telco_like()


@pytest.fixture
def cfg(tmp_path):
    """Real config, but fast settings and all outputs redirected to a temp dir."""
    c = load_config(ROOT / "configs" / "config.yaml")
    c.training.cv_folds = 2
    c.segmentation.k_max = 4
    c.models["svm"] = {**c.models["svm"]}
    c.models["random_forest"] = {**c.models["random_forest"], "n_estimators": 20}
    c.models["lightgbm"] = {**c.models["lightgbm"], "n_estimators": 20}
    c.paths.artifacts_dir = tmp_path / "artifacts"
    c.paths.results_dir = tmp_path / "results"
    return c
