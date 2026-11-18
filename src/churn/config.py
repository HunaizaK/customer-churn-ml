"""Typed configuration loaded from YAML.

Validating the config up front means a typo fails immediately with a clear
message instead of halfway through a 10-minute training run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

MODEL_NAMES = ("logistic_regression", "svm", "random_forest", "lightgbm")


class DataConfig(BaseModel):
    url: str
    sha256: str | None = None
    raw_path: Path
    target: str = "Churn"
    id_column: str = "customerID"
    test_size: float = Field(0.2, gt=0, lt=1)


class SegmentationConfig(BaseModel):
    features: list[str]
    k_min: int = Field(2, ge=2)
    k_max: int = 8


class TrainingConfig(BaseModel):
    cv_folds: int = Field(5, ge=2)
    selection_metric: str = "roc_auc"
    cost_false_negative: float = Field(5.0, gt=0)
    cost_false_positive: float = Field(1.0, gt=0)


class PathsConfig(BaseModel):
    artifacts_dir: Path = Path("artifacts")
    results_dir: Path = Path("results")


class Config(BaseModel):
    seed: int = 42
    data: DataConfig
    segmentation: SegmentationConfig
    training: TrainingConfig
    models: dict[str, dict[str, Any]]
    paths: PathsConfig = PathsConfig()

    @field_validator("models")
    @classmethod
    def known_models(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        unknown = set(v) - set(MODEL_NAMES)
        if unknown:
            raise ValueError(f"Unknown models in config: {sorted(unknown)}. Allowed: {MODEL_NAMES}")
        return v


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    with Path(path).open(encoding="utf-8") as f:
        return Config(**yaml.safe_load(f))
