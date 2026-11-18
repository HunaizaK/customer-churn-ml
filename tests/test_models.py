import numpy as np
import pytest

from churn.config import MODEL_NAMES
from churn.data import FEATURE_COLUMNS, clean
from churn.models import build_pipeline
from churn.train import choose_threshold


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_outputs_probabilities(name, raw_df, cfg):
    df = clean(raw_df)
    pipe = build_pipeline(name, cfg.models[name], cfg.seed).fit(df[FEATURE_COLUMNS], df["Churn"])
    proba = pipe.predict_proba(df[FEATURE_COLUMNS])
    assert proba.shape == (len(df), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_unknown_model_rejected(cfg):
    with pytest.raises(ValueError):
        build_pipeline("deep_magic", {}, cfg.seed)


def test_threshold_lowers_when_false_negatives_cost_more():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    # Overlapping score distributions, so the cost trade-off has a unique optimum.
    proba = 1 / (1 + np.exp(-(2 * (2 * y - 1) + rng.normal(0, 1.5, 2000))))
    cheap_fn, _ = choose_threshold(y, proba, cost_fn=1, cost_fp=1)
    costly_fn, _ = choose_threshold(y, proba, cost_fn=10, cost_fp=1)
    assert costly_fn < cheap_fn


def test_training_is_deterministic(raw_df, cfg):
    df = clean(raw_df)
    X, y = df[FEATURE_COLUMNS], df["Churn"]
    a = build_pipeline("lightgbm", cfg.models["lightgbm"], cfg.seed).fit(X, y).predict_proba(X)
    b = build_pipeline("lightgbm", cfg.models["lightgbm"], cfg.seed).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)
