"""End-to-end smoke test + API contract test on synthetic data."""

import json

import pytest
from fastapi.testclient import TestClient

from churn.data import clean, split, validate
from churn.evaluate import evaluate
from churn.explain import explain
from churn.segment import assign_segments, run_segmentation
from churn.train import train


@pytest.fixture
def trained(raw_df, cfg):
    df = clean(raw_df)
    validate(df)
    X_train, X_test, y_train, y_test = split(df, cfg.data, cfg.seed)
    run_segmentation(df.loc[X_train.index], cfg)
    artifact = train(X_train, y_train, cfg)
    return artifact, X_train, X_test, y_train, y_test


def test_full_pipeline_writes_all_outputs(trained, cfg):
    import joblib

    artifact, X_train, X_test, _, y_test = trained
    segments = assign_segments(joblib.load(cfg.paths.artifacts_dir / "segmenter.joblib"), X_test, cfg.segmentation.features)
    metrics = evaluate(artifact, X_test, y_test, cfg, segments)
    explain(artifact, X_train, X_test, cfg, n_rows=30)

    assert 0 <= metrics["at_cost_threshold"]["roc_auc"] <= 1
    for name in ["cv_comparison.csv", "test_metrics.json", "slice_metrics.csv", "segment_profile.csv", "shap_beeswarm.png", "test_curves.png"]:
        assert (cfg.paths.results_dir / name).exists(), name
    assert json.loads((cfg.paths.results_dir / "test_metrics.json").read_text())["test_rows"] == len(X_test)


def test_api_predicts(trained, cfg, monkeypatch, raw_df):
    import joblib

    model_path = cfg.paths.artifacts_dir / "model.joblib"
    joblib.dump(trained[0], model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from churn.serve import app

    customer = clean(raw_df).drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/predict", json=[customer, customer])
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 2 and 0 <= body[0]["churn_probability"] <= 1

        bad = {**customer, "Contract": "Lifetime"}
        assert client.post("/predict", json=[bad]).status_code == 422
