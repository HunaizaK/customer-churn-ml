"""SHAP explanations for the selected model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from churn.config import Config  # noqa: E402
from churn.utils import get_logger, save_json  # noqa: E402

logger = get_logger(__name__)

TREE_MODELS = {"random_forest", "lightgbm"}


def shap_values(artifact: dict, X: pd.DataFrame, background: pd.DataFrame) -> shap.Explanation:
    """Explain the classifier in the space of its (one-hot/scaled) input features.

    Tree models use exact TreeSHAP; linear models use LinearExplainer; anything
    else falls back to the model-agnostic permutation explainer.
    """
    pipe = artifact["pipeline"]
    pre = pipe[:-1]  # features + preprocessing
    clf = pipe[-1]
    Xt = pre.transform(X)
    bt = pre.transform(background)

    if artifact["model_name"] in TREE_MODELS:
        explanation = shap.TreeExplainer(clf, bt)(Xt, check_additivity=False)
    elif artifact["model_name"] == "logistic_regression":
        explanation = shap.LinearExplainer(clf, bt)(Xt)
    else:
        explanation = shap.Explainer(lambda a: clf.predict_proba(pd.DataFrame(a, columns=Xt.columns))[:, 1], bt)(Xt)

    if explanation.values.ndim == 3:  # (rows, features, classes) -> keep the "churn" class
        explanation = explanation[:, :, 1]
    return explanation


def explain(artifact: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, cfg: Config, n_rows: int = 500) -> pd.DataFrame:
    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    background = X_train.sample(min(200, len(X_train)), random_state=cfg.seed)
    sample = X_test.sample(min(n_rows, len(X_test)), random_state=cfg.seed)
    exp = shap_values(artifact, sample, background)

    importance = (
        pd.DataFrame({"feature": exp.feature_names, "mean_abs_shap": np.abs(exp.values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(out_dir / "shap_importance.csv", index=False)
    save_json({"top_features": importance.head(10).to_dict(orient="records")}, out_dir / "shap_top10.json")

    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title(f"SHAP: what pushes churn risk up/down ({artifact['model_name']})")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=120)
    plt.close("all")
    logger.info("Top SHAP features:\n%s", importance.head(10).to_string(index=False))
    return importance
