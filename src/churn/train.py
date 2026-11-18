"""Cross-validated model comparison, threshold selection, final fit."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from churn import __version__
from churn.config import Config
from churn.data import FEATURE_COLUMNS
from churn.models import build_pipeline
from churn.utils import get_logger, save_json

logger = get_logger(__name__)

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}


def compare_models(X: pd.DataFrame, y: pd.Series, cfg: Config) -> pd.DataFrame:
    """Same folds for every model, so differences come from the model, not the split."""
    cv = StratifiedKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.seed)
    rows = []
    for name, params in cfg.models.items():
        start = time.perf_counter()
        scores = cross_validate(build_pipeline(name, params, cfg.seed), X, y, cv=cv, scoring=SCORING)
        row = {"model": name, "fit_seconds": round(time.perf_counter() - start, 1)}
        for metric in SCORING:
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(vals.mean()), 4)
            row[f"{metric}_std"] = round(float(vals.std()), 4)
        logger.info("%-20s ROC-AUC %.4f ± %.4f | F1 %.4f", name, row["roc_auc_mean"], row["roc_auc_std"], row["f1_mean"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"{cfg.training.selection_metric}_mean", ascending=False).reset_index(drop=True)


def choose_threshold(y_true: np.ndarray, proba: np.ndarray, cost_fn: float, cost_fp: float) -> tuple[float, pd.DataFrame]:
    """Pick the probability cutoff that minimises expected business cost.

    0.5 is arbitrary: if losing a customer costs 5x a retention offer, we should
    flag customers at lower probabilities. Uses out-of-fold predictions so the
    threshold is never tuned on data the model was trained on.
    """
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for t in thresholds:
        pred = proba >= t
        fn = int(((~pred) & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        rows.append({"threshold": t, "false_negatives": fn, "false_positives": fp, "cost": cost_fn * fn + cost_fp * fp})
    table = pd.DataFrame(rows)
    best = float(table.loc[table["cost"].idxmin(), "threshold"])
    return best, table


def train(X_train: pd.DataFrame, y_train: pd.Series, cfg: Config) -> dict:
    results_dir = Path(cfg.paths.results_dir)
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    comparison = compare_models(X_train, y_train, cfg)
    comparison.to_csv(results_dir / "cv_comparison.csv", index=False)
    best_name = comparison.loc[0, "model"]
    logger.info("Best model by %s: %s", cfg.training.selection_metric, best_name)

    cv = StratifiedKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.seed)
    oof = cross_val_predict(build_pipeline(best_name, cfg.models[best_name], cfg.seed), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, cost_table = choose_threshold(y_train.to_numpy(), oof, cfg.training.cost_false_negative, cfg.training.cost_false_positive)
    cost_table.to_csv(results_dir / "threshold_costs.csv", index=False)
    logger.info(
        "Cost-optimal threshold (FN cost %.1f, FP cost %.1f): %.2f",
        cfg.training.cost_false_negative,
        cfg.training.cost_false_positive,
        threshold,
    )

    final = build_pipeline(best_name, cfg.models[best_name], cfg.seed).fit(X_train, y_train)
    artifact = {
        "pipeline": final,
        "model_name": best_name,
        "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "package_version": __version__,
        "trained_on_rows": len(X_train),
    }
    joblib.dump(artifact, artifacts_dir / "model.joblib")
    save_json(
        {"best_model": best_name, "threshold": threshold, "cv": comparison.to_dict(orient="records")},
        results_dir / "train_summary.json",
    )
    return artifact
