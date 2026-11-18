"""Held-out test evaluation and slice-based error analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.config import Config  # noqa: E402
from churn.features import add_features  # noqa: E402
from churn.utils import get_logger, save_json  # noqa: E402

logger = get_logger(__name__)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "flagged_share": round(float(pred.mean()), 4),
    }


def slice_report(X: pd.DataFrame, y: pd.Series, proba: np.ndarray, threshold: float, slice_cols: list[str]) -> pd.DataFrame:
    """Per-slice metrics: where does the model miss churners or raise false alarms?"""
    frame = add_features(X).assign(y=y.to_numpy(), proba=proba, pred=(proba >= threshold).astype(int))
    rows = []
    for col in slice_cols:
        for value, g in frame.groupby(col, observed=True):
            churners = int(g["y"].sum())
            rows.append(
                {
                    "slice": col,
                    "value": str(value),
                    "customers": len(g),
                    "churn_rate": round(g["y"].mean(), 3),
                    "recall": round(((g["pred"] == 1) & (g["y"] == 1)).sum() / churners, 3) if churners else np.nan,
                    "precision": round(((g["pred"] == 1) & (g["y"] == 1)).sum() / max(g["pred"].sum(), 1), 3),
                    "missed_churners": int(((g["pred"] == 0) & (g["y"] == 1)).sum()),
                    "false_alarms": int(((g["pred"] == 1) & (g["y"] == 0)).sum()),
                    "roc_auc": round(roc_auc_score(g["y"], g["proba"]), 3) if g["y"].nunique() == 2 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_curves(y: np.ndarray, proba: np.ndarray, threshold: float, model_name: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    RocCurveDisplay.from_predictions(y, proba, ax=axes[0], name=model_name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC curve (test)")
    PrecisionRecallDisplay.from_predictions(y, proba, ax=axes[1], name=model_name)
    axes[1].axhline(y.mean(), color="k", ls="--", lw=1, label="base rate")
    axes[1].set_title("Precision-recall (test)")
    ConfusionMatrixDisplay.from_predictions(y, (proba >= threshold).astype(int), ax=axes[2], display_labels=["stay", "churn"], colorbar=False)
    axes[2].set_title(f"Confusion matrix @ threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(out_dir / "test_curves.png", dpi=120)
    plt.close(fig)


def evaluate(artifact: dict, X_test: pd.DataFrame, y_test: pd.Series, cfg: Config, segments: pd.Series | None = None) -> dict:
    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proba = artifact["pipeline"].predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()

    metrics = {
        "model": artifact["model_name"],
        "test_rows": len(y),
        "at_cost_threshold": classification_metrics(y, proba, artifact["threshold"]),
        "at_0.5": classification_metrics(y, proba, 0.5),
    }
    save_json(metrics, out_dir / "test_metrics.json")
    plot_curves(y, proba, artifact["threshold"], artifact["model_name"], out_dir)

    X_slices = X_test.assign(segment=segments.to_numpy()) if segments is not None else X_test
    slice_cols = ["Contract", "tenure_band", "InternetService", "PaymentMethod"] + (["segment"] if segments is not None else [])
    slices = slice_report(X_slices, y_test, proba, artifact["threshold"], slice_cols)
    slices.to_csv(out_dir / "slice_metrics.csv", index=False)

    logger.info("Test metrics @ cost threshold: %s", metrics["at_cost_threshold"])
    return metrics
