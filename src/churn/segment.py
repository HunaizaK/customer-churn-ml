"""Unsupervised customer segmentation with k-means.

Segments are fitted on behaviour only (tenure, spend, product count) — the churn
label is NOT an input. Churn rate is then compared across segments to see
which behaviours carry risk.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from churn.config import Config  # noqa: E402
from churn.features import add_features  # noqa: E402
from churn.utils import get_logger, save_json  # noqa: E402

logger = get_logger(__name__)


def choose_k(X: pd.DataFrame, k_min: int, k_max: int, seed: int) -> tuple[int, dict[int, float]]:
    """Pick k with the best silhouette score (how well separated clusters are, -1..1)."""
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        pipe = Pipeline([("scale", StandardScaler()), ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed))])
        labels = pipe.fit_predict(X)
        scores[k] = float(silhouette_score(pipe["scale"].transform(X), labels))
    best = max(scores, key=scores.get)
    return best, scores


def fit_segments(df: pd.DataFrame, cfg: Config) -> tuple[Pipeline, pd.DataFrame, dict[int, float]]:
    seg_cfg = cfg.segmentation
    X = add_features(df)[seg_cfg.features]
    k, scores = choose_k(X, seg_cfg.k_min, seg_cfg.k_max, cfg.seed)
    logger.info("Silhouette by k: %s -> chose k=%d", {k_: round(s, 3) for k_, s in scores.items()}, k)

    model = Pipeline([("scale", StandardScaler()), ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=cfg.seed))])
    labels = model.fit_predict(X)
    # Relabel clusters by mean tenure so segment ids are stable and readable (0 = newest customers).
    order = X.groupby(labels)["tenure"].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    labels = pd.Series(labels, index=df.index).map(remap)
    model.segment_order_ = remap  # stored so predict-time labels match the profile table

    profile = (
        df.assign(segment=labels, num_services=X["num_services"])
        .groupby("segment")
        .agg(
            customers=("tenure", "size"),
            mean_tenure=("tenure", "mean"),
            mean_monthly=("MonthlyCharges", "mean"),
            mean_services=("num_services", "mean"),
            month_to_month_share=("Contract", lambda s: (s == "Month-to-month").mean()),
            churn_rate=(cfg.data.target, "mean"),
        )
        .round(3)
    )
    return model, profile, scores


def assign_segments(model: Pipeline, df: pd.DataFrame, features: list[str]) -> pd.Series:
    raw = model.predict(add_features(df)[features])
    return pd.Series(raw, index=df.index).map(model.segment_order_)


def plot_segments(df: pd.DataFrame, labels: pd.Series, scores: dict[int, float], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(list(scores), list(scores.values()), marker="o")
    axes[0].set(title="Silhouette score by k", xlabel="k (clusters)", ylabel="silhouette")
    for seg in sorted(labels.unique()):
        m = labels == seg
        axes[1].scatter(df.loc[m, "tenure"], df.loc[m, "MonthlyCharges"], s=6, alpha=0.5, label=f"segment {seg}")
    axes[1].set(title="Customer segments", xlabel="tenure (months)", ylabel="monthly charges ($)")
    axes[1].legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(out_dir / "segments.png", dpi=120)
    plt.close(fig)


def run_segmentation(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, profile, scores = fit_segments(df, cfg)
    labels = assign_segments(model, df, cfg.segmentation.features)
    plot_segments(df, labels, scores, out_dir)
    profile.to_csv(out_dir / "segment_profile.csv")
    save_json({"silhouette_by_k": scores, "chosen_k": len(profile)}, out_dir / "segmentation.json")
    Path(cfg.paths.artifacts_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(cfg.paths.artifacts_dir) / "segmenter.joblib")
    logger.info("Segment profile:\n%s", profile.to_string())
    return profile
