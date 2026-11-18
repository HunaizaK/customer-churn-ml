"""Command-line entry point.

python -m churn all          # download -> segment -> train -> evaluate -> explain
python -m churn train        # individual stages
"""

from __future__ import annotations

import argparse
import time

from churn.config import load_config
from churn.data import download_data, load_dataset, split
from churn.evaluate import evaluate
from churn.explain import explain
from churn.predict import load_artifact
from churn.segment import assign_segments, run_segmentation
from churn.train import train
from churn.utils import get_logger, set_seeds

logger = get_logger("churn")
STAGES = ("download", "segment", "train", "evaluate", "explain", "all")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Telco customer churn pipeline")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seeds(cfg.seed)
    start = time.perf_counter()

    download_data(cfg.data)
    if args.stage == "download":
        return

    df = load_dataset(cfg.data)
    X_train, X_test, y_train, y_test = split(df, cfg.data, cfg.seed)
    model_path = cfg.paths.artifacts_dir / "model.joblib"

    if args.stage in ("segment", "all"):
        # Segments are fitted on training customers only, then applied to test.
        run_segmentation(df.loc[X_train.index], cfg)
    if args.stage in ("train", "all"):
        train(X_train, y_train, cfg)
    if args.stage in ("evaluate", "all"):
        segmenter_path = cfg.paths.artifacts_dir / "segmenter.joblib"
        segments = None
        if segmenter_path.exists():
            import joblib

            segments = assign_segments(joblib.load(segmenter_path), X_test, cfg.segmentation.features)
        evaluate(load_artifact(model_path), X_test, y_test, cfg, segments)
    if args.stage in ("explain", "all"):
        explain(load_artifact(model_path), X_train, X_test, cfg)

    logger.info("Stage '%s' finished in %.1fs", args.stage, time.perf_counter() - start)


if __name__ == "__main__":
    main()
