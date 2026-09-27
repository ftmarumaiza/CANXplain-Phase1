"""Stage 1 runner: data + features only.

Run this first. It verifies that every configured dataset loads, splits
and featurises correctly, and writes Tables 1 and 2 plus Figure 1. It
trains nothing, so it finishes in seconds and tells you whether the rest
of the pipeline will work before you spend an hour on it.

    python -m canxplain.experiments.run_pipeline --config configs/debug.yaml
"""
from __future__ import annotations

import os

import pandas as pd

from ..evaluation import plots, tables
from ..features.extractor import resolve_feature_names
from ..utils import get_logger
from .common import build_parser, dataset_names, setup

LOG = get_logger()


def main():
    parser = build_parser("CANXplain Phase 1 — data and feature pipeline")
    parser.add_argument("--save-features", action="store_true",
                        help="write the featurised splits to parquet")
    args = parser.parse_args()

    cfg, tracker, paths = setup(args, experiment="pipeline")
    from .pipeline import prepare_dataset

    groups = list(cfg.features.groups)
    LOG.info("feature groups: %s -> %d features",
             groups, len(resolve_feature_names(groups)))

    stream_stats = []
    for name in dataset_names(cfg):
        try:
            data = prepare_dataset(cfg, name, groups)
        except Exception as exc:
            LOG.error("dataset '%s' failed to prepare: %s", name, exc)
            continue

        stats = dict(data.stream_stats)
        stats.update({
            "train_windows": len(data.X_train),
            "val_windows": len(data.X_val),
            "test_windows": len(data.X_test),
            "train_attack_rate": float(data.y_train.mean()),
            "val_attack_rate": float(data.y_val.mean()),
            "test_attack_rate": float(data.y_test.mean()),
            "n_features": len(data.feature_names),
        })
        stream_stats.append(stats)

        if args.save_features:
            feature_dir = os.path.join(paths["artifacts"], "features", name)
            os.makedirs(feature_dir, exist_ok=True)
            for split, (X, y) in [("train", (data.X_train, data.y_train)),
                                  ("val", (data.X_val, data.y_val)),
                                  ("test", (data.X_test, data.y_test))]:
                frame = X.copy()
                frame["label"] = y
                frame.to_parquet(os.path.join(feature_dir, f"{split}.parquet"),
                                 index=False)
            LOG.info("saved featurised splits for '%s' to %s", name, feature_dir)

        if data.y_train.min() == data.y_train.max():
            LOG.warning(
                "dataset '%s': the training block is single-class. The "
                "chronological split has landed entirely inside one regime. "
                "Increase max_messages, or check that the capture actually "
                "interleaves normal and attack traffic.", name)

    if not stream_stats:
        LOG.error("no dataset could be prepared; check data.datasets paths")
        return

    tables.table1_dataset_statistics(stream_stats, paths["tables"])
    tables.table2_feature_description(groups, paths["tables"])
    plots.figure1_pipeline(paths["figures"])

    pd.DataFrame(stream_stats).to_csv(
        os.path.join(paths["artifacts"], "dataset_stats.csv"), index=False)
    tracker.flush()
    LOG.info("pipeline check complete. Tables 1-2 and Figure 1 written to %s",
             paths["root"])


if __name__ == "__main__":
    main()
