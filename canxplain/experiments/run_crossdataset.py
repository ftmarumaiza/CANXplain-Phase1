"""Stage 3 runner: cross-dataset generalisation.

Trains the full system on each source dataset and evaluates it on every
other configured dataset. The target is loaded only after the source
model, scaler and feature set are frozen.

Cross-dataset numbers are normally much lower than same-dataset numbers.
That is the honest picture of vendor transfer, not a bug in the
pipeline.

    python -m canxplain.experiments.run_crossdataset --config configs/full.yaml
"""
from __future__ import annotations

from ..evaluation import plots, tables
from ..utils import get_logger
from .ablations import ARMS
from .common import (
    build_parser,
    collect_traces,
    cross_targets,
    setup,
    source_datasets,
    write_summary,
)

LOG = get_logger()


def main():
    parser = build_parser("CANXplain Phase 1 — cross-dataset generalisation")
    parser.add_argument("--arm", default="A0",
                        help="which arm to transfer (default A0)")
    args = parser.parse_args()

    cfg, tracker, paths = setup(args, experiment="crossdataset")
    from .pipeline import run_arm

    arm = ARMS[args.arm]
    sources = source_datasets(cfg)
    if len(sources) < 2:
        LOG.warning(
            "only %d dataset configured. Cross-dataset evaluation needs at "
            "least two; the run will produce same-dataset rows only.", len(sources))

    prepared_cache = {}
    results = []
    for source in sources:
        targets = cross_targets(cfg, source)
        LOG.info("source '%s' -> targets %s", source, targets or "(none)")
        for seed in [int(s) for s in cfg.run.seeds]:
            try:
                results.append(run_arm(cfg, arm, source, seed, tracker,
                                       cross_targets=targets,
                                       prepared_cache=prepared_cache))
            except Exception as exc:
                LOG.error("%s/seed%d failed: %s", source, seed, exc)

    if not results:
        LOG.error("no source dataset completed")
        tracker.flush()
        return

    tracker.flush()
    runs = tracker.to_frame()
    rounds, candidates = collect_traces(results)

    tables.table7_cross_dataset(runs, paths["tables"])
    for metric in ("f1", "accuracy", "roc_auc"):
        plots.figure6_cross_dataset(runs, paths["figures"], metric=metric,
                                    arm=args.arm)

    cross = runs[runs["evaluation_type"] == "cross_dataset"]
    same = runs[runs["evaluation_type"] == "same_dataset"]
    lines = []
    if not cross.empty and not same.empty:
        drop = same["f1"].mean() - cross["f1"].mean()
        lines = [
            f"Mean same-dataset F1  : {same['f1'].mean():.4f}",
            f"Mean cross-dataset F1 : {cross['f1'].mean():.4f}",
            f"Transfer gap          : {drop:.4f}",
            "",
            "The transfer gap is the quantity a generalisable IDS has to "
            "shrink. Report it directly rather than reporting only the "
            "same-dataset number.",
        ]
    write_summary(paths, runs, lines)


if __name__ == "__main__":
    main()
