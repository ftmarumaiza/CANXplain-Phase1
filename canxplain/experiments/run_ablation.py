"""The main runner: every ablation arm, every seed, every source dataset.

This is the experiment the study is built on. It produces Tables 4-8,
Figures 2-7, the paired statistical tests, the saved models and the run
log.

    python -m canxplain.experiments.run_ablation --config configs/debug.yaml
    python -m canxplain.experiments.run_ablation --config configs/full.yaml
    python -m canxplain.experiments.run_ablation -c configs/full.yaml \
        --arms A0 A2 --set run.seeds=[0,1,2,3,4]
"""
from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from ..evaluation import plots, tables
from ..evaluation.metrics import paired_test
from ..utils import get_logger
from .ablations import DEFAULT_ARM_ORDER, ablation_documentation, get_arms
from .common import (
    build_parser,
    collect_traces,
    cross_targets,
    pick_reference_result,
    setup,
    source_datasets,
    write_summary,
)

LOG = get_logger()


def run_statistical_tests(runs: pd.DataFrame, reference: str = "A0",
                          metrics: List[str] | None = None) -> List[Dict]:
    """Paired comparison of the reference arm against every other arm.

    Pairs are matched on (dataset_train, dataset_test, evaluation_type,
    seed), so each pair differs only in the arm. Unmatched rows are
    dropped rather than compared across different conditions.
    """
    metrics = metrics or ["f1", "accuracy", "shap_consistency"]
    records: List[Dict] = []

    for evaluation_type in runs["evaluation_type"].dropna().unique():
        if evaluation_type not in ("same_dataset", "cross_dataset"):
            continue
        subset = runs[runs["evaluation_type"] == evaluation_type]
        pair_keys = ["dataset_train", "dataset_test", "seed"]

        ref = subset[subset["arm"] == reference].set_index(pair_keys)
        if ref.empty:
            continue

        for arm in sorted(subset["arm"].dropna().unique()):
            if arm == reference:
                continue
            other = subset[subset["arm"] == arm].set_index(pair_keys)
            shared = ref.index.intersection(other.index)
            if len(shared) == 0:
                continue

            for metric in metrics:
                if metric not in ref.columns or metric not in other.columns:
                    continue
                record = paired_test(
                    ref.loc[shared, metric].to_numpy(),
                    other.loc[shared, metric].to_numpy(),
                    label_a=reference, label_b=arm,
                )
                # The pair count alone hides what the pairs are made of:
                # 6 pairs can be 6 seeds on one dataset pair, or 3 seeds
                # on two. They are not equivalent evidence, so both
                # counts are reported next to the p-value.
                shared_frame = shared.to_frame(index=False)
                record.update({
                    "metric": metric,
                    "evaluation_type": evaluation_type,
                    "n_seeds": int(shared_frame["seed"].nunique()),
                    "n_dataset_pairs": int(
                        shared_frame[["dataset_train", "dataset_test"]]
                        .drop_duplicates().shape[0]),
                    "pairing_unit": "(dataset_train, dataset_test, seed)",
                })
                records.append(record)
    return records


def main():
    parser = build_parser("CANXplain Phase 1 — ablation study")
    parser.add_argument("--arms", nargs="*", default=None,
                        help=f"subset of arms to run (default: {DEFAULT_ARM_ORDER})")
    parser.add_argument("--no-save-models", action="store_true",
                        help="skip writing model bundles")
    args = parser.parse_args()

    cfg, tracker, paths = setup(args, experiment="ablation")
    from .pipeline import run_arm, save_model_bundle, prepare_dataset

    arm_codes = args.arms or list(cfg.get("ablation", {}).get("arms", DEFAULT_ARM_ORDER))
    arms = get_arms(arm_codes)
    seeds = [int(s) for s in cfg.run.seeds]
    sources = source_datasets(cfg)

    LOG.info("ablation plan: %d arms x %d seeds x %d source datasets = %d runs",
             len(arms), len(seeds), len(sources), len(arms) * len(seeds) * len(sources))

    with open(os.path.join(paths["root"], "ABLATION.md"), "w", encoding="utf-8") as fh:
        fh.write(ablation_documentation())

    prepared_cache: Dict[str, object] = {}
    results = []
    stream_stats = []

    for source in sources:
        targets = cross_targets(cfg, source)
        for arm in arms:
            for seed in seeds:
                try:
                    result = run_arm(cfg, arm, source, seed, tracker,
                                     cross_targets=targets,
                                     prepared_cache=prepared_cache)
                except Exception as exc:
                    LOG.error("[%s] %s/seed%d failed: %s", arm.code, source, seed, exc,
                              exc_info=cfg.run.get("debug_traceback", False))
                    continue
                results.append(result)

                if not args.no_save_models and cfg.run.get("save_models", True):
                    key = f"{source}|{','.join(arm.groups(list(cfg.features.groups)))}"
                    data = prepared_cache.get(key)
                    if data is not None:
                        save_model_bundle(result, data, paths["models"])

    if not results:
        LOG.error("no arm completed. Nothing to report.")
        tracker.flush()
        return

    for key, data in prepared_cache.items():
        stats = dict(data.stream_stats)
        stats["feature_config"] = key.split("|", 1)[1]
        stream_stats.append(stats)

    tracker.flush()
    runs = tracker.to_frame()
    rounds, candidates = collect_traces(results)

    tracker.save_frame("search_rounds", rounds)
    tracker.save_frame("search_candidates", candidates)

    reference = pick_reference_result(results, "A0")
    tracker.save_artifact("selected_model_A0", {
        "arm": reference.arm_code, "dataset": reference.dataset,
        "seed": reference.seed, "family": reference.final_model.family,
        "hyperparameters": reference.final_model.params,
        "complexity": reference.final_complexity,
        "fitness": reference.search.best_fitness,
        "same_dataset_metrics": reference.same_dataset_metrics,
        "cross_dataset_metrics": reference.cross_dataset_metrics,
    })
    if not reference.shap_importance.empty:
        tracker.save_frame("shap_importance_A0", reference.shap_importance)

    test_records = run_statistical_tests(runs, reference="A0")

    tables.generate_all_tables(cfg, runs, stream_stats, candidates, rounds,
                               test_records, paths["tables"])
    plots.generate_all_figures(
        runs, rounds, candidates, reference.shap_importance,
        reference.search.best_evaluation.consistency, paths["figures"])

    underpowered = [r for r in test_records if not r.get("sufficient_power")]
    notes = []
    if underpowered:
        notes.append(
            f"NOTE: {len(underpowered)} of {len(test_records)} paired tests ran "
            f"on fewer than 5 seeds and are flagged sufficient_power=False in "
            f"statistical_tests.csv. Those p-values are not evidence.")
    write_summary(paths, runs, notes)


if __name__ == "__main__":
    main()
