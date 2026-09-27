"""Stage 2 runner: the meta-learning search alone (arm A0).

Useful for inspecting and tuning the search budget before committing to
the full ablation, which runs the same search up to nine times over.

    python -m canxplain.experiments.run_metalearning --config configs/debug.yaml
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
    parser = build_parser("CANXplain Phase 1 — meta-learning search only")
    parser.add_argument("--dataset", default=None,
                        help="source dataset (default: first configured)")
    args = parser.parse_args()

    cfg, tracker, paths = setup(args, experiment="metalearning")
    from .pipeline import run_arm, save_model_bundle

    source = args.dataset or source_datasets(cfg)[0]
    targets = cross_targets(cfg, source)
    arm = ARMS["A0"]
    prepared_cache = {}
    results = []

    for seed in [int(s) for s in cfg.run.seeds]:
        result = run_arm(cfg, arm, source, seed, tracker,
                         cross_targets=targets, prepared_cache=prepared_cache)
        results.append(result)

        key = f"{source}|{','.join(arm.groups(list(cfg.features.groups)))}"
        if cfg.run.get("save_models", True) and key in prepared_cache:
            path = save_model_bundle(result, prepared_cache[key], paths["models"])
            LOG.info("saved model bundle -> %s", path)

    tracker.flush()
    runs = tracker.to_frame()
    rounds, candidates = collect_traces(results)
    tracker.save_frame("search_rounds", rounds)
    tracker.save_frame("search_candidates", candidates)

    tables.table3_candidate_configurations(cfg, paths["tables"])
    tables.table4_metalearning_search(candidates, rounds, paths["tables"])
    tables.table5_full_performance(runs, paths["tables"])

    reference = results[0]
    plots.figure2_search_process(rounds, paths["figures"])
    plots.figure3_shap_importance(reference.shap_importance, paths["figures"],
                                  f" ({source}, A0)")
    plots.figure4_shap_consistency(
        reference.search.best_evaluation.consistency, paths["figures"])
    plots.figure7_tradeoff(candidates, paths["figures"])

    selected = reference.final_model
    lines = [
        "Selected model (seed %d):" % reference.seed,
        f"  family          : {selected.family}",
        f"  hyperparameters : {selected.params}",
        f"  complexity      : {reference.final_complexity:.0f}",
        f"  fitness         : {reference.search.best_fitness:.4f}",
        f"  SHAP consistency: {reference.search.best_evaluation.consistency.score:.4f}",
        f"  search cost     : {reference.search.n_evaluations} evaluations, "
        f"{reference.search.search_time_s:.1f}s",
    ]
    write_summary(paths, runs, lines)


if __name__ == "__main__":
    main()
