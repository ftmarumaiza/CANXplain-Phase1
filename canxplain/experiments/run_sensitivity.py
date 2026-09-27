"""Fitness-weight sensitivity experiment.

The default weights (0.5 / 0.3 / 0.2) are argued for in meta/fitness.py,
but an argument is not evidence. This runner re-runs the search under a
grid of (alpha, beta, gamma) settings and reports how the SELECTED model
changes -- its family, its complexity, its accuracy and its consistency.

What to look for:

  * If the selection is identical across the whole grid, the weights do
    not matter on this data and the fitness function is decorative.
    Say so.
  * If the selection flips wildly between neighbouring settings, the
    objective is unstable and the default cannot be defended.
  * The useful outcome is in between: a broad plateau around the default
    with sensible movement at the extremes (alpha=1 reproducing
    accuracy-only selection, high gamma collapsing onto small models).

    python -m canxplain.experiments.run_sensitivity --config configs/debug.yaml
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..evaluation import plots
from ..meta.achilles import AchillesSearch
from ..meta.evaluator import CandidateEvaluator
from ..meta.fitness import FitnessSpec
from ..meta.search_space import complexity_reference_bounds, space_from_config
from ..utils import ensure_dir, get_logger
from .common import build_parser, setup, source_datasets, write_summary

LOG = get_logger()

DEFAULT_GRID: List[List[float]] = [
    [1.00, 0.00, 0.00],   # accuracy only -- reproduces ablation A2's objective
    [0.80, 0.10, 0.10],   # performance-dominant
    [0.60, 0.25, 0.15],
    [0.50, 0.30, 0.20],   # the proposed default
    [0.40, 0.40, 0.20],   # accuracy and explanation weighted equally
    [0.34, 0.33, 0.33],   # uniform
    [0.30, 0.50, 0.20],   # explanation-dominant
    [0.40, 0.20, 0.40],   # complexity-dominant
]


def main():
    parser = build_parser("CANXplain Phase 1 — fitness weight sensitivity")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    cfg, tracker, paths = setup(args, experiment="sensitivity")
    from .pipeline import prepare_dataset

    source = args.dataset or source_datasets(cfg)[0]
    groups = list(cfg.features.groups)
    data = prepare_dataset(cfg, source, groups)

    space = space_from_config(cfg)
    bounds = complexity_reference_bounds(space, len(data.feature_names))
    base = FitnessSpec.from_config(cfg.fitness)
    backend = cfg.models.dnn.get("backend", "sklearn")

    grid = cfg.get("sensitivity", {}).get("grid") or DEFAULT_GRID
    seeds = [int(s) for s in cfg.run.seeds]
    rows: List[Dict] = []

    for alpha, beta, gamma in grid:
        for seed in seeds:
            spec = base.variant(
                alpha=float(alpha), beta=float(beta), gamma=float(gamma),
                use_shap_consistency=float(beta) > 0,
                use_complexity=float(gamma) > 0,
            )
            evaluator = CandidateEvaluator(
                cfg, data.X_train, data.y_train, bounds, seed=seed,
                data_tag=f"{source}|{','.join(groups)}")

            search = AchillesSearch(
                space=space, evaluator=evaluator, spec=spec,
                population_size=int(cfg.meta.population),
                rounds=int(cfg.meta.rounds),
                survivors=int(cfg.meta.survivors),
                epsilon=float(cfg.meta.epsilon),
                seed=seed, backend=backend,
            ).run()

            evaluation = search.best_evaluation
            row = {
                "weights": f"α={alpha:g} β={beta:g} γ={gamma:g}",
                "alpha": alpha, "beta": beta, "gamma": gamma, "seed": seed,
                "dataset": source,
                "selected_family": evaluation.candidate.family,
                "selected_config_id": evaluation.candidate.config_id,
                "selected_accuracy": evaluation.accuracy,
                "selected_shap_consistency": evaluation.consistency.score,
                "selected_complexity": evaluation.complexity,
                "selected_complexity_norm": evaluation.complexity_norm,
                "fitness": search.best_fitness,
                "is_default": (alpha, beta, gamma) == (base.alpha, base.beta, base.gamma),
            }
            rows.append(row)
            tracker.log({**row, "evaluation_type": "sensitivity",
                         "arm": "sensitivity", "dataset_train": source,
                         "model_family": evaluation.candidate.family})
            LOG.info("α=%.2f β=%.2f γ=%.2f seed=%d -> %s (acc %.4f, cons %.4f, cx %.0f)",
                     alpha, beta, gamma, seed, evaluation.candidate.family,
                     evaluation.accuracy, evaluation.consistency.score,
                     evaluation.complexity)

    frame = pd.DataFrame(rows)
    averaged = (frame.groupby(["weights", "alpha", "beta", "gamma"], as_index=False)
                .agg({"selected_accuracy": "mean",
                      "selected_shap_consistency": "mean",
                      "selected_complexity": "mean",
                      "selected_family": lambda s: s.mode().iat[0]}))

    ensure_dir(paths["tables"])
    frame.to_csv(f"{paths['tables']}/table9_fitness_sensitivity_raw.csv", index=False)
    averaged.to_csv(f"{paths['tables']}/table9_fitness_sensitivity.csv", index=False)
    plots.figure_sensitivity(averaged, paths["figures"])
    tracker.flush()

    n_distinct = frame["selected_config_id"].nunique()
    lines = [
        "Fitness weight sensitivity",
        "-" * 26,
        averaged.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        f"Distinct configurations selected across the grid: {n_distinct}",
    ]
    if n_distinct == 1:
        lines.append(
            "Only one configuration was ever selected. On this data the "
            "weights make no difference, so the fitness function cannot be "
            "credited with the result. Report this.")
    else:
        lines.append(
            "The selection does respond to the weights. Report this table "
            "alongside the default rather than presenting 0.5/0.3/0.2 as if "
            "it were derived.")
    write_summary(paths, tracker.to_frame(), lines)


if __name__ == "__main__":
    main()
