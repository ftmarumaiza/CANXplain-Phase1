"""The search space -- this project's equivalent of ACHILLES's model_archi_DB.

Every hyperparameter is a DISCRETE, ordered list of values. Two reasons:

  1. The Q-table in the meta-learner indexes hyperparameter values
     directly, so a continuous space would need binning anyway.
  2. Ordered levels make "mutate one step" a meaningful local move
     during exploitation, which is what turns the evolution operator
     into a real local search rather than a re-roll.

Bounds are deliberately modest. The target platform is an i3 with no
GPU, so the DNN space tops out at 4 hidden layers and 128 neurons; the
ACHILLES paper searched up to ~10 layers and ~256 neurons on a
128 GB / dual-TITAN machine. This is a stated budget difference, not an
attempt to reproduce their search.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..models.registry import CandidateModel, theoretical_dnn_complexity


def space_from_config(cfg) -> Dict[str, Dict[str, List[Any]]]:
    """Read the per-family discrete grids out of the config."""
    return {
        "rf": {k: list(v) for k, v in cfg.models.rf.space.to_dict().items()},
        "dnn": {k: list(v) for k, v in cfg.models.dnn.space.to_dict().items()},
    }


def sample_config(space: Dict[str, List[Any]], rng: np.random.Generator) -> Dict[str, Any]:
    """Uniform random draw -- the exploration operator."""
    return {key: values[int(rng.integers(len(values)))] for key, values in space.items()}


def mutate_config(
    params: Dict[str, Any],
    space: Dict[str, List[Any]],
    rng: np.random.Generator,
    n_changes: int = 1,
) -> Dict[str, Any]:
    """Move a winner one step along a few axes -- the local search move."""
    out = dict(params)
    keys = list(space)
    if not keys:
        return out
    n_changes = max(1, min(n_changes, len(keys)))
    for key in rng.choice(keys, size=n_changes, replace=False):
        values = space[key]
        try:
            current = values.index(out.get(key))
        except (ValueError, TypeError):
            current = int(rng.integers(len(values)))
        step = int(rng.choice([-1, 1]))
        out[key] = values[int(np.clip(current + step, 0, len(values) - 1))]
    return out


def crossover_config(
    parent_a: Dict[str, Any],
    parent_b: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Uniform crossover between two configurations of the same family."""
    return {
        key: (parent_a[key] if rng.random() < 0.5 else parent_b.get(key, parent_a[key]))
        for key in parent_a
    }


def initial_population(
    space: Dict[str, Dict[str, List[Any]]],
    size: int,
    seed: int,
    families: List[str] | None = None,
    backend: str = "sklearn",
) -> List[CandidateModel]:
    """Seed the population with a balanced mix of RF and DNN candidates.

    Balanced on purpose: if one family dominated the initial draw it
    would also dominate the early competition rounds, and the search
    would converge on that family for reasons of sampling rather than
    fitness.
    """
    families = families or ["rf", "dnn"]
    rng = np.random.default_rng(seed)
    population: List[CandidateModel] = []
    for i in range(size):
        family = families[i % len(families)]
        params = sample_config(space[family], rng)
        backend_for = backend if family == "dnn" else "sklearn"
        population.append(CandidateModel(family, params, seed=seed, backend=backend_for))
    return population


def complexity_reference_bounds(
    space: Dict[str, Dict[str, List[Any]]],
    n_features: int,
) -> Dict[str, tuple]:
    """Fixed (min, max) complexity per family, derived from the space itself.

    These bounds are computed from the SEARCH SPACE, not from the
    population that happens to be alive in a given round. That keeps the
    complexity term on an identical scale across rounds, across seeds,
    and across ablations -- so an A0 fitness value and an A4 fitness
    value are comparable numbers rather than two differently-scaled
    quantities that happen to share a name.
    """
    rf_space = space["rf"]
    min_trees, max_trees = min(rf_space["n_estimators"]), max(rf_space["n_estimators"])
    depths = [d for d in rf_space["max_depth"] if d is not None]
    min_depth = min(depths) if depths else 4
    max_depth = max(depths) if depths else 20

    # A tree of depth d holds at most 2^(d+1)-1 nodes; the realistic
    # minimum for a fitted tree is a handful of nodes.
    rf_min = float(min_trees * (2 * min_depth + 1))
    rf_max = float(max_trees * min(2 ** (max_depth + 1) - 1, 20_000))

    dnn_params = []
    for layers in space["dnn"]["n_hidden_layers"]:
        for width in space["dnn"]["neurons_per_layer"]:
            for taper in space["dnn"].get("taper", [1.0]):
                dnn_params.append(theoretical_dnn_complexity(
                    {"n_hidden_layers": layers, "neurons_per_layer": width,
                     "taper": taper}, n_features))
    dnn_min, dnn_max = float(min(dnn_params)), float(max(dnn_params))

    return {"rf": (rf_min, rf_max), "dnn": (dnn_min, dnn_max)}
