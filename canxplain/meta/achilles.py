"""ACHILLES-inspired meta-learning model search.

RELATIONSHIP TO THE ACHILLES PAPER -- READ THIS BEFORE CITING
-------------------------------------------------------------
Borrowed from Mowla et al., Algorithms 1 and 2:

  * a population of learning models treated as meta-agents
  * repeated COMPETITION ROUNDS in which candidates are paired at random
    and ranked by a fitness function
  * an EVOLUTION operator that replaces each loser either by exploiting
    the winner and the meta-learner's hierarchical Q-table (probability
    1-epsilon) or by exploring a fresh random configuration
    (probability epsilon)
  * an ELIMINATION stage that reduces the survivor set W to a fixed size
    n through further pairwise competition
  * final selection of the best configuration seen across all rounds

NOT borrowed -- these are the CANXplain changes, and they are the thing
being evaluated:

  * The fitness function. ACHILLES competes candidates on detection
    performance (accuracy / precision). CANXplain competes them on
    alpha*accuracy + beta*SHAP_consistency - gamma*complexity. Ablation
    A2 turns this back into ACHILLES-style accuracy-only selection,
    which is the direct comparison.
  * SHAP consistency as a first-class quantity rather than a
    post-hoc figure. ACHILLES computes SHAP to interpret the chosen
    model; here it feeds the selection itself.
  * Blocked, purged cross-validation inside the search, so consistency
    and accuracy are both measured without leakage across overlapping
    windows.

This is a re-implementation in the spirit of the published algorithm,
written from the paper's description. It is not the authors' code and
should not be presented as a reproduction of their results.

HIERARCHICAL Q-TABLE
--------------------
Q[family][hyperparameter][value] -> running mean fitness observed for
candidates carrying that value, plus a visit count. "Hierarchical" in
the sense used in the paper: the table is indexed by family first, then
by the axes of that family's configuration space, rather than over whole
joint configurations (which would almost never see a repeat visit).

Exploitation builds a child by crossing the round's winner with the
Q-table's current per-axis argmax, then mutating one axis. So the child
inherits what won locally, what has worked globally, and one step of
fresh local search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models.registry import CandidateModel
from ..utils import get_logger
from .evaluator import CandidateEvaluation, CandidateEvaluator
from .fitness import FitnessSpec
from .search_space import (
    crossover_config,
    initial_population,
    mutate_config,
    sample_config,
)

LOG = get_logger()


class HierarchicalQTable:
    """Running mean fitness per (family, hyperparameter, value)."""

    def __init__(self, space: Dict[str, Dict[str, List[Any]]]):
        self.space = space
        self.table: Dict[str, Dict[str, Dict[Any, List[float]]]] = {
            family: {axis: {self._key(v): [0.0, 0.0] for v in values}
                     for axis, values in axes.items()}
            for family, axes in space.items()
        }

    @staticmethod
    def _key(value) -> str:
        return repr(value)

    def update(self, family: str, params: Dict[str, Any], fitness: float) -> None:
        if not np.isfinite(fitness):
            return
        for axis, value in params.items():
            cell = self.table.get(family, {}).get(axis)
            if cell is None:
                continue
            key = self._key(value)
            if key not in cell:
                cell[key] = [0.0, 0.0]
            total, count = cell[key]
            cell[key] = [total + float(fitness), count + 1.0]

    def best_params(self, family: str) -> Dict[str, Any]:
        """Per-axis argmax of mean observed fitness.

        Axes with no visits yet fall back to the middle of their range,
        which is a neutral choice rather than an accidental bias toward
        whichever value happens to sort first.
        """
        out: Dict[str, Any] = {}
        for axis, values in self.space[family].items():
            cell = self.table[family][axis]
            best_value, best_mean = None, -np.inf
            for value in values:
                total, count = cell.get(self._key(value), [0.0, 0.0])
                if count > 0:
                    mean = total / count
                    if mean > best_mean:
                        best_mean, best_value = mean, value
            out[axis] = best_value if best_value is not None else values[len(values) // 2]
        return out

    def as_records(self) -> List[Dict[str, Any]]:
        rows = []
        for family, axes in self.table.items():
            for axis, cell in axes.items():
                for value_key, (total, count) in cell.items():
                    if count > 0:
                        rows.append({
                            "family": family, "hyperparameter": axis,
                            "value": value_key, "visits": int(count),
                            "mean_fitness": total / count,
                        })
        return rows


@dataclass
class SearchTrace:
    """Per-round record of the search, the source of Figure 2 and Table 4."""
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def add_round(self, **kwargs):
        self.rounds.append(kwargs)

    def add_candidate(self, record: Dict[str, Any]):
        self.candidates.append(record)


@dataclass
class SearchResult:
    """What the meta-learner hands back."""
    best_evaluation: CandidateEvaluation
    best_fitness: float
    spec: FitnessSpec
    trace: SearchTrace
    q_table: List[Dict[str, Any]]
    n_evaluations: int
    n_cache_hits: int
    search_time_s: float

    @property
    def best_candidate(self) -> CandidateModel:
        return self.best_evaluation.candidate


class AchillesSearch:
    """Competition-and-evolution search over RF and DNN configurations."""

    def __init__(
        self,
        space: Dict[str, Dict[str, List[Any]]],
        evaluator: CandidateEvaluator,
        spec: FitnessSpec,
        population_size: int = 8,
        rounds: int = 3,
        survivors: int = 4,
        epsilon: float = 0.3,
        seed: int = 0,
        families: Optional[List[str]] = None,
        backend: str = "sklearn",
    ):
        if survivors > population_size:
            raise ValueError("survivors cannot exceed population_size")
        if population_size < 2:
            raise ValueError("population_size must be at least 2")

        self.space = space
        self.evaluator = evaluator
        self.spec = spec
        self.population_size = int(population_size)
        self.rounds = int(rounds)
        self.survivors = int(survivors)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.families = families or ["rf", "dnn"]
        self.backend = backend

        self.rng = np.random.default_rng(seed)
        self.q_table = HierarchicalQTable(space)
        self.trace = SearchTrace()
        # SHAP is only computed when the objective actually consumes it.
        self.need_shap = spec.use_shap_consistency

    # ---------------------------------------------------------- helpers --
    def _score(self, evaluation: CandidateEvaluation) -> float:
        return evaluation.score(self.spec)

    def _evaluate_population(
        self, population: List[CandidateModel], round_idx: int
    ) -> List[Tuple[CandidateModel, CandidateEvaluation, float]]:
        scored = []
        for candidate in population:
            try:
                evaluation = self.evaluator.evaluate(candidate, need_shap=self.need_shap)
            except Exception as exc:
                LOG.warning("candidate %s failed to evaluate (%s); dropped",
                            candidate.config_id, exc)
                continue
            fitness = self._score(evaluation)
            self.q_table.update(candidate.family, candidate.params, fitness)

            record = evaluation.as_record(self.spec)
            record.update({"round": round_idx, "fitness_selected": fitness})
            self.trace.add_candidate(record)
            scored.append((candidate, evaluation, fitness))
        return scored

    def _evolve(self, winner: CandidateModel, loser: CandidateModel) -> CandidateModel:
        """Algorithm 2: exploit with probability 1-epsilon, else explore."""
        family = winner.family
        if self.rng.random() > self.epsilon:
            # exploitation: winner x Q-table argmax, then one local step
            q_best = self.q_table.best_params(family)
            child_params = crossover_config(winner.params, q_best, self.rng)
            child_params = mutate_config(child_params, self.space[family], self.rng,
                                         n_changes=1)
        else:
            # exploration: fresh random configuration, family redrawn too,
            # so a family that is losing can still be re-entered
            family = self.families[int(self.rng.integers(len(self.families)))]
            child_params = sample_config(self.space[family], self.rng)

        backend = self.backend if family == "dnn" else "sklearn"
        return CandidateModel(family, child_params, seed=self.seed, backend=backend)

    def _compete(
        self, scored: List[Tuple[CandidateModel, CandidateEvaluation, float]]
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Random pairwise competition -> (survivors W, non-survivors W')."""
        order = self.rng.permutation(len(scored))
        winners, losers = [], []
        for i in range(0, len(order) - 1, 2):
            a, b = scored[order[i]], scored[order[i + 1]]
            if a[2] >= b[2]:
                winners.append(a)
                losers.append(b)
            else:
                winners.append(b)
                losers.append(a)
        if len(order) % 2 == 1:           # odd one out advances by default
            winners.append(scored[order[-1]])
        return winners, losers

    def _eliminate(self, winners: List[Tuple], target: int) -> List[Tuple]:
        """Elimination stage: shrink W to `target` by further competition."""
        pool = list(winners)
        while len(pool) > target:
            order = self.rng.permutation(len(pool))
            survivors = []
            for i in range(0, len(order) - 1, 2):
                a, b = pool[order[i]], pool[order[i + 1]]
                survivors.append(a if a[2] >= b[2] else b)
                if len(survivors) + (len(order) - i - 2) // 2 <= target:
                    pass
            if len(order) % 2 == 1:
                survivors.append(pool[order[-1]])
            if len(survivors) == len(pool):       # no progress; cut by rank
                survivors = sorted(pool, key=lambda t: t[2], reverse=True)[:target]
            pool = survivors
        return sorted(pool, key=lambda t: t[2], reverse=True)[:target]

    # ------------------------------------------------------------- run ---
    def run(self) -> SearchResult:
        from ..utils import timer

        population = initial_population(
            self.space, self.population_size, self.seed,
            families=self.families, backend=self.backend,
        )
        best: Optional[Tuple[CandidateModel, CandidateEvaluation, float]] = None

        LOG.info("meta-learning: %d candidates x %d rounds, objective = %s",
                 self.population_size, self.rounds, self.spec.describe())

        with timer() as elapsed:
            for round_idx in range(1, self.rounds + 1):
                scored = self._evaluate_population(population, round_idx)
                if not scored:
                    raise RuntimeError("every candidate in the population failed")

                round_best = max(scored, key=lambda t: t[2])
                if best is None or round_best[2] > best[2]:
                    best = round_best

                fitness_values = [f for _, _, f in scored]
                self.trace.add_round(
                    round=round_idx,
                    population_size=len(scored),
                    best_fitness=float(round_best[2]),
                    mean_fitness=float(np.mean(fitness_values)),
                    std_fitness=float(np.std(fitness_values)),
                    worst_fitness=float(np.min(fitness_values)),
                    best_family=round_best[0].family,
                    best_accuracy=float(round_best[1].accuracy),
                    best_shap_consistency=float(round_best[1].consistency.score),
                    best_complexity=float(round_best[1].complexity),
                    best_config_id=round_best[0].config_id,
                    running_best_fitness=float(best[2]),
                )
                LOG.info(
                    "round %d/%d | best %.4f (%s) | mean %.4f | acc %.4f | cons %.4f",
                    round_idx, self.rounds, round_best[2], round_best[0].family,
                    float(np.mean(fitness_values)), round_best[1].accuracy,
                    round_best[1].consistency.score,
                )

                if round_idx == self.rounds:
                    break

                winners, losers = self._compete(scored)
                survivors = self._eliminate(winners, self.survivors)

                # Next population: survivors kept, plus one evolved child
                # per open slot, each derived from a surviving winner.
                next_population = [w[0] for w in survivors]
                pairs = losers if losers else survivors
                while len(next_population) < self.population_size:
                    winner = survivors[int(self.rng.integers(len(survivors)))][0]
                    loser = pairs[int(self.rng.integers(len(pairs)))][0]
                    next_population.append(self._evolve(winner, loser))
                population = next_population

            search_time = elapsed()

        stats = self.evaluator.stats()
        LOG.info("meta-learning done in %.1fs (%d evaluations, %d cache hits)",
                 search_time, stats["evaluations"], stats["cache_hits"])

        return SearchResult(
            best_evaluation=best[1],
            best_fitness=float(best[2]),
            spec=self.spec,
            trace=self.trace,
            q_table=self.q_table.as_records(),
            n_evaluations=stats["evaluations"],
            n_cache_hits=stats["cache_hits"],
            search_time_s=search_time,
        )


def select_without_search(
    evaluator: CandidateEvaluator,
    candidates: List[CandidateModel],
    spec: FitnessSpec,
) -> SearchResult:
    """No-search baseline used by ablations A1 and A5.

    Evaluates a fixed set of default configurations under the same
    objective and picks the best. Same evaluation code, same folds, same
    metrics -- the only thing removed is the search. That isolation is
    what makes A1 and A5 interpretable.
    """
    from ..utils import timer

    trace = SearchTrace()
    scored = []
    need_shap = spec.use_shap_consistency

    with timer() as elapsed:
        for candidate in candidates:
            evaluation = evaluator.evaluate(candidate, need_shap=need_shap)
            fitness = evaluation.score(spec)
            record = evaluation.as_record(spec)
            record.update({"round": 0, "fitness_selected": fitness})
            trace.add_candidate(record)
            scored.append((candidate, evaluation, fitness))
        search_time = elapsed()

    best = max(scored, key=lambda t: t[2])
    trace.add_round(
        round=0, population_size=len(scored),
        best_fitness=float(best[2]),
        mean_fitness=float(np.mean([f for _, _, f in scored])),
        std_fitness=float(np.std([f for _, _, f in scored])),
        worst_fitness=float(np.min([f for _, _, f in scored])),
        best_family=best[0].family,
        best_accuracy=float(best[1].accuracy),
        best_shap_consistency=float(best[1].consistency.score),
        best_complexity=float(best[1].complexity),
        best_config_id=best[0].config_id,
        running_best_fitness=float(best[2]),
    )

    stats = evaluator.stats()
    return SearchResult(
        best_evaluation=best[1], best_fitness=float(best[2]), spec=spec,
        trace=trace, q_table=[], n_evaluations=stats["evaluations"],
        n_cache_hits=stats["cache_hits"], search_time_s=search_time,
    )
