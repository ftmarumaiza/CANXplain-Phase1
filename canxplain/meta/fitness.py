"""The proposed CANXplain fitness function.

    Fitness = alpha * Accuracy_norm
            + beta  * SHAP_Consistency
            - gamma * Complexity_norm

All three terms live in [0, 1] before weighting, so the weights mean
what they look like they mean. Getting that wrong is the usual way a
multi-objective selection criterion ends up dominated by whichever term
happens to have the largest numeric range.

NORMALISATION OF EACH TERM
--------------------------
Accuracy          Already in [0, 1]. Which accuracy is used is
                  configurable: 'balanced_accuracy' is the default
                  because CAN window datasets are imbalanced and plain
                  accuracy rewards a model that never fires.

SHAP_Consistency  Already mapped to [0, 1] in explain/consistency.py.

Complexity        Raw complexity spans several orders of magnitude
                  (a 50-tree forest has ~10^4 nodes; a 2x32 MLP has
                  ~10^3 parameters), so it is normalised on a LOG10
                  scale against the FIXED per-family reference bounds
                  from the search space:

                      C_norm = (log10(c) - log10(c_min))
                             / (log10(c_max) - log10(c_min))

                  Log scale because the deployment cost of going from
                  1k to 2k parameters is not the same as going from
                  100k to 101k. Fixed bounds because population-relative
                  normalisation would make fitness values incomparable
                  across rounds and across ablation arms.

DEFAULT WEIGHTS AND WHY
-----------------------
    alpha = 0.50, beta = 0.30, gamma = 0.20

The reasoning, stated so a reviewer can disagree with it explicitly:

  - Detection performance stays the largest single term. An IDS that
    explains itself beautifully and misses attacks is not an IDS. alpha
    must therefore exceed beta and gamma individually.

  - beta = 0.30 is set so that explanation stability can act as a
    TIE-BREAKER but not an OVERRIDE. On these datasets, candidates
    cluster within a few points of accuracy of each other; a 0.30
    weight lets consistency decide among near-equal candidates while
    still requiring roughly a 0.6-point consistency gain to justify each
    1-point accuracy sacrifice (0.5/0.3). It cannot rescue a materially
    worse detector.

  - gamma = 0.20 is the smallest term because complexity is a deployment
    constraint rather than a research objective. It is large enough to
    break ties toward the smaller model -- which is the whole point of
    A4 -- without letting a trivially small model win on parsimony
    alone.

  - alpha + beta = 0.80 > gamma * 1 means no candidate can win purely by
    being small: a maximally simple model scores at most 0.20 from the
    complexity term while giving up at most 0.80 from the other two.

These are defended defaults, not derived optima. There is no ground
truth for the correct trade-off between accuracy and explanation
stability -- it is a deployment policy choice. That is exactly why
experiments/run_sensitivity.py sweeps them and reports how the selected
model changes, and why any write-up should report the sweep rather than
the single default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

DEFAULT_WEIGHTS = {"alpha": 0.50, "beta": 0.30, "gamma": 0.20}


@dataclass
class FitnessSpec:
    """How fitness is computed for one experiment arm.

    The ablation arms are expressed purely as flags here, which keeps the
    meta-learning loop identical across A0/A2/A3/A4 -- only the objective
    changes, so any difference in outcome is attributable to the
    objective and not to a different search implementation.
    """
    alpha: float = DEFAULT_WEIGHTS["alpha"]
    beta: float = DEFAULT_WEIGHTS["beta"]
    gamma: float = DEFAULT_WEIGHTS["gamma"]
    use_shap_consistency: bool = True
    use_complexity: bool = True
    accuracy_metric: str = "balanced_accuracy"
    consistency_metric: str = "spearman"

    @classmethod
    def from_config(cls, cfg_fitness) -> "FitnessSpec":
        return cls(
            alpha=float(cfg_fitness.get("alpha", DEFAULT_WEIGHTS["alpha"])),
            beta=float(cfg_fitness.get("beta", DEFAULT_WEIGHTS["beta"])),
            gamma=float(cfg_fitness.get("gamma", DEFAULT_WEIGHTS["gamma"])),
            use_shap_consistency=bool(cfg_fitness.get("use_shap_consistency", True)),
            use_complexity=bool(cfg_fitness.get("use_complexity", True)),
            accuracy_metric=cfg_fitness.get("accuracy_metric", "balanced_accuracy"),
            consistency_metric=cfg_fitness.get("consistency_metric", "spearman"),
        )

    def variant(self, **changes) -> "FitnessSpec":
        merged = {**self.__dict__, **changes}
        return FitnessSpec(**merged)

    def describe(self) -> str:
        terms = [f"{self.alpha:g}*{self.accuracy_metric}"]
        if self.use_shap_consistency and self.beta:
            terms.append(f"{self.beta:g}*SHAPconsistency")
        if self.use_complexity and self.gamma:
            terms.append(f"-{self.gamma:g}*complexity_norm")
        return " + ".join(terms).replace("+ -", "- ")

    def as_record(self) -> Dict[str, object]:
        return {
            "fitness_alpha": self.alpha,
            "fitness_beta": self.beta if self.use_shap_consistency else 0.0,
            "fitness_gamma": self.gamma if self.use_complexity else 0.0,
            "fitness_uses_shap": self.use_shap_consistency,
            "fitness_uses_complexity": self.use_complexity,
            "fitness_accuracy_metric": self.accuracy_metric,
            "fitness_expression": self.describe(),
        }


def normalise_complexity(
    raw_complexity: float,
    family: str,
    reference_bounds: Dict[str, tuple],
) -> float:
    """Log10 min-max normalisation against fixed search-space bounds."""
    low, high = reference_bounds.get(family, (1.0, 1e6))
    raw = max(float(raw_complexity), 1.0)
    log_raw, log_low, log_high = np.log10(raw), np.log10(max(low, 1.0)), np.log10(max(high, 10.0))
    if log_high <= log_low:
        return 0.0
    return float(np.clip((log_raw - log_low) / (log_high - log_low), 0.0, 1.0))


def compute_fitness(
    accuracy: float,
    shap_consistency: Optional[float],
    complexity_norm: Optional[float],
    spec: FitnessSpec,
) -> float:
    """Combine the normalised terms under the given objective spec."""
    score = spec.alpha * float(accuracy)

    if spec.use_shap_consistency:
        if shap_consistency is None or not np.isfinite(shap_consistency):
            raise ValueError(
                "fitness requires SHAP consistency but none was supplied; "
                "this usually means the SHAP stage failed silently"
            )
        score += spec.beta * float(shap_consistency)

    if spec.use_complexity:
        if complexity_norm is None or not np.isfinite(complexity_norm):
            raise ValueError("fitness requires complexity but none was supplied")
        score -= spec.gamma * float(complexity_norm)

    return float(score)


def fitness_breakdown(
    accuracy: float,
    shap_consistency: Optional[float],
    complexity_norm: Optional[float],
    spec: FitnessSpec,
) -> Dict[str, float]:
    """Per-term contributions, for Table 4 and for debugging a surprise."""
    acc_term = spec.alpha * float(accuracy)
    shap_term = (spec.beta * float(shap_consistency)
                 if spec.use_shap_consistency and shap_consistency is not None else 0.0)
    cx_term = (-spec.gamma * float(complexity_norm)
               if spec.use_complexity and complexity_norm is not None else 0.0)
    return {
        "fitness_accuracy_term": acc_term,
        "fitness_shap_term": shap_term,
        "fitness_complexity_term": cx_term,
        "fitness": acc_term + shap_term + cx_term,
    }


ACCURACY_ONLY = FitnessSpec(
    alpha=1.0, beta=0.0, gamma=0.0,
    use_shap_consistency=False, use_complexity=False,
)
