"""Candidate model families: Random Forest and a small DNN.

Both are wrapped in a single CandidateModel interface so the
meta-learning loop never has to branch on family, and so complexity is
measured in a way that is comparable across families.

COMPLEXITY DEFINITION
---------------------
Complexity is the count of learned parameters in the fitted model:

  Random Forest : total node count across all trees, summed from the
                  fitted estimators (not the n_estimators * 2^max_depth
                  upper bound -- trees stop early, and using the bound
                  would penalise deep-but-sparse forests unfairly).

  DNN           : total trainable parameters, sum over layers of
                  (fan_in * fan_out + fan_out).

Both are "number of learned quantities the ECU must store and traverse",
which is the quantity that actually matters for on-vehicle deployment.
They are not on the same absolute scale across families, which is why
fitness.py normalises complexity on a log10 scale against fixed
reference bounds derived from the search space rather than comparing
raw counts.

DNN BACKEND
-----------
Default is scikit-learn's MLPClassifier: no extra dependency, CPU-only,
and fast enough for a meta-learning loop on an i3. Its one limitation is
that it has no dropout -- it regularises with L2 (alpha). If the config
asks for dropout, set models.dnn.backend: torch, which uses the optional
PyTorch backend below. The README explains the trade-off.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from ..utils import get_logger, stable_hash

LOG = get_logger()

FAMILIES = ("rf", "dnn")


# ---------------------------------------------------------------- RF ------
def build_rf(params: Dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=params.get("max_depth", None),
        min_samples_split=int(params.get("min_samples_split", 2)),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        max_features=params.get("max_features", "sqrt"),
        class_weight=params.get("class_weight", None),
        bootstrap=True,
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=seed,
    )


def rf_complexity(model: RandomForestClassifier) -> float:
    """Total node count across the fitted forest."""
    return float(sum(est.tree_.node_count for est in model.estimators_))


# --------------------------------------------------------------- DNN ------
def build_sklearn_dnn(params: Dict[str, Any], seed: int) -> MLPClassifier:
    n_layers = int(params.get("n_hidden_layers", 2))
    width = int(params.get("neurons_per_layer", 64))
    taper = float(params.get("taper", 1.0))   # 0.5 -> each layer half the last

    sizes = []
    current = width
    for _ in range(n_layers):
        sizes.append(max(4, int(round(current))))
        current *= taper

    return MLPClassifier(
        hidden_layer_sizes=tuple(sizes),
        learning_rate_init=float(params.get("learning_rate", 1e-3)),
        batch_size=int(params.get("batch_size", 256)),
        max_iter=int(params.get("epochs", 40)),
        alpha=float(params.get("alpha", 1e-4)),
        activation=params.get("activation", "relu"),
        solver="adam",
        early_stopping=False,     # validation is handled by BlockedKFold
        shuffle=False,            # windows are temporal; do not reshuffle
        random_state=seed,
    )


def mlp_complexity(model: MLPClassifier) -> float:
    """Trainable parameter count of a fitted MLP."""
    total = 0
    for w, b in zip(model.coefs_, model.intercepts_):
        total += int(np.prod(w.shape)) + int(np.prod(b.shape))
    return float(total)


def theoretical_dnn_complexity(params: Dict[str, Any], n_features: int) -> float:
    """Parameter count without fitting -- used for search-space bounds."""
    n_layers = int(params.get("n_hidden_layers", 2))
    width = int(params.get("neurons_per_layer", 64))
    taper = float(params.get("taper", 1.0))
    sizes, current = [], width
    for _ in range(n_layers):
        sizes.append(max(4, int(round(current))))
        current *= taper
    total, fan_in = 0, n_features
    for size in sizes:
        total += fan_in * size + size
        fan_in = size
    total += fan_in * 2 + 2      # binary output layer
    return float(total)


# --------------------------------------------------- unified wrapper ------
class CandidateModel:
    """One (family, hyperparameters, seed) candidate.

    Holds nothing but configuration until fit() is called, so a
    population of candidates is cheap to carry around.
    """

    def __init__(self, family: str, params: Dict[str, Any], seed: int = 0,
                 backend: str = "sklearn"):
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {family}")
        self.family = family
        self.params = dict(params)
        self.seed = int(seed)
        self.backend = backend
        self.estimator = None
        self.n_features_ = None

    # -- identity ---------------------------------------------------------
    @property
    def config_id(self) -> str:
        return f"{self.family}-{stable_hash({'f': self.family, 'p': self.params})}"

    def describe(self) -> Dict[str, Any]:
        return {"family": self.family, "config_id": self.config_id,
                **{f"hp_{k}": v for k, v in self.params.items()}}

    def clone(self, params: Optional[Dict[str, Any]] = None,
              seed: Optional[int] = None) -> "CandidateModel":
        return CandidateModel(
            self.family,
            dict(self.params if params is None else params),
            self.seed if seed is None else seed,
            self.backend,
        )

    # -- lifecycle --------------------------------------------------------
    def _build(self):
        if self.family == "rf":
            return build_rf(self.params, self.seed)
        if self.backend == "torch":
            from .torch_dnn import build_torch_dnn      # optional dependency
            return build_torch_dnn(self.params, self.seed)
        return build_sklearn_dnn(self.params, self.seed)

    def fit(self, X, y) -> "CandidateModel":
        self.estimator = self._build()
        self.n_features_ = X.shape[1]
        import warnings
        with warnings.catch_warnings():
            # MLP convergence warnings are expected: the epoch budget is
            # a searched hyperparameter, not an accident.
            warnings.filterwarnings("ignore")
            self.estimator.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        proba = self.estimator.predict_proba(X)
        return np.asarray(proba)

    def positive_score(self, X) -> np.ndarray:
        """P(attack). Handles the degenerate single-class-fit case."""
        proba = self.predict_proba(X)
        classes = getattr(self.estimator, "classes_", np.array([0, 1]))
        if proba.shape[1] == 1:
            return np.full(len(X), float(classes[0]))
        pos = int(np.where(np.asarray(classes) == 1)[0][0]) if 1 in classes else 1
        return proba[:, pos]

    def complexity(self) -> float:
        if self.estimator is None:
            raise RuntimeError("fit() before complexity()")
        if self.family == "rf":
            return rf_complexity(self.estimator)
        if self.backend == "torch":
            return float(sum(p.numel() for p in self.estimator.parameters()))
        return mlp_complexity(self.estimator)

    def __repr__(self):
        return f"<CandidateModel {self.family} {self.params}>"


def default_candidate(family: str, cfg, seed: int = 0) -> CandidateModel:
    """The fixed, non-searched configuration used by ablations A1 and A5.

    These are the conventional textbook defaults a practitioner would
    reach for without any search -- deliberately reasonable, not
    deliberately weak. An ablation that compares against a strawman
    proves nothing.
    """
    defaults = cfg.models.defaults[family].to_dict()
    backend = cfg.models.dnn.get("backend", "sklearn") if family == "dnn" else "sklearn"
    return CandidateModel(family, defaults, seed=seed, backend=backend)
