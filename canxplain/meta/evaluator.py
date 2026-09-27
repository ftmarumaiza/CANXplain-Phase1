"""Evaluating one candidate configuration -- the inner loop of the search.

WHAT HAPPENS PER CANDIDATE
--------------------------
Given only the TRAINING split:

  for each blocked, purged CV fold k:
      fit candidate on fold-train
      predict fold-validation        -> accuracy_k
      SHAP on a fold-validation sample -> importance vector I_k
      record fitted complexity        -> c_k

  accuracy   = mean_k accuracy_k
  consistency = rank agreement across {I_k}   (explain/consistency.py)
  complexity  = mean_k c_k

Nothing here touches the test split or, in the cross-dataset setting,
the target dataset. That is the entire reason the CV is blocked and
purged rather than shuffled.

CACHING
-------
Survivors are re-evaluated in later rounds, and the accuracy /
consistency / complexity of a candidate do not depend on the fitness
WEIGHTS. So the expensive part is cached on
(config_id, seed, data fingerprint) and only the cheap weighted
combination is recomputed per objective. On an i3 this is the
difference between a search that finishes and one that does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.windowing import BlockedKFold
from ..evaluation.metrics import compute_metrics
from ..explain.consistency import ConsistencyResult, compute_consistency
from ..explain.shap_utils import shap_importance, subsample
from ..models.registry import CandidateModel
from ..utils import get_logger, stable_hash, timer
from .fitness import FitnessSpec, compute_fitness, fitness_breakdown, normalise_complexity

LOG = get_logger()


@dataclass
class CandidateEvaluation:
    """Objective-independent evaluation of one candidate."""
    candidate: CandidateModel
    accuracy: float                       # mean over folds, chosen metric
    fold_metrics: List[Dict[str, float]]
    consistency: ConsistencyResult
    complexity: float                     # mean fitted complexity over folds
    complexity_norm: float
    training_time_s: float
    shap_time_s: float
    feature_names: List[str] = field(default_factory=list)

    def score(self, spec: FitnessSpec) -> float:
        return compute_fitness(
            self.accuracy,
            self.consistency.score if spec.use_shap_consistency else None,
            self.complexity_norm if spec.use_complexity else None,
            spec,
        )

    def as_record(self, spec: FitnessSpec) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            **self.candidate.describe(),
            "cv_accuracy": self.accuracy,
            "complexity": self.complexity,
            "complexity_norm": self.complexity_norm,
            "training_time_s": self.training_time_s,
            "shap_time_s": self.shap_time_s,
        }
        record.update(self.consistency.to_record())
        record.update(fitness_breakdown(
            self.accuracy,
            self.consistency.score if spec.use_shap_consistency else None,
            self.complexity_norm if spec.use_complexity else None,
            spec,
        ))
        record.update(spec.as_record())
        for key in ("balanced_accuracy", "f1", "precision", "recall", "roc_auc"):
            values = [m.get(key, np.nan) for m in self.fold_metrics]
            record[f"cv_{key}_mean"] = float(np.nanmean(values)) if values else np.nan
        return record


class CandidateEvaluator:
    """Evaluates candidates against a fixed training split, with a cache."""

    def __init__(self, cfg, X_train, y_train, reference_bounds: Dict[str, tuple],
                 seed: int = 0, data_tag: str = ""):
        self.cfg = cfg
        self.X_train = X_train
        self.y_train = np.asarray(y_train)
        self.reference_bounds = reference_bounds
        self.seed = int(seed)
        self.feature_names = list(X_train.columns)

        meta = cfg.meta
        self.n_folds = int(meta.get("cv_folds", 3))
        self.purge = int(cfg.data.window_size) - 1
        self.accuracy_metric = cfg.fitness.get("accuracy_metric", "balanced_accuracy")
        self.consistency_metric = cfg.fitness.get("consistency_metric", "spearman")
        self.topk = int(cfg.meta.shap.get("topk", 5))

        self.shap_explain_n = int(meta.shap.get("max_explain_sample", 200))
        self.shap_background_n = int(meta.shap.get("max_background", 100))
        self.shap_max_evals = int(meta.shap.get("dnn_max_evals", 200))

        self.data_fingerprint = stable_hash({
            "tag": data_tag,
            "shape": list(np.shape(X_train)),
            "features": self.feature_names,
            "y": [int(self.y_train.sum()), int(len(self.y_train))],
        })
        self._cache: Dict[str, CandidateEvaluation] = {}
        self.n_evaluations = 0
        self.n_cache_hits = 0

    # ------------------------------------------------------------------
    def cache_key(self, candidate: CandidateModel) -> str:
        return stable_hash({
            "cfg": candidate.config_id,
            "seed": self.seed,
            "data": self.data_fingerprint,
            "folds": self.n_folds,
            "shap": [self.shap_explain_n, self.shap_background_n, self.shap_max_evals],
        })

    def _key(self, candidate: CandidateModel, need_shap: bool) -> str:
        # need_shap changes what the result CONTAINS (a real consistency
        # score vs NaN), so two calls with different need_shap must not
        # share a cache entry.
        return stable_hash({"base": self.cache_key(candidate), "shap": bool(need_shap)})

    def evaluate(self, candidate: CandidateModel,
                 need_shap: bool = True) -> CandidateEvaluation:
        key = self._key(candidate, need_shap)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]

        result = self._evaluate_uncached(candidate, need_shap=need_shap)
        self._cache[key] = result
        self.n_evaluations += 1
        return result

    # ------------------------------------------------------------------
    def _evaluate_uncached(self, candidate: CandidateModel,
                           need_shap: bool) -> CandidateEvaluation:
        splitter = BlockedKFold(n_splits=self.n_folds, purge=self.purge)
        fold_metrics: List[Dict[str, float]] = []
        importances: List[np.ndarray] = []
        complexities: List[float] = []
        train_seconds = 0.0
        shap_seconds = 0.0

        for fold_idx, (train_idx, val_idx) in enumerate(
                splitter.split(self.X_train, self.y_train)):
            X_tr = self.X_train.iloc[train_idx]
            y_tr = self.y_train[train_idx]
            X_va = self.X_train.iloc[val_idx]
            y_va = self.y_train[val_idx]

            if len(np.unique(y_tr)) < 2:
                LOG.debug("fold %d has a single class in training; skipped", fold_idx)
                continue

            fold_model = candidate.clone(seed=self.seed + fold_idx)
            with timer() as elapsed:
                fold_model.fit(X_tr, y_tr)
            train_seconds += elapsed()

            y_pred = fold_model.predict(X_va)
            y_score = fold_model.positive_score(X_va)
            fold_metrics.append(compute_metrics(y_va, y_pred, y_score))
            complexities.append(fold_model.complexity())

            if need_shap:
                with timer() as elapsed:
                    X_explain = subsample(X_va, self.shap_explain_n,
                                          seed=self.seed + fold_idx)
                    X_background = subsample(X_tr, self.shap_background_n,
                                             seed=self.seed + fold_idx)
                    importances.append(shap_importance(
                        fold_model, X_explain, X_background,
                        max_evals=self.shap_max_evals, seed=self.seed + fold_idx,
                    ))
                shap_seconds += elapsed()

        if not fold_metrics:
            raise RuntimeError(
                f"candidate {candidate.config_id} produced no usable CV folds. "
                "The training split is probably single-class -- check the "
                "chronological split against the attack distribution."
            )

        accuracy = float(np.nanmean(
            [m.get(self.accuracy_metric, m["accuracy"]) for m in fold_metrics]))

        if need_shap and importances:
            consistency = compute_consistency(
                importances, self.feature_names,
                metric=self.consistency_metric, topk=self.topk)
        else:
            # A3/A2 skip SHAP entirely to save compute. A neutral 0.5 is
            # stored so the field is never silently absent from a record,
            # but it is NOT used in those arms' fitness.
            n_features = len(self.feature_names)
            consistency = compute_consistency(
                [np.zeros(n_features)], self.feature_names,
                metric=self.consistency_metric, topk=self.topk)
            consistency.score = float("nan")

        complexity = float(np.mean(complexities)) if complexities else float("nan")
        complexity_norm = normalise_complexity(
            complexity, candidate.family, self.reference_bounds)

        return CandidateEvaluation(
            candidate=candidate,
            accuracy=accuracy,
            fold_metrics=fold_metrics,
            consistency=consistency,
            complexity=complexity,
            complexity_norm=complexity_norm,
            training_time_s=train_seconds,
            shap_time_s=shap_seconds,
            feature_names=self.feature_names,
        )

    def stats(self) -> Dict[str, int]:
        return {"evaluations": self.n_evaluations, "cache_hits": self.n_cache_hits}
