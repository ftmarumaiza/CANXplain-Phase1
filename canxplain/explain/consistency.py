"""SHAP consistency: how stable a model's explanation is across folds.

WHAT IS BEING MEASURED AND WHY
------------------------------
Two models can score identical accuracy while disagreeing completely
about WHY a window is an attack. Retrain one of them on a slightly
different slice of the same traffic and its top features reshuffle; the
other keeps the same ranking. The second model is the one you can
deploy, audit, and defend to a safety reviewer. Accuracy alone cannot
tell them apart. This metric can.

DEFINITION
----------
For a candidate configuration and a K-fold blocked, purged split of the
TRAINING data only:

  1. For each fold k, fit the candidate on the fold-train part and
     compute the global SHAP importance vector on the fold-validation
     part:   I_k in R^F.
  2. Convert each I_k to a rank vector R_k (rank 1 = most important).
  3. Compute the rank correlation between every pair of folds:
         rho_ij = corr(R_i, R_j),  i < j
  4. Consistency is the mean pairwise correlation, rescaled to [0, 1]:
         C = (mean(rho) + 1) / 2

The rescaling matters: raw Spearman lives in [-1, 1], and the fitness
function adds it to accuracy, which lives in [0, 1]. Combining the two
without rescaling would let a model with anti-correlated explanations
subtract from its own accuracy term at an arbitrary exchange rate.
C = 1.0 means the feature ranking is identical across every fold;
C = 0.5 means the rankings are unrelated.

WHICH RANK CORRELATION
----------------------
Default is SPEARMAN. With F in the 15-25 range, Spearman uses the full
rank vector and is sensitive to movement anywhere in the ranking, which
is what "is the explanation stable?" should mean. Kendall's tau is
available (consistency.metric: kendall); it is more robust to a few
large rank swaps and more conservative, so it is the right choice if a
reviewer objects to Spearman's sensitivity to tail features. Both are
reported in the diagnostics either way, so the choice never has to be
defended blind.

A third view, Jaccard overlap of the top-k feature sets, is also
reported. It answers a different and more operational question -- "do
the folds agree on which features matter at all?" -- and is not used in
the fitness function because it discards the ordering.

LEAKAGE
-------
Every quantity here comes from the training split. The test set is never
touched. For cross-dataset evaluation, the target dataset is not
involved in this computation at any point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

SUPPORTED_METRICS = ("spearman", "kendall")

# A rank correlation over a very short feature vector is close to
# meaningless: with F = 3 there are only six possible rankings, so two
# folds agree perfectly by chance far too often. Arms that change the
# feature set (A6/A7/A8) therefore produce consistency values that are
# NOT on the same scale as the full-feature arms. Rather than silently
# dropping the number, every result carries the feature count and a
# comparability flag so tables can mark it.
MIN_FEATURES_FOR_COMPARABILITY = 5


@dataclass
class ConsistencyResult:
    """Everything needed to report and plot the consistency of one candidate."""
    score: float                          # the [0,1] value used in fitness
    metric: str
    mean_raw_correlation: float           # mean pairwise rho in [-1, 1]
    std_raw_correlation: float
    pairwise: np.ndarray                  # (K, K) correlation matrix
    importance_matrix: np.ndarray         # (K, F) mean|SHAP| per fold
    rank_matrix: np.ndarray               # (K, F) ranks, 1 = most important
    feature_names: List[str] = field(default_factory=list)
    spearman_score: float = np.nan        # both reported regardless of choice
    kendall_score: float = np.nan
    jaccard_topk: float = np.nan
    topk: int = 5
    n_folds: int = 0
    n_features: int = 0

    @property
    def comparable(self) -> bool:
        """False when the feature vector is too short for the rank
        correlation to be comparable against a full-feature arm."""
        return self.n_features >= MIN_FEATURES_FOR_COMPARABILITY

    def to_record(self) -> Dict[str, float]:
        return {
            "shap_consistency": float(self.score),
            "shap_consistency_metric": self.metric,
            "shap_consistency_raw_mean": float(self.mean_raw_correlation),
            "shap_consistency_raw_std": float(self.std_raw_correlation),
            "shap_consistency_spearman": float(self.spearman_score),
            "shap_consistency_kendall": float(self.kendall_score),
            "shap_jaccard_top%d" % self.topk: float(self.jaccard_topk),
            "shap_consistency_folds": int(self.n_folds),
            "shap_consistency_n_features": int(self.n_features),
            "shap_consistency_comparable": bool(self.comparable),
        }

    def mean_importance(self) -> np.ndarray:
        return self.importance_matrix.mean(axis=0)


def _rank(importance: np.ndarray) -> np.ndarray:
    """Rank 1 = most important. Ties get the average rank."""
    return rankdata(-np.asarray(importance, dtype=float), method="average")


def _pairwise_matrix(ranks: np.ndarray, metric: str) -> np.ndarray:
    k = len(ranks)
    out = np.eye(k, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            if metric == "kendall":
                rho = kendalltau(ranks[i], ranks[j]).correlation
            else:
                rho = spearmanr(ranks[i], ranks[j]).correlation
            rho = 0.0 if not np.isfinite(rho) else float(rho)
            out[i, j] = out[j, i] = rho
    return out


def _mean_offdiag(matrix: np.ndarray) -> tuple[float, float]:
    k = len(matrix)
    if k < 2:
        return np.nan, np.nan
    iu = np.triu_indices(k, k=1)
    values = matrix[iu]
    return float(np.mean(values)), float(np.std(values))


def jaccard_top_k(ranks: np.ndarray, k: int = 5) -> float:
    """Mean pairwise Jaccard overlap of the top-k feature sets."""
    n_folds, n_features = ranks.shape
    k = min(k, n_features)
    if n_folds < 2:
        return np.nan
    tops = [set(np.argsort(r)[:k].tolist()) for r in ranks]
    scores = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            union = tops[i] | tops[j]
            scores.append(len(tops[i] & tops[j]) / len(union) if union else 0.0)
    return float(np.mean(scores))


def compute_consistency(
    importance_per_fold: Sequence[np.ndarray],
    feature_names: Sequence[str] | None = None,
    metric: str = "spearman",
    topk: int = 5,
) -> ConsistencyResult:
    """Turn per-fold SHAP importance vectors into a consistency result."""
    metric = metric.lower()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"consistency metric must be one of {SUPPORTED_METRICS}, got '{metric}'"
        )

    matrix = np.vstack([np.asarray(v, dtype=float) for v in importance_per_fold])
    n_folds, n_features = matrix.shape
    ranks = np.vstack([_rank(row) for row in matrix])

    if n_folds < 2:
        # A single fold carries no information about stability. Returning
        # a neutral 0.5 rather than 1.0 avoids silently rewarding an
        # under-specified CV setting with a perfect consistency score.
        return ConsistencyResult(
            score=0.5, metric=metric, mean_raw_correlation=np.nan,
            std_raw_correlation=np.nan, pairwise=np.ones((1, 1)),
            importance_matrix=matrix, rank_matrix=ranks,
            feature_names=list(feature_names or []), n_folds=n_folds, topk=topk,
            n_features=int(n_features),
        )

    chosen = _pairwise_matrix(ranks, metric)
    mean_rho, std_rho = _mean_offdiag(chosen)
    score = (mean_rho + 1.0) / 2.0

    spearman_mean, _ = _mean_offdiag(_pairwise_matrix(ranks, "spearman"))
    kendall_mean, _ = _mean_offdiag(_pairwise_matrix(ranks, "kendall"))

    return ConsistencyResult(
        score=float(np.clip(score, 0.0, 1.0)),
        metric=metric,
        mean_raw_correlation=mean_rho,
        std_raw_correlation=std_rho,
        pairwise=chosen,
        importance_matrix=matrix,
        rank_matrix=ranks,
        feature_names=list(feature_names or [f"f{i}" for i in range(n_features)]),
        spearman_score=(spearman_mean + 1.0) / 2.0,
        kendall_score=(kendall_mean + 1.0) / 2.0,
        jaccard_topk=jaccard_top_k(ranks, topk),
        topk=topk,
        n_folds=n_folds,
        n_features=int(n_features),
    )
