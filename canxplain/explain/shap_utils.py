"""SHAP computation, with explainer choice driven by model family.

  Random Forest : TreeExplainer. Exact Shapley values for tree ensembles
                  in polynomial time -- no sampling noise, which matters
                  because a consistency score computed on noisy
                  attributions would measure sampler variance rather
                  than model stability.

  DNN           : PermutationExplainer over a small background set.
                  KernelExplainer is the usual fallback but costs
                  O(2^features) sampling and is not viable inside a
                  meta-learning loop on an i3. With ~15-20 features,
                  permutation SHAP with a capped evaluation budget gives
                  a stable global importance ranking in seconds.

Global importance for a feature is mean(|SHAP value|) over the explained
sample -- the standard global aggregation, and the one ACHILLES uses for
its feature-importance figures.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger

LOG = get_logger()


def subsample(X, n: int, seed: int = 0):
    """Deterministic row subsample used for SHAP background/explain sets."""
    if n is None or len(X) <= n:
        return X
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=n, replace=False))
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _collapse_binary(values: np.ndarray) -> np.ndarray:
    """Reduce SHAP output to (n_samples, n_features).

    shap returns different shapes across versions and explainers for
    binary classifiers: a list of two arrays, or (n, f, 2), or (n, f).
    For a binary problem the two class attributions are mirror images,
    so taking the positive class (or the single array) is correct.
    """
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def shap_importance(
    candidate,
    X_explain,
    X_background=None,
    max_evals: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Global SHAP importance vector: mean |SHAP| per feature."""
    import shap

    feature_names = list(X_explain.columns) if hasattr(X_explain, "columns") else None
    X_arr = X_explain.to_numpy() if hasattr(X_explain, "to_numpy") else np.asarray(X_explain)

    if candidate.family == "rf":
        explainer = shap.TreeExplainer(candidate.estimator)
        raw = explainer.shap_values(X_arr, check_additivity=False)
        if isinstance(raw, list):
            raw = raw[-1]
        values = _collapse_binary(raw)
    else:
        if X_background is None:
            X_background = X_explain
        bg = X_background.to_numpy() if hasattr(X_background, "to_numpy") else np.asarray(X_background)
        explainer = shap.PermutationExplainer(
            candidate.positive_score, bg, seed=seed
        )
        raw = explainer(X_arr, max_evals=max_evals, silent=True)
        values = _collapse_binary(raw.values)

    importance = np.abs(values).mean(axis=0)
    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)

    if feature_names is not None and len(importance) != len(feature_names):
        raise RuntimeError(
            f"SHAP returned {len(importance)} importances for "
            f"{len(feature_names)} features"
        )
    return importance.astype(np.float64)


def shap_importance_frame(importance: np.ndarray, feature_names) -> pd.DataFrame:
    df = pd.DataFrame({"feature": list(feature_names),
                       "mean_abs_shap": np.asarray(importance, dtype=float)})
    total = df["mean_abs_shap"].sum()
    df["share"] = df["mean_abs_shap"] / total if total > 0 else 0.0
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def shap_budget_for(
    n_rows: int, cfg_shap, split_name: str = "val"
) -> Tuple[int, Optional[int]]:
    """Resolve (explain_sample_size, background_size) from config."""
    explain_n = min(int(cfg_shap.get("max_explain_sample", 200)), n_rows)
    background_n = int(cfg_shap.get("max_background", 100))
    return explain_n, background_n
