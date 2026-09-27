"""Detection metrics, inference timing, and paired statistical tests."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "roc_auc", "mcc",
]


def compute_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    """Standard binary detection metrics.

    balanced_accuracy and MCC are included alongside accuracy because CAN
    window datasets are often heavily imbalanced -- a model that predicts
    'normal' for everything can post a high plain accuracy while
    detecting nothing.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
        if len(np.unique(y_true)) > 1 else float("nan"),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")

    if len(np.unique(np.concatenate([y_true, y_pred]))) <= 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        out["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return out


def measure_inference_time(candidate, X, repeats: int = 3,
                           batch_size: int = 1024) -> Dict[str, float]:
    """Per-window and per-batch inference latency.

    Reported per window in milliseconds, which is the number that maps
    onto an on-vehicle deployment budget. A warm-up pass is discarded so
    the figure is not dominated by first-call allocation.
    """
    import time

    X_batch = X.iloc[:batch_size] if hasattr(X, "iloc") else X[:batch_size]
    candidate.predict(X_batch)                       # warm-up, discarded

    timings = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        candidate.predict(X_batch)
        timings.append(time.perf_counter() - start)

    total = float(np.median(timings))
    n = len(X_batch)
    return {
        "inference_time_batch_s": total,
        "inference_time_per_window_ms": float(total / n * 1000.0) if n else float("nan"),
        "inference_batch_size": int(n),
    }


def aggregate_over_seeds(records: Sequence[Dict], group_keys: List[str],
                         metrics: List[str] | None = None) -> pd.DataFrame:
    """Mean and standard deviation of each metric over random seeds."""
    df = pd.DataFrame(list(records))
    if df.empty:
        return df
    metrics = metrics or [c for c in METRIC_COLUMNS + [
        "shap_consistency", "complexity", "fitness",
        "training_time_s", "inference_time_per_window_ms"] if c in df.columns]

    grouped = df.groupby(group_keys, dropna=False)[metrics]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    return summary.reset_index()


def paired_test(values_a: Sequence[float], values_b: Sequence[float],
                label_a: str = "A0", label_b: str = "ablation",
                alpha: float = 0.05) -> Dict[str, object]:
    """Paired comparison of two configurations across matched seeds.

    Paired because both configurations see the same seeds, the same
    splits and the same data -- the seed is the blocking factor.

    With the small number of seeds that fit an i3 budget (3-5), this test
    has very little power. The returned record therefore carries
    n_pairs and a `sufficient_power` flag, and the reporting layer refuses
    to print a significance claim when n_pairs < 5. A p-value from three
    paired observations is not evidence and should not be written up as
    though it were.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)

    record = {
        "comparison": f"{label_a} vs {label_b}",
        "n_pairs": int(n),
        "mean_a": float(a.mean()) if n else float("nan"),
        "mean_b": float(b.mean()) if n else float("nan"),
        "mean_difference": float((a - b).mean()) if n else float("nan"),
        "t_statistic": float("nan"),
        "p_value_ttest": float("nan"),
        "p_value_wilcoxon": float("nan"),
        "cohens_dz": float("nan"),
        "sufficient_power": bool(n >= 5),
        "significant_at_alpha": None,
        "note": "",
    }

    if n < 2 or np.allclose(a, b):
        record["note"] = ("identical or too few paired observations; "
                          "no test performed")
        return record

    t_stat, p_t = stats.ttest_rel(a, b)
    record["t_statistic"] = float(t_stat)
    record["p_value_ttest"] = float(p_t)

    diff = a - b
    sd = diff.std(ddof=1)
    record["cohens_dz"] = float(diff.mean() / sd) if sd > 0 else float("nan")

    if n >= 5:
        try:
            record["p_value_wilcoxon"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            pass
        record["significant_at_alpha"] = bool(p_t < alpha)
        record["note"] = f"paired t-test over {n} seeds, alpha={alpha}"
    else:
        record["note"] = (
            f"only {n} paired seeds: p-value reported for completeness but "
            "underpowered. Do not report this as a significance result."
        )
    return record
