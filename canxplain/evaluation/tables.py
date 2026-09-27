"""Automatic generation of Tables 1-8.

Every table is derived from the tracked run log plus a few artifacts, so
nothing has to be re-run to regenerate a table, and no number in a table
can exist that is not in runs.csv.

Tables are written as CSV (for further analysis) and Markdown (for
dropping into a draft).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..explain.consistency import MIN_FEATURES_FOR_COMPARABILITY
from ..utils import ensure_dir, get_logger

LOG = get_logger()

REPORT_METRICS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
    "shap_consistency", "complexity",
    "training_time_s", "inference_time_per_window_ms",
]


def _write(df: pd.DataFrame, out_dir: str, name: str, caption: str = "") -> str:
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    md_path = os.path.join(out_dir, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        if caption:
            fh.write(f"**{caption}**\n\n")
        fh.write(df.to_markdown(index=False, floatfmt=".4f"))
        fh.write("\n")
    LOG.info("wrote %s (%d rows)", csv_path, len(df))
    return csv_path


def _agg(df: pd.DataFrame, group_keys: List[str],
         metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Mean +/- std across seeds, formatted for reporting."""
    metrics = [m for m in (metrics or REPORT_METRICS) if m in df.columns]
    if df.empty or not metrics:
        return pd.DataFrame()

    grouped = df.groupby(group_keys, dropna=False)
    out = grouped[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    # `n_runs` is the number of rows pooled into the cell; when a table
    # groups over arms but not datasets, that is seeds x datasets, not
    # seeds. Reporting it as n_seeds overstated the seed count, so both
    # are now given and the seed count is the true nunique.
    out["n_runs"] = grouped.size().values
    if "seed" in df.columns:
        out["n_seeds"] = grouped["seed"].nunique().values
    else:
        out["n_seeds"] = out["n_runs"]
    return out.reset_index()


def format_mean_std(df: pd.DataFrame, metrics: Sequence[str],
                    decimals: int = 4) -> pd.DataFrame:
    """Collapse metric_mean / metric_std into a single 'x +/- y' column."""
    out = df.copy()
    for metric in metrics:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col not in out.columns:
            continue
        std = out[std_col] if std_col in out.columns else pd.Series(np.nan, index=out.index)
        out[metric] = [
            f"{m:.{decimals}f}" + (f" ± {s:.{decimals}f}" if np.isfinite(s) else "")
            for m, s in zip(out[mean_col], std)
        ]
        out = out.drop(columns=[c for c in (mean_col, std_col) if c in out.columns])
    return out


# ---------------------------------------------------------------- tables --
def table1_dataset_statistics(stream_stats: List[Dict], out_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(stream_stats).drop_duplicates(subset=["dataset"])
    _write(df, out_dir, "table1_dataset_statistics",
           "Table 1: Dataset statistics after adapter standardisation")
    return df


def table2_feature_description(groups: Sequence[str], out_dir: str) -> pd.DataFrame:
    from ..features.extractor import feature_table
    df = feature_table(groups)
    _write(df, out_dir, "table2_feature_description",
           "Table 2: Vendor-agnostic feature set (no payload content)")
    return df


def table3_candidate_configurations(cfg, out_dir: str) -> pd.DataFrame:
    """The searchable configuration space plus the fixed defaults."""
    rows = []
    for family in ("rf", "dnn"):
        space = cfg.models[family].space.to_dict()
        for axis, values in space.items():
            rows.append({
                "family": family.upper(),
                "hyperparameter": axis,
                "search_values": ", ".join(str(v) for v in values),
                "n_levels": len(values),
                "default_value": str(cfg.models.defaults[family].to_dict().get(axis, "-")),
            })
    df = pd.DataFrame(rows)
    _write(df, out_dir, "table3_candidate_configurations",
           "Table 3: Candidate model configuration space and fixed defaults "
           "(defaults are used by ablations A1 and A5)")
    return df


def table4_metalearning_search(trace_candidates: pd.DataFrame,
                               trace_rounds: pd.DataFrame,
                               out_dir: str) -> pd.DataFrame:
    """Per-round search progress plus the top candidates found."""
    if trace_rounds.empty:
        return pd.DataFrame()

    rounds = trace_rounds.copy()
    keep = [c for c in ["dataset", "seed", "arm", "round", "population_size",
                        "best_fitness", "mean_fitness", "std_fitness",
                        "best_family", "best_accuracy", "best_shap_consistency",
                        "best_complexity", "running_best_fitness"]
            if c in rounds.columns]
    _write(rounds[keep], out_dir, "table4_metalearning_search",
           "Table 4: Meta-learning competition rounds")

    if not trace_candidates.empty:
        cols = [c for c in ["arm", "dataset", "seed", "round", "family", "config_id",
                            "cv_accuracy", "shap_consistency", "complexity",
                            "fitness_selected"] if c in trace_candidates.columns]
        top = (trace_candidates[cols]
               .sort_values("fitness_selected", ascending=False)
               .head(25))
        _write(top, out_dir, "table4b_top_candidates",
               "Table 4b: Highest-fitness candidates encountered during the search")
    return rounds[keep]


def table5_full_performance(runs: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Full CANXplain (A0) performance, same-dataset."""
    subset = runs[(runs["arm"] == "A0") &
                  (runs["evaluation_type"] == "same_dataset")]
    if subset.empty:
        return pd.DataFrame()

    agg = _agg(subset, ["dataset_train", "model_family"])
    out = format_mean_std(agg, REPORT_METRICS)
    _write(out, out_dir, "table5_full_canxplain_performance",
           "Table 5: Full CANXplain (A0) same-dataset performance, "
           "mean ± std over seeds")
    return out


def table6_ablation(runs: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Every arm, both evaluation settings, every reported metric."""
    subset = runs[runs["evaluation_type"].isin(["same_dataset", "cross_dataset"])]
    if subset.empty:
        return pd.DataFrame()

    agg = _agg(subset, ["arm", "arm_name", "evaluation_type"])
    out = format_mean_std(agg, REPORT_METRICS)

    order = {code: i for i, code in enumerate(
        ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"])}
    out["__order"] = out["arm"].map(order).fillna(99)
    out = out.sort_values(["__order", "evaluation_type"]).drop(columns="__order")

    _write(out, out_dir, "table6_ablation_study",
           "Table 6: Ablation study. Every arm reported on accuracy, "
           "precision, recall, F1, ROC-AUC, SHAP consistency, complexity "
           "and timing, in both same-dataset and cross-dataset settings.")
    return out


def table7_cross_dataset(runs: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Source x target generalisation matrix."""
    subset = runs[runs["evaluation_type"].isin(["same_dataset", "cross_dataset"])]
    subset = subset[subset["arm"] == "A0"]
    if subset.empty:
        return pd.DataFrame()

    agg = _agg(subset, ["dataset_train", "dataset_test", "evaluation_type"])
    out = format_mean_std(agg, REPORT_METRICS)
    _write(out, out_dir, "table7_cross_dataset_generalization",
           "Table 7: Cross-dataset generalisation for full CANXplain (A0). "
           "Rows where train != test are the generalisation result; the "
           "target dataset was unseen during feature scaling, model "
           "selection and training.")

    if "f1" in agg.columns or "f1_mean" in agg.columns:
        pivot = agg.pivot_table(index="dataset_train", columns="dataset_test",
                                values="f1_mean", aggfunc="mean")
        _write(pivot.reset_index(), out_dir, "table7b_cross_dataset_f1_matrix",
               "Table 7b: Cross-dataset F1 matrix (rows = trained on, "
               "columns = tested on)")
    return out


def table8_shap_consistency(runs: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """SHAP consistency of the selected model, per arm.

    A2/A3/A5 do not use consistency in selection, but the consistency of
    whatever model they selected is still measured -- otherwise the arms
    could not be compared on the quantity the study is about.
    """
    subset = runs[runs["evaluation_type"] == "same_dataset"]
    if subset.empty:
        return pd.DataFrame()

    metrics = [c for c in ["shap_consistency", "shap_consistency_spearman",
                           "shap_consistency_kendall", "shap_jaccard_top5",
                           "shap_consistency_raw_mean", "shap_consistency_raw_std",
                           "n_features"]
               if c in subset.columns]
    agg = _agg(subset, ["arm", "arm_name", "dataset_train", "model_family"],
               metrics=metrics)
    out = format_mean_std(agg, metrics)

    # Arms that shrink the feature set produce a rank correlation over a
    # shorter vector, which is not on the same scale as a 23-feature arm.
    # Mark those rows instead of letting A6 look like the most stable
    # model in the study on the strength of having three features.
    if "n_features_mean" in agg.columns:
        out["comparable_to_A0"] = [
            "yes" if n >= MIN_FEATURES_FOR_COMPARABILITY else "no (short feature vector)"
            for n in agg["n_features_mean"]
        ]
        out["n_features"] = agg["n_features_mean"].astype(int)

    _write(out, out_dir, "table8_shap_consistency_comparison",
           "Table 8: SHAP consistency of the selected model per ablation arm. "
           "Higher is more stable; 0.5 means fold rankings are unrelated. "
           "Rows marked comparable_to_A0 = no use a different (shorter) "
           "feature vector, so their rank correlation is inflated by "
           "dimension and must not be read as a like-for-like comparison.")
    return out


def table_statistical_tests(test_records: List[Dict], out_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(test_records)
    if df.empty:
        return df
    _write(df, out_dir, "statistical_tests",
           "Paired comparisons of A0 against each ablation arm across "
           "matched seeds. Rows with sufficient_power=False are "
           "underpowered and must not be reported as significance results.")
    return df


def generate_all_tables(cfg, runs: pd.DataFrame, stream_stats: List[Dict],
                        trace_candidates: pd.DataFrame, trace_rounds: pd.DataFrame,
                        test_records: List[Dict], out_dir: str) -> Dict[str, pd.DataFrame]:
    ensure_dir(out_dir)
    tables = {
        "table1": table1_dataset_statistics(stream_stats, out_dir),
        "table2": table2_feature_description(list(cfg.features.groups), out_dir),
        "table3": table3_candidate_configurations(cfg, out_dir),
        "table4": table4_metalearning_search(trace_candidates, trace_rounds, out_dir),
        "table5": table5_full_performance(runs, out_dir),
        "table6": table6_ablation(runs, out_dir),
        "table7": table7_cross_dataset(runs, out_dir),
        "table8": table8_shap_consistency(runs, out_dir),
        "stats": table_statistical_tests(test_records, out_dir),
    }
    return {k: v for k, v in tables.items() if v is not None}
