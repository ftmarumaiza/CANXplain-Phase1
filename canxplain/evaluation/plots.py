"""Figures 1-7.

Plotting rules applied throughout, because the spec asks for figures
that do not imply unsupported results:

  * Bar charts start at zero. A truncated y-axis turns a 0.01 difference
    into a visual landslide.
  * Error bars are drawn wherever more than one seed exists, and the
    seed count is written into the axis label. A bar with no error bar
    says "one seed" rather than pretending to precision.
  * Missing data is left blank, never interpolated or zero-filled.
  * Nothing is annotated as "better" or "significant" by the plotting
    code. Figures show numbers; claims belong in text that a human has
    taken responsibility for.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import ensure_dir, get_logger

LOG = get_logger()

DPI = 150
PALETTE = ["#2E5A88", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD",
           "#937860", "#DA8BC3", "#8C8C8C"]


def _save(fig, out_dir: str, name: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", path)
    return path


def _agg(df: pd.DataFrame, keys: List[str], metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    out = df.groupby(keys, dropna=False)[metric].agg(["mean", "std", "count"])
    return out.reset_index()


# ----------------------------------------------------- Figure 1: pipeline --
def figure1_pipeline(out_dir: str) -> str:
    """Schematic of the Phase 1 pipeline."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    stages = [
        ("Raw CAN logs\n(Car-Hacking, CAN-IDS,\nORNL, SAD)", 0.3, "#E8EEF5"),
        ("Dataset adapter\ntimestamp | CAN_ID\nDLC | label", 2.4, "#E8EEF5"),
        ("Chronological split\n+ purge gap\n(no window overlap)", 4.5, "#FFF0E0"),
        ("Sliding window\nn=10, step=1", 6.6, "#FFF0E0"),
        ("Vendor-agnostic features\ntemporal + frequency\n(NO payload)", 8.7, "#E6F3E6"),
        ("StandardScaler\nfitted on TRAIN only", 10.8, "#E6F3E6"),
    ]
    for label, x, colour in stages:
        ax.add_patch(plt.Rectangle((x, 3.5), 1.9, 1.3, facecolor=colour,
                                   edgecolor="#333", linewidth=1.2))
        ax.text(x + 0.95, 4.15, label, ha="center", va="center", fontsize=7.6)
        if x < 10.8:
            ax.annotate("", xy=(x + 2.1, 4.15), xytext=(x + 1.9, 4.15),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))

    lower = [
        ("Candidate population\nRF + small DNN", 0.3, "#F3E8F5"),
        ("ACHILLES-inspired\ncompetition & evolution\n(Q-table, exploit/explore)", 2.4, "#F3E8F5"),
        ("Blocked purged CV\nSHAP per fold", 4.5, "#FDE8E8"),
        ("SHAP consistency\n(rank agreement)", 6.6, "#FDE8E8"),
        ("Fitness\nαAcc + βCons − γCx", 8.7, "#FDE8E8"),
        ("Selected model\n+ scaler bundle", 10.8, "#E8EEF5"),
    ]
    for label, x, colour in lower:
        ax.add_patch(plt.Rectangle((x, 1.4), 1.9, 1.3, facecolor=colour,
                                   edgecolor="#333", linewidth=1.2))
        ax.text(x + 0.95, 2.05, label, ha="center", va="center", fontsize=7.6)
        if x < 10.8:
            ax.annotate("", xy=(x + 2.1, 2.05), xytext=(x + 1.9, 2.05),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))

    ax.annotate("", xy=(0.9, 2.8), xytext=(11.75, 3.5),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.2,
                                connectionstyle="arc3,rad=0.15", linestyle="--"))

    for x, label, colour in [(3.0, "Same-dataset test\n(held out, used once)", "#DCE9F7"),
                             (7.5, "Cross-dataset test\n(target never seen during\n"
                                   "scaling, selection or training)", "#DCE9F7")]:
        ax.add_patch(plt.Rectangle((x, 0.05), 3.2, 0.95, facecolor=colour,
                                   edgecolor="#333", linewidth=1.2))
        ax.text(x + 1.6, 0.52, label, ha="center", va="center", fontsize=7.6)
    ax.annotate("", xy=(4.6, 1.05), xytext=(11.5, 1.4),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2,
                                connectionstyle="arc3,rad=0.2"))

    ax.text(6.5, 5.0, "Figure 1 — CANXplain Phase 1 pipeline",
            ha="center", fontsize=11, weight="bold")
    ax.text(6.5, 4.78,
            "Orange = leakage controls   Green = vendor-agnostic representation   "
            "Pink = explanation-aware selection",
            ha="center", fontsize=7.5, color="#555")
    return _save(fig, out_dir, "figure1_pipeline")


# ------------------------------------------------ Figure 2: search process --
def figure2_search_process(rounds: pd.DataFrame, out_dir: str) -> Optional[str]:
    if rounds.empty or "round" not in rounds.columns:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    agg = rounds.groupby("round")[["best_fitness", "mean_fitness",
                                   "worst_fitness", "running_best_fitness"]].agg(
        ["mean", "std"])
    x = agg.index.to_numpy()

    ax = axes[0]
    for i, (col, label) in enumerate([
            ("running_best_fitness", "Running best"),
            ("best_fitness", "Round best"),
            ("mean_fitness", "Population mean"),
            ("worst_fitness", "Round worst")]):
        if (col, "mean") not in agg.columns:
            continue
        mean = agg[(col, "mean")].to_numpy()
        std = agg[(col, "std")].to_numpy()
        ax.plot(x, mean, marker="o", color=PALETTE[i], label=label, lw=1.8)
        if np.isfinite(std).any():
            ax.fill_between(x, mean - std, mean + std, color=PALETTE[i], alpha=0.15)
    ax.set_xlabel("Competition round")
    ax.set_ylabel("Fitness")
    ax.set_title("Search progress (shaded = ±1 SD across runs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(x)

    ax = axes[1]
    if "best_family" in rounds.columns:
        counts = (rounds.groupby(["round", "best_family"]).size()
                  .unstack(fill_value=0))
        bottom = np.zeros(len(counts))
        for i, family in enumerate(counts.columns):
            ax.bar(counts.index, counts[family], bottom=bottom,
                   color=PALETTE[i], label=str(family).upper())
            bottom += counts[family].to_numpy()
        ax.set_xlabel("Competition round")
        ax.set_ylabel("Times this family held the round best")
        ax.set_title("Which family wins each round")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        ax.set_xticks(counts.index)

    fig.suptitle("Figure 2 — ACHILLES-inspired meta-learning search",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "figure2_metalearning_search")


# --------------------------------------------- Figure 3: SHAP importance ---
def figure3_shap_importance(importance: pd.DataFrame, out_dir: str,
                            title_suffix: str = "") -> Optional[str]:
    if importance is None or importance.empty:
        return None
    top = importance.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(top))))
    ax.barh(top["feature"], top["mean_abs_shap"], color=PALETTE[0])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Figure 3 — Global SHAP feature importance{title_suffix}",
                 fontsize=11, weight="bold")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return _save(fig, out_dir, "figure3_shap_importance")


# ------------------------------------ Figure 4: consistency across folds ---
def figure4_shap_consistency(consistency, out_dir: str) -> Optional[str]:
    if consistency is None or consistency.rank_matrix is None:
        return None
    ranks = consistency.rank_matrix
    n_folds, n_features = ranks.shape
    if n_folds < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    order = np.argsort(ranks.mean(axis=0))[:12]
    names = [consistency.feature_names[i] for i in order]
    for fold in range(n_folds):
        ax.plot(range(len(order)), ranks[fold, order], marker="o",
                color=PALETTE[fold % len(PALETTE)], alpha=0.85,
                label=f"Fold {fold + 1}", lw=1.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("SHAP importance rank (1 = most important)")
    ax.invert_yaxis()
    ax.set_title("Per-fold feature ranking\n(flat, overlapping lines = stable)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    im = ax.imshow(consistency.pairwise, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(n_folds))
    ax.set_yticks(range(n_folds))
    ax.set_xticklabels([f"F{i+1}" for i in range(n_folds)])
    ax.set_yticklabels([f"F{i+1}" for i in range(n_folds)])
    for i in range(n_folds):
        for j in range(n_folds):
            ax.text(j, i, f"{consistency.pairwise[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)
    ax.set_title(f"Pairwise {consistency.metric} rank correlation\n"
                 f"consistency score = {consistency.score:.4f}", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Figure 4 — SHAP explanation stability across validation folds",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "figure4_shap_consistency_folds")


# ------------------------------------------ Figure 5: ablation comparison --
def figure5_ablation(runs: pd.DataFrame, out_dir: str,
                     arms: Optional[Sequence[str]] = None) -> Optional[str]:
    """The mandatory ablation figure: F1, cross-dataset accuracy, consistency."""
    from ..experiments.ablations import ARMS, FIGURE_ARMS

    arms = list(arms or FIGURE_ARMS)
    same = runs[(runs["evaluation_type"] == "same_dataset") & runs["arm"].isin(arms)]
    cross = runs[(runs["evaluation_type"] == "cross_dataset") & runs["arm"].isin(arms)]
    if same.empty:
        return None

    panels = [
        ("F1 (same-dataset)", _agg(same, ["arm"], "f1")),
        ("Accuracy (cross-dataset)", _agg(cross, ["arm"], "accuracy")),
        ("SHAP consistency", _agg(same, ["arm"], "shap_consistency")),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    labels = {code: f"{code}\n{ARMS[code].name}" for code in arms if code in ARMS}

    for ax, (title, data) in zip(axes, panels):
        if data.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            continue

        data = data.set_index("arm").reindex([a for a in arms if a in data["arm"].values])
        data = data.dropna(subset=["mean"])
        positions = np.arange(len(data))
        errors = data["std"].to_numpy()
        errors = np.where(np.isfinite(errors), errors, 0.0)
        n_seeds = int(data["count"].max()) if "count" in data else 1

        ax.bar(positions, data["mean"].to_numpy(),
               yerr=errors if n_seeds > 1 else None, capsize=4,
               color=[PALETTE[0] if idx == "A0" else PALETTE[5]
                      for idx in data.index],
               edgecolor="#333", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels([labels.get(i, i) for i in data.index],
                           rotation=30, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.0)                      # zero baseline, always
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.set_xlabel(f"n_seeds = {n_seeds}" +
                      ("  (error bars = ±1 SD)" if n_seeds > 1 else
                       "  (single seed: no error bars)"), fontsize=7.5)

        if title.startswith("SHAP"):
            ax.axhline(0.5, color="#C44E52", linestyle=":", lw=1.2)
            ax.text(0.02, 0.52, "0.5 = fold rankings unrelated",
                    transform=ax.get_yaxis_transform(), fontsize=7, color="#C44E52")

    fig.suptitle("Figure 5 — Ablation comparison (bars start at zero; "
                 "differences within error bars are not results)",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "figure5_ablation_comparison")


# ---------------------------------------- Figure 6: cross-dataset heatmap --
def figure6_cross_dataset(runs: pd.DataFrame, out_dir: str,
                          metric: str = "f1", arm: str = "A0") -> Optional[str]:
    subset = runs[(runs["arm"] == arm) &
                  runs["evaluation_type"].isin(["same_dataset", "cross_dataset"])]
    if subset.empty or metric not in subset.columns:
        return None

    pivot = subset.pivot_table(index="dataset_train", columns="dataset_test",
                               values=metric, aggfunc="mean")
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(1.7 * len(pivot.columns) + 3,
                                    1.2 * len(pivot.index) + 2.6))
    masked = np.ma.masked_invalid(pivot.to_numpy())
    im = ax.imshow(masked, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Tested on")
    ax.set_ylabel("Trained on")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                diagonal = pivot.index[i] == pivot.columns[j]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                        fontsize=9, weight="bold" if diagonal else "normal",
                        color="white" if value > 0.6 else "black")

    ax.set_title(f"Figure 6 — Cross-dataset {metric.upper()} ({arm}).\n"
                 "Diagonal = same-dataset; off-diagonal = generalisation.",
                 fontsize=10.5, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, label=metric.upper())
    fig.tight_layout()
    return _save(fig, out_dir, "figure6_cross_dataset_performance")


# ---------------- Figure 7: accuracy vs consistency vs complexity ----------
def figure7_tradeoff(candidates: pd.DataFrame, out_dir: str) -> Optional[str]:
    """Every candidate the search evaluated, in objective space."""
    if candidates.empty:
        return None
    needed = {"cv_accuracy", "shap_consistency", "complexity"}
    if not needed.issubset(candidates.columns):
        return None

    data = candidates.dropna(subset=["cv_accuracy", "shap_consistency", "complexity"])
    if data.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sizes = 25 + 180 * (np.log10(data["complexity"].clip(lower=1)) /
                        np.log10(data["complexity"].clip(lower=1)).max())
    for i, family in enumerate(sorted(data["family"].dropna().unique())):
        mask = data["family"] == family
        ax.scatter(data.loc[mask, "cv_accuracy"],
                   data.loc[mask, "shap_consistency"],
                   s=sizes[mask], alpha=0.6, color=PALETTE[i],
                   edgecolors="#333", linewidths=0.5, label=str(family).upper())
    ax.set_xlabel("CV accuracy (training split only)")
    ax.set_ylabel("SHAP consistency")
    ax.set_title("Objective space\n(marker size ∝ log complexity)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color="#C44E52", linestyle=":", lw=1)

    ax = axes[1]
    for i, family in enumerate(sorted(data["family"].dropna().unique())):
        mask = data["family"] == family
        ax.scatter(data.loc[mask, "complexity"], data.loc[mask, "cv_accuracy"],
                   alpha=0.6, color=PALETTE[i], edgecolors="#333",
                   linewidths=0.5, label=str(family).upper())
    ax.set_xscale("log")
    ax.set_xlabel("Model complexity (learned parameters / tree nodes, log scale)")
    ax.set_ylabel("CV accuracy")
    ax.set_title("Accuracy vs complexity", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Figure 7 — Accuracy vs SHAP consistency vs complexity "
                 "across all evaluated candidates", fontsize=11, weight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "figure7_accuracy_consistency_complexity")


def figure_sensitivity(sensitivity: pd.DataFrame, out_dir: str) -> Optional[str]:
    """Supplementary: how the selected model shifts with alpha/beta/gamma."""
    if sensitivity is None or sensitivity.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = sensitivity["weights"].astype(str)
    positions = np.arange(len(sensitivity))

    for ax, column, title in zip(
            axes,
            ["selected_accuracy", "selected_shap_consistency", "selected_complexity"],
            ["CV accuracy of selection", "SHAP consistency of selection",
             "Complexity of selection"]):
        if column not in sensitivity.columns:
            ax.axis("off")
            continue
        ax.bar(positions, sensitivity[column], color=PALETTE[3],
               edgecolor="#333", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.3, axis="y")
        if column == "selected_complexity":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1)

    fig.suptitle("Fitness weight sensitivity (alpha, beta, gamma)",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "figure8_fitness_sensitivity")


def generate_all_figures(runs: pd.DataFrame, rounds: pd.DataFrame,
                         candidates: pd.DataFrame, importance: pd.DataFrame,
                         consistency, out_dir: str) -> Dict[str, Optional[str]]:
    ensure_dir(out_dir)
    return {
        "figure1": figure1_pipeline(out_dir),
        "figure2": figure2_search_process(rounds, out_dir),
        "figure3": figure3_shap_importance(importance, out_dir,
                                           " (full CANXplain, A0)"),
        "figure4": figure4_shap_consistency(consistency, out_dir),
        "figure5": figure5_ablation(runs, out_dir),
        "figure6": figure6_cross_dataset(runs, out_dir),
        "figure7": figure7_tradeoff(candidates, out_dir),
    }
