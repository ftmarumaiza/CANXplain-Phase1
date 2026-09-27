"""Shared setup for every runner: CLI parsing, output layout, collection."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import Config, load_config, parse_cli_overrides, save_config_snapshot
from ..evaluation.tracking import ExperimentTracker
from ..utils import ensure_dir, get_logger

LOG = get_logger()


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", "-c", default="configs/debug.yaml",
                        help="path to the YAML configuration file")
    # action="extend" rather than the default "store": with plain
    # nargs="*", a second --set silently replaced the first, so
    # `--set run.seeds=[0,1,2] --set run.output_dir=...` applied only the
    # output directory and the run quietly stayed on one seed.
    parser.add_argument("--set", "-s", nargs="*", action="extend", default=None,
                        dest="overrides", metavar="KEY=VALUE",
                        help="override any config key, repeatable, "
                             "e.g. --set meta.rounds=5 --set run.seeds=[0,1,2]")
    parser.add_argument("--output", "-o", default=None,
                        help="override run.output_dir")
    parser.add_argument("--tag", default=None,
                        help="suffix appended to the output directory")
    return parser


def setup(args, experiment: str) -> Tuple[Config, ExperimentTracker, Dict[str, str]]:
    cfg = load_config(args.config, parse_cli_overrides(args.overrides or []))

    output_dir = args.output or cfg.run.output_dir
    if args.tag:
        output_dir = f"{output_dir}_{args.tag}"
    cfg.run.output_dir = output_dir

    paths = {
        "root": ensure_dir(output_dir),
        "tables": ensure_dir(os.path.join(output_dir, "tables")),
        "figures": ensure_dir(os.path.join(output_dir, "figures")),
        "models": ensure_dir(os.path.join(output_dir, "models")),
        "artifacts": ensure_dir(os.path.join(output_dir, "artifacts")),
    }

    save_config_snapshot(cfg, output_dir)
    tracker = ExperimentTracker(output_dir, experiment=experiment)

    LOG.info("mode=%s | output=%s | seeds=%s",
             cfg.run.get("mode", "?"), output_dir, list(cfg.run.seeds))
    return cfg, tracker, paths


def dataset_names(cfg) -> List[str]:
    return list(cfg.data.datasets.keys())


def source_datasets(cfg) -> List[str]:
    """Datasets used for training. Defaults to all configured datasets."""
    requested = cfg.run.get("source_datasets")
    return list(requested) if requested else dataset_names(cfg)


def cross_targets(cfg, source: str) -> List[str]:
    """Targets for cross-dataset evaluation from a given source."""
    if not cfg.run.get("cross_dataset", True):
        return []
    requested = cfg.run.get("cross_targets")
    targets = list(requested) if requested else dataset_names(cfg)
    return [t for t in targets if t != source]


def collect_traces(results) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten per-arm search traces into round-level and candidate-level frames."""
    round_rows, candidate_rows = [], []
    for result in results:
        context = {"arm": result.arm_code, "dataset": result.dataset,
                   "seed": result.seed}
        for row in result.search.trace.rounds:
            round_rows.append({**context, **row})
        for row in result.search.trace.candidates:
            candidate_rows.append({**context, **row})
    return pd.DataFrame(round_rows), pd.DataFrame(candidate_rows)


def pick_reference_result(results, arm: str = "A0"):
    """The A0 run used for the single-model figures (3 and 4)."""
    matching = [r for r in results if r.arm_code == arm]
    if not matching:
        return results[0] if results else None
    return matching[0]


def write_summary(paths: Dict[str, str], runs: pd.DataFrame,
                  extra_lines: Optional[List[str]] = None) -> str:
    """A short, deliberately unexcited plain-text summary of the run.

    The aggregate is built from the complete runs.csv in the results
    directory, not only from the records of the stage that happens to be
    writing the file. run_all finishes with the sensitivity sweep, which
    logs three records; summarising those alone used to overwrite the
    ablation summary with a near-empty one.
    """
    stage_runs = runs
    csv_path = os.path.join(paths["root"], "runs.csv")
    if os.path.exists(csv_path):
        try:
            full = pd.read_csv(csv_path)
            if not full.empty:
                runs = full
        except Exception as exc:  # pragma: no cover - summary must never fail
            LOG.warning("could not read %s for the summary: %s", csv_path, exc)

    lines = ["CANXplain Phase 1 — run summary", "=" * 34, ""]

    if runs.empty:
        lines.append("No runs were recorded.")
    else:
        same = runs[runs["evaluation_type"] == "same_dataset"]
        cross = runs[runs["evaluation_type"] == "cross_dataset"]
        lines.append(f"Total run records : {len(runs)} "
                     f"({len(stage_runs)} from this stage)")
        lines.append(f"Arms evaluated    : {', '.join(sorted(runs['arm'].dropna().unique()))}")
        lines.append(f"Seeds             : {sorted(runs['seed'].dropna().unique().tolist())}")
        lines.append("")

        if not same.empty:
            lines.append("Same-dataset (mean over seeds and datasets):")
            summary = same.groupby("arm")[["f1", "accuracy", "shap_consistency",
                                           "complexity"]].mean()
            lines.append(summary.to_string(float_format=lambda v: f"{v:.4f}"))
            lines.append("")

        if not cross.empty:
            lines.append("Cross-dataset (mean over seeds and pairs) "
                         "— the generalisation result:")
            summary = cross.groupby("arm")[["f1", "accuracy", "roc_auc"]].mean()
            lines.append(summary.to_string(float_format=lambda v: f"{v:.4f}"))
            lines.append("")

        n_seeds = runs["seed"].nunique()
        lines.append("Reading these numbers:")
        lines.append(f"  - {n_seeds} seed(s) were run. " + (
            "Differences smaller than the seed-to-seed spread in Table 6 are "
            "not results." if n_seeds > 1 else
            "With a single seed there is no variance estimate at all, so no "
            "comparison between arms is supported. Re-run with more seeds "
            "before drawing any conclusion."))
        lines.append("  - Cross-dataset rows are the ones that bear on the "
                     "generalisation claim; same-dataset rows do not.")
        lines.append("  - No claim that CANXplain improves over its ablations "
                     "is made by this file. Check Table 6 and "
                     "statistical_tests.csv and decide for yourself.")

    if extra_lines:
        lines += [""] + extra_lines

    path = os.path.join(paths["root"], "SUMMARY.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    LOG.info("wrote %s", path)
    print("\n" + "\n".join(lines) + "\n")
    return path
