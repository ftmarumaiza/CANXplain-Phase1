"""Experiment tracking.

Every evaluated configuration appends one row carrying its full
provenance: dataset, feature configuration, model family,
hyperparameters, seed, the exact split boundaries, every metric, timing,
complexity and fitness. Results land in both runs.csv (for reading) and
runs.jsonl (for anything nested that a CSV cell would mangle).

The point is that a table in the eventual paper should be derivable from
runs.csv by a groupby, without re-running anything.
"""
from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils import ensure_dir, get_logger, safe_json

LOG = get_logger()

# Columns pinned to the front of runs.csv so the file is readable raw.
LEADING_COLUMNS = [
    "run_id", "timestamp", "experiment", "arm", "arm_description",
    "dataset_train", "dataset_test", "evaluation_type", "seed",
    "feature_groups", "n_features", "model_family", "config_id",
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
    "mcc", "shap_consistency", "complexity", "fitness",
    "training_time_s", "inference_time_per_window_ms",
]


class ExperimentTracker:
    """Append-only run log."""

    def __init__(self, output_dir: str, experiment: str = "phase1"):
        self.output_dir = ensure_dir(output_dir)
        self.experiment = experiment
        self.csv_path = os.path.join(self.output_dir, "runs.csv")
        self.jsonl_path = os.path.join(self.output_dir, "runs.jsonl")
        self.records: List[Dict[str, Any]] = []
        self._counter = 0
        # Distinguishes this process from earlier ones writing to the same
        # results directory, so re-running an experiment cannot mint a
        # run_id that already exists in runs.jsonl.
        self.session = datetime.now(timezone.utc).strftime("%m%d%H%M%S")
        self.environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
        }

    def log(self, record: Dict[str, Any], **extra) -> Dict[str, Any]:
        self._counter += 1
        full = {
            "run_id": f"{self.experiment}-{self.session}-{self._counter:05d}",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experiment": self.experiment,
            **record,
            **extra,
        }
        full = safe_json(full)
        self.records.append(full)

        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(full, default=str) + "\n")
        return full

    def history(self) -> List[Dict[str, Any]]:
        """Every record ever written to this results directory.

        runs.jsonl is the durable log: each process appends to it. This
        reads it back so runs.csv can be rebuilt as the full history
        rather than just the records of the process writing it.
        """
        if not os.path.exists(self.jsonl_path):
            return list(self.records)
        rows: List[Dict[str, Any]] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    LOG.warning("skipping malformed line in %s", self.jsonl_path)
        return rows

    def to_frame(self, records: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """Session records by default; pass history() for the full log."""
        records = self.records if records is None else records
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        leading = [c for c in LEADING_COLUMNS if c in df.columns]
        rest = sorted(c for c in df.columns if c not in leading)
        return df[leading + rest]

    def flush(self) -> str:
        """Rebuild runs.csv from the complete jsonl log.

        Earlier versions wrote only this process's records, so running
        run_crossdataset after run_ablation silently truncated runs.csv
        to the last stage. The jsonl was always complete; the csv now
        mirrors it, and duplicate run_ids are collapsed keeping the last.
        """
        df = self.to_frame(self.history())
        if df.empty:
            LOG.warning("tracker has no records to write")
            return self.csv_path
        if "run_id" in df.columns:
            df = df.drop_duplicates(subset=["run_id"], keep="last")
        df.to_csv(self.csv_path, index=False)

        env_path = os.path.join(self.output_dir, "environment.json")
        with open(env_path, "w", encoding="utf-8") as fh:
            json.dump(self.environment, fh, indent=2)

        LOG.info("wrote %d run records (%d from this process) to %s",
                 len(df), len(self.records), self.csv_path)
        return self.csv_path

    def save_artifact(self, name: str, payload: Any) -> str:
        """Dump a nested structure (search trace, Q-table, SHAP matrix)."""
        path = os.path.join(ensure_dir(os.path.join(self.output_dir, "artifacts")),
                            f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(safe_json(payload), fh, indent=2, default=str)
        return path

    def save_frame(self, name: str, df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return None
        path = os.path.join(ensure_dir(os.path.join(self.output_dir, "artifacts")),
                            f"{name}.csv")
        df.to_csv(path, index=False)
        return path
