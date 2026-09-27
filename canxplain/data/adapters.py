"""Dataset adapters.

Each public CAN dataset ships in its own layout. An adapter's only job is
to turn whatever it finds on disk into the standard four-column frame
defined in schema.py. Adding ORNL/ROAD or SAD later means writing one
adapter class and registering it -- nothing downstream changes.

Registered adapters:
    car_hacking  Car-Hacking / Attack & Defense Challenge 2020
    can_ids      CAN intrusion detection dataset (OTIDS-style logs)
    ornl         ORNL ROAD dataset
    sad          Survival Analysis Dataset
    generic_csv  any CSV, with column names given in the config
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils import get_logger
from . import schema

LOG = get_logger()

_REGISTRY: Dict[str, type] = {}


def register(name: str):
    def wrap(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return wrap


def get_adapter(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown adapter '{name}'. registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def available_adapters() -> List[str]:
    return sorted(_REGISTRY)


class BaseAdapter:
    """Common file discovery + assembly logic."""

    name = "base"
    file_patterns = ("*.csv", "*.txt", "*.log", "*.parquet")

    def __init__(self, **options):
        self.options = options

    # --- to be overridden -------------------------------------------------
    def read_file(self, path: str) -> pd.DataFrame:
        raise NotImplementedError

    # --- shared -----------------------------------------------------------
    def discover(self, path: str) -> List[str]:
        if os.path.isfile(path):
            return [path]
        if not os.path.isdir(path):
            raise FileNotFoundError(f"dataset path does not exist: {path}")
        files: List[str] = []
        for pattern in self.file_patterns:
            files.extend(glob.glob(os.path.join(path, "**", pattern),
                                   recursive=True))
        if not files:
            raise FileNotFoundError(
                f"no data files under {path} matching {self.file_patterns}"
            )
        return sorted(files)

    def load(self, path: str, max_messages: Optional[int] = None) -> pd.DataFrame:
        """Read every capture file, standardise, and stack them by capture.

        Each file becomes one capture with its own index. Files are NOT
        merged into a single global time order -- see schema.py for why
        that would destroy the inter-arrival features.
        """
        frames = []
        capture_names = []
        for capture_idx, file_path in enumerate(self.discover(path)):
            try:
                frame = self.read_file(file_path)
            except Exception as exc:  # one malformed capture must not kill a run
                LOG.warning("skipping %s (%s)", os.path.basename(file_path), exc)
                continue
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            frame["capture"] = len(capture_names)
            capture_names.append(os.path.basename(file_path))
            frames.append(frame)

        if not frames:
            raise ValueError(f"adapter '{self.name}' read no usable rows from {path}")

        df = schema.finalise(pd.concat(frames, ignore_index=True))

        if max_messages is not None and len(df) > max_messages:
            df = self._cap_per_capture(df, max_messages)

        df.attrs["dataset_files"] = len(capture_names)
        df.attrs["capture_names"] = capture_names
        return df

    @staticmethod
    def _cap_per_capture(df: pd.DataFrame, max_messages: int) -> pd.DataFrame:
        """Shrink the stream by taking a contiguous block from EACH capture.

        Two properties matter here and a naive cap breaks both:

        1. Contiguous, not random. Random row sampling would destroy
           inter-arrival times, which are the whole basis of the feature
           set.
        2. Proportional across captures, not one central slice of the
           concatenation. Taking a single central block would land
           entirely inside whichever capture happens to sit in the
           middle -- typically producing an all-normal or all-attack
           stream and a single-class training split.
        """
        captures = df["capture"].to_numpy()
        sizes = df.groupby("capture").size()
        total = int(sizes.sum())
        keep_parts = []

        for capture_id, size in sizes.items():
            quota = max(1, int(round(max_messages * size / total)))
            quota = min(quota, int(size))
            block = df.loc[captures == capture_id]
            start = (len(block) - quota) // 2          # central, contiguous
            keep_parts.append(block.iloc[start:start + quota])

        out = pd.concat(keep_parts, ignore_index=True)
        LOG.info("capped stream to %d messages across %d captures (from %d)",
                 len(out), len(sizes), total)
        return out


@register("generic_csv")
class GenericCSVAdapter(BaseAdapter):
    """CSV with explicit or auto-detected column names.

    Options: timestamp_col, can_id_col, dlc_col, label_col, sep, header.
    """

    file_patterns = ("*.csv", "*.tsv", "*.parquet")

    def read_file(self, path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            raw = pd.read_parquet(path)
        else:
            raw = pd.read_csv(
                path,
                sep=self.options.get("sep", ","),
                low_memory=False,
            )
        return self._map_columns(raw)

    def _map_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        opts = self.options
        ts_col = opts.get("timestamp_col") or schema.find_column(
            raw.columns, schema.TIMESTAMP_ALIASES)
        id_col = opts.get("can_id_col") or schema.find_column(
            raw.columns, schema.CAN_ID_ALIASES)
        dlc_col = opts.get("dlc_col") or schema.find_column(
            raw.columns, schema.DLC_ALIASES)
        label_col = opts.get("label_col") or schema.find_column(
            raw.columns, schema.LABEL_ALIASES)

        if ts_col is None or id_col is None:
            raise ValueError(
                f"could not identify timestamp/CAN-ID columns in {list(raw.columns)[:10]}"
            )

        out = pd.DataFrame()
        out["timestamp"] = pd.to_numeric(raw[ts_col], errors="coerce")
        out["can_id"] = schema.parse_can_id(raw[id_col])
        out["dlc"] = (pd.to_numeric(raw[dlc_col], errors="coerce")
                      if dlc_col else -1)
        if label_col is None:
            raise ValueError("no label column found; supply label_col in config")
        out["label"] = schema.normalise_labels(raw[label_col])
        return out


@register("car_hacking")
class CarHackingAdapter(GenericCSVAdapter):
    """Car-Hacking dataset and the Attack & Defense Challenge 2020 variant.

    Two layouts are handled:

    1. Headerless per-attack CSVs (DoS_dataset.csv, Fuzzy_dataset.csv, ...):
       Timestamp, CAN_ID, DLC, DATA[0..DLC-1], Flag
       The number of data columns varies row to row, so the flag is the
       last non-empty field on the line, not a fixed column index.

    2. Challenge 2020 CSV with a header:
       Timestamp, Arbitration_ID, DLC, Data, Class, SubClass
    """

    file_patterns = ("*.csv", "*.txt")

    def read_file(self, path: str) -> pd.DataFrame:
        head = pd.read_csv(path, nrows=3, header=None, low_memory=False,
                           dtype=str, on_bad_lines="skip")
        first_row = [str(v).strip().lower() for v in head.iloc[0].tolist()]
        has_header = any(
            tok in first_row
            for tok in ("timestamp", "arbitration_id", "class", "can_id")
        )
        if has_header:
            return super().read_file(path)
        return self._read_headerless(path)

    @staticmethod
    def _max_field_count(path: str, scan_lines: int = 5000) -> int:
        """Widest row in the file.

        Car-Hacking rows carry DLC data bytes inline, so a DLC-8 frame
        has four more fields than a DLC-4 frame. pandas infers the column
        count from the first row and then DROPS every wider row. Since
        injected attack frames are usually DLC 8 while normal traffic is
        mixed, that silently deletes most of the attack class and yields
        a dataset with a 0.0 attack rate. Scanning for the true width
        first is the fix.
        """
        widest = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                if i >= scan_lines:
                    break
                widest = max(widest, line.count(",") + 1)
        return max(widest, 4)

    def _read_headerless(self, path: str) -> pd.DataFrame:
        n_columns = self._max_field_count(path)
        raw = pd.read_csv(path, header=None, dtype=str, low_memory=False,
                          names=list(range(n_columns)), on_bad_lines="skip")
        raw = raw.replace({None: np.nan})

        # The flag is the last non-null field of each row.
        values = raw.to_numpy(dtype=object)
        flags = np.empty(len(raw), dtype=object)
        for i, row in enumerate(values):
            flag = None
            for cell in row[::-1]:
                if isinstance(cell, str) and cell.strip():
                    flag = cell.strip()
                    break
            flags[i] = flag

        out = pd.DataFrame()
        out["timestamp"] = pd.to_numeric(raw[0], errors="coerce")
        out["can_id"] = schema.parse_can_id(raw[1])
        out["dlc"] = pd.to_numeric(raw[2], errors="coerce")
        out["label"] = schema.normalise_labels(pd.Series(flags))
        return out


@register("can_ids")
class CanIdsAdapter(BaseAdapter):
    """CAN intrusion detection dataset (OTIDS-style).

    Handles the plain-text log format:
        Timestamp: 1478198376.389427  ID: 0440  000  DLC: 8  05 21 68 ...
    and CSV exports of the same data.

    Text logs carry no per-line label; the attack type is encoded in the
    file name (e.g. DoS_dataset.txt, Attack_free_dataset.txt), so the
    label is taken from there.
    """

    file_patterns = ("*.txt", "*.log", "*.csv")

    LINE_RE = re.compile(
        r"Timestamp:\s*(?P<ts>[0-9.]+).*?"
        r"ID:\s*(?P<id>[0-9a-fA-F]+).*?"
        r"DLC:\s*(?P<dlc>\d+)",
        re.IGNORECASE,
    )

    def read_file(self, path: str) -> pd.DataFrame:
        if path.lower().endswith(".csv"):
            return GenericCSVAdapter(**self.options).read_file(path)
        return self._read_log(path)

    def _read_log(self, path: str) -> pd.DataFrame:
        name = os.path.basename(path).lower()
        is_normal = any(tok in name for tok in
                        ("attack_free", "normal", "benign", "free"))
        file_label = 0 if is_normal else 1

        records = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                match = self.LINE_RE.search(line)
                if match:
                    records.append((
                        float(match.group("ts")),
                        int(match.group("id"), 16),
                        int(match.group("dlc")),
                    ))
        if not records:
            raise ValueError("no parseable log lines")

        out = pd.DataFrame(records, columns=["timestamp", "can_id", "dlc"])
        out["label"] = np.int8(file_label)

        if file_label == 1:
            LOG.info(
                "%s: labelled attack from file name. Attack captures usually "
                "contain normal traffic too, so window labels from this file "
                "are coarse -- see README 'Label provenance'.",
                os.path.basename(path),
            )
        return out


@register("ornl")
class OrnlAdapter(GenericCSVAdapter):
    """ORNL ROAD dataset.

    ROAD ships per-capture CSVs with a header and a companion metadata
    JSON giving the injection interval. Where a per-row label column is
    absent, options may supply attack_start/attack_end in capture time.
    """

    file_patterns = ("*.csv",)

    def read_file(self, path: str) -> pd.DataFrame:
        raw = pd.read_csv(path, low_memory=False)
        label_col = self.options.get("label_col") or schema.find_column(
            raw.columns, schema.LABEL_ALIASES)
        if label_col is not None:
            return self._map_columns(raw)

        ts_col = schema.find_column(raw.columns, schema.TIMESTAMP_ALIASES)
        id_col = schema.find_column(raw.columns, schema.CAN_ID_ALIASES)
        dlc_col = schema.find_column(raw.columns, schema.DLC_ALIASES)
        if ts_col is None or id_col is None:
            raise ValueError("ORNL capture lacks timestamp/ID columns")

        out = pd.DataFrame()
        out["timestamp"] = pd.to_numeric(raw[ts_col], errors="coerce")
        out["can_id"] = schema.parse_can_id(raw[id_col])
        out["dlc"] = (pd.to_numeric(raw[dlc_col], errors="coerce")
                      if dlc_col else -1)

        start = self.options.get("attack_start")
        end = self.options.get("attack_end")
        if start is None or end is None:
            raise ValueError(
                f"{os.path.basename(path)} has no label column and no "
                "attack_start/attack_end in the config"
            )
        within = (out["timestamp"] >= float(start)) & (out["timestamp"] <= float(end))
        out["label"] = within.astype("int8")
        return out


@register("sad")
class SadAdapter(GenericCSVAdapter):
    """Survival Analysis Dataset (Hyundai Sonata / Kia Soul / Chevrolet Spark).

    Same shape as Car-Hacking per-attack CSVs; the flag column marks
    injected frames.
    """

    file_patterns = ("*.csv", "*.txt")

    def read_file(self, path: str) -> pd.DataFrame:
        try:
            return super().read_file(path)
        except Exception:
            return CarHackingAdapter(**self.options)._read_headerless(path)


def load_dataset(name: str, spec, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Load one dataset from its config spec, with a parquet cache.

    spec keys: adapter, path, max_messages, plus adapter-specific options.
    """
    from ..utils import ensure_dir, stable_hash

    spec = dict(spec)
    adapter_name = spec.pop("adapter")
    path = spec.pop("path")
    max_messages = spec.pop("max_messages", None)

    cache_path = None
    if cache_dir:
        key = stable_hash({"n": name, "a": adapter_name, "p": path,
                           "m": max_messages, "o": spec})
        cache_path = os.path.join(ensure_dir(cache_dir), f"{name}_{key}.parquet")
        if os.path.exists(cache_path):
            LOG.info("dataset '%s': loaded from cache", name)
            return pd.read_parquet(cache_path)

    adapter = get_adapter(adapter_name, **spec)
    df = adapter.load(path, max_messages=max_messages)
    LOG.info("dataset '%s': %d messages, %d unique IDs, attack rate %.4f",
             name, len(df), df["can_id"].nunique(), df["label"].mean())

    if cache_path:
        df.to_parquet(cache_path, index=False)
    return df
