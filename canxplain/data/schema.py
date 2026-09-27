"""The one internal representation every dataset is converted into.

Every adapter must return a pandas DataFrame with exactly these columns:

    timestamp : float64  seconds, monotonically non-decreasing WITHIN a capture
    can_id    : int64    arbitration ID as an integer (hex is parsed)
    dlc       : int64    data length code, or -1 when the dataset omits it
    label     : int8     0 = normal, 1 = attack
    capture   : int64    index of the source recording this message came from

The `capture` column is load-bearing, not bookkeeping. Public CAN
datasets ship as several independent recordings (one per attack type),
each with its own clock. Merging them into one global time-sorted stream
would interleave unrelated recordings and make every inter-arrival
feature meaningless. So the stream is ordered by (capture, timestamp),
windows never cross a capture boundary, and the train/val/test split is
applied within each capture.

Note what is deliberately absent: payload bytes. Phase 1 is
vendor-agnostic by construction, so payload content never enters the
pipeline, not even as an intermediate column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STANDARD_COLUMNS = ["timestamp", "can_id", "dlc", "label", "capture"]

# Candidate source column names seen across public CAN datasets.
TIMESTAMP_ALIASES = ["timestamp", "time", "ts", "t", "arrival_time", "abs_time"]
CAN_ID_ALIASES = [
    "can_id", "canid", "id", "arbitration_id", "arbitrationid",
    "message_id", "msg_id", "pid",
]
DLC_ALIASES = ["dlc", "data_length", "len", "length", "datalength"]
LABEL_ALIASES = [
    "label", "flag", "class", "attack", "is_attack", "target",
    "category", "subclass", "attack_type",
]

# Strings that mean "this frame is part of an attack".
ATTACK_TOKENS = {
    "t", "attack", "1", "true", "yes", "malicious", "injected", "anomaly",
    "dos", "fuzzy", "fuzzing", "spoofing", "replay", "impersonation",
    "gear", "rpm", "flooding", "malfunction", "masquerade", "fabrication",
}
NORMAL_TOKENS = {
    "r", "normal", "0", "false", "no", "benign", "attack_free",
    "n-attack", "nattack", "clean",
}


def find_column(columns, aliases):
    """Match a source column to a standard field, case/underscore-insensitive."""
    normalised = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c
                  for c in columns}
    for alias in aliases:
        if alias in normalised:
            return normalised[alias]
    # fall back to a substring match, longest alias first
    for alias in sorted(aliases, key=len, reverse=True):
        for norm, original in normalised.items():
            if alias in norm:
                return original
    return None


def parse_can_id(series: pd.Series) -> pd.Series:
    """CAN IDs appear as hex strings, ints, or '0x1f0'. Normalise to int."""
    if pd.api.types.is_integer_dtype(series):
        return series.astype("int64")
    if pd.api.types.is_float_dtype(series):
        return series.fillna(-1).astype("int64")

    text = series.astype(str).str.strip().str.lower()
    text = text.str.replace("^0x", "", regex=True)

    parsed = pd.to_numeric(text, errors="coerce")
    looks_hex = text.str.fullmatch(r"[0-9a-f]+")
    # If any value contains a-f it must be hex; parse the whole column as hex
    # so that "0440" and "04f0" land in the same numbering scheme.
    has_letters = text.str.contains(r"[a-f]", regex=True, na=False).any()
    if has_letters:
        parsed = text.where(looks_hex).map(
            lambda v: int(v, 16) if isinstance(v, str) else np.nan
        )
    return pd.to_numeric(parsed, errors="coerce").fillna(-1).astype("int64")


def normalise_labels(series: pd.Series) -> pd.Series:
    """Map heterogeneous label encodings onto {0, 1}.

    Anything that is not recognisably normal and not recognisably an
    attack is left as NaN so the caller can decide (we drop those rows
    rather than guess).
    """
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) != 0).astype("int8")

    text = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[text.isin(NORMAL_TOKENS)] = 0.0
    out[text.isin(ATTACK_TOKENS)] = 1.0

    unresolved = out.isna()
    if unresolved.any():
        # e.g. "DoS_attack", "Fuzzy Attack" -> substring match
        contains_attack = text[unresolved].apply(
            lambda v: any(tok in v for tok in ATTACK_TOKENS if len(tok) > 2)
        )
        out.loc[unresolved] = contains_attack.map({True: 1.0, False: 0.0})
    return out.fillna(0).astype("int8")


def finalise(df: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
    """Validate, clean and sort a standardised frame."""
    if "capture" not in df.columns:
        df = df.copy()
        df["capture"] = 0
    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"adapter produced frame missing columns: {missing}")

    df = df[STANDARD_COLUMNS].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["can_id"] = pd.to_numeric(df["can_id"], errors="coerce")
    df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(-1)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["timestamp", "can_id", "label"])
    df = df[df["can_id"] >= 0]
    dropped = before - len(df)

    df["can_id"] = df["can_id"].astype("int64")
    df["dlc"] = df["dlc"].astype("int64")
    df["label"] = df["label"].astype("int8")

    df["capture"] = pd.to_numeric(df["capture"], errors="coerce").fillna(0).astype("int64")

    if sort:
        # Order by capture first, then time within the capture. Stable so
        # identical timestamps keep their original arrival order.
        df = df.sort_values(["capture", "timestamp"], kind="mergesort")

    df = df.reset_index(drop=True)
    df.attrs["dropped_invalid_rows"] = int(dropped)
    return df
