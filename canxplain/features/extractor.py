"""Vendor-agnostic feature extraction.

No payload byte ever reaches this module. Features come only from
arrival timing, CAN-ID identity statistics, and DLC (a protocol-level
length field, not content). That is what makes the representation
portable across vehicles and OEMs -- and it is also why absolute
performance here will sit below payload-aware IDS papers.

FEATURE GROUPS (toggled independently, which is what makes A6-A8 possible)

  basic        Minimal conventional CAN window descriptors. The A6
               "reduced feature set" baseline.
  temporal     Inter-frame arrival statistics over the window.
  temporal_id  Per-CAN-ID recurrence timing (the dt_ID family).
  frequency    CAN-ID count/diversity/entropy statistics.
  dlc          Data length code statistics. Auto-disabled when the
               dataset does not carry DLC.

IMPLEMENTATION NOTE
-------------------
Everything is vectorised over windows in chunks. With step=1 the window
count equals the message count, so a per-window Python loop would be
unusable on an i3. The per-window ID statistics use a pairwise-equality
trick: for a window of n messages, eq[i,j] = (id_i == id_j), so
count_i = sum_j eq[i,j] is each message's ID multiplicity. Then

    n_unique   = sum_i 1/count_i
    n_repeated = sum_i (count_i > 1)/count_i
    entropy    = -sum_i (p_i log p_i)/count_i ,  p_i = count_i/n

which gives exact per-distinct-ID quantities without any grouping loop.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger

LOG = get_logger()

EPS = 1e-12

FEATURE_GROUPS: Dict[str, List[str]] = {
    "basic": [
        "msg_count",
        "window_duration",
        "n_unique_ids_basic",
    ],
    "temporal": [
        "iat_mean",
        "iat_std",
        "iat_min",
        "iat_max",
        "iat_median",
        "iat_cv",
        "iat_range",
        "msg_rate_hz",
    ],
    "temporal_id": [
        "dt_id_mean",
        "dt_id_std",
        "dt_id_min",
        "dt_id_max",
        "dt_id_cv",
    ],
    "frequency": [
        "n_unique_ids",
        "id_diversity",
        "cmafid",
        "cmifid",
        "id_entropy",
        "n_repeated_ids",
        "max_id_share",
    ],
    "dlc": [
        "dlc_mean",
        "dlc_std",
        "dlc_max",
    ],
}

FEATURE_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    # name -> (group, description)  -- source of Table 2
    "msg_count":          ("basic", "Number of CAN messages in the window (constant at step=1; kept for the minimal baseline)"),
    "window_duration":    ("basic", "Wall-clock span of the window, t_last - t_first, in seconds"),
    "n_unique_ids_basic": ("basic", "Count of distinct CAN IDs in the window (minimal-baseline copy)"),

    "iat_mean":    ("temporal", "Mean inter-arrival time between consecutive frames (the mean inter-frame gap, wmdt)"),
    "iat_std":     ("temporal", "Standard deviation of inter-arrival times; raw timing variation"),
    "iat_min":     ("temporal", "Minimum inter-arrival time; collapses under flooding/DoS injection"),
    "iat_max":     ("temporal", "Maximum inter-arrival time; grows when legitimate traffic is starved"),
    "iat_median":  ("temporal", "Median inter-arrival time; outlier-robust centre of the timing distribution"),
    "iat_cv":      ("temporal", "Coefficient of variation of inter-arrival times (std/mean); scale-free burstiness"),
    "iat_range":   ("temporal", "Max minus min inter-arrival time"),
    "msg_rate_hz": ("temporal", "Messages per second over the window, msg_count / window_duration"),

    "dt_id_mean": ("temporal_id", "Mean time since the previous occurrence of the same CAN ID (dt_ID family)"),
    "dt_id_std":  ("temporal_id", "Standard deviation of same-ID recurrence intervals; periodic IDs have low values"),
    "dt_id_min":  ("temporal_id", "Minimum same-ID recurrence interval; drops sharply under ID spoofing"),
    "dt_id_max":  ("temporal_id", "Maximum same-ID recurrence interval"),
    "dt_id_cv":   ("temporal_id", "Coefficient of variation of same-ID recurrence intervals"),

    "n_unique_ids":   ("frequency", "Number of distinct CAN IDs in the window"),
    "id_diversity":   ("frequency", "Distinct IDs divided by message count; 1.0 means no ID repeats"),
    "cmafid":         ("frequency", "Occurrence count of the most frequent CAN ID in the window"),
    "cmifid":         ("frequency", "Occurrence count of the least frequent CAN ID in the window"),
    "id_entropy":     ("frequency", "Shannon entropy (nats) of the CAN-ID distribution; falls when one ID dominates"),
    "n_repeated_ids": ("frequency", "Number of distinct IDs appearing more than once in the window"),
    "max_id_share":   ("frequency", "Share of the window held by the most frequent ID, cmafid / msg_count"),

    "dlc_mean": ("dlc", "Mean data length code across the window (protocol length field, not payload content)"),
    "dlc_std":  ("dlc", "Standard deviation of the data length code"),
    "dlc_max":  ("dlc", "Maximum data length code in the window"),
}

DEFAULT_GROUPS = ["temporal", "temporal_id", "frequency", "dlc"]


def resolve_feature_names(groups: Sequence[str]) -> List[str]:
    """Ordered feature list for a set of groups."""
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown feature group(s) {unknown}; available: {sorted(FEATURE_GROUPS)}"
        )
    names: List[str] = []
    for group in FEATURE_GROUPS:          # fixed canonical order
        if group in groups:
            names.extend(FEATURE_GROUPS[group])
    return names


def feature_table(groups: Sequence[str] | None = None) -> pd.DataFrame:
    """Table 2: feature description table."""
    names = resolve_feature_names(groups) if groups else list(FEATURE_DESCRIPTIONS)
    rows = []
    for name in names:
        group, desc = FEATURE_DESCRIPTIONS[name]
        rows.append({"feature": name, "group": group, "description": desc})
    return pd.DataFrame(rows)


def _per_message_dt_id(timestamp: np.ndarray, can_id: np.ndarray,
                       capture: np.ndarray | None = None) -> np.ndarray:
    """Time since the previous frame carrying the same CAN ID.

    Computed per capture, then aggregated per window. Recurrence is
    scoped to the capture because a "gap" spanning two independent
    recordings is an artefact of concatenation, not a bus observation.

    The first occurrence of each ID in a capture has no predecessor and
    is marked NaN, so it is excluded from window statistics rather than
    contributing a spurious zero.
    """
    if capture is not None and len(np.unique(capture)) > 1:
        out = np.full(len(timestamp), np.nan, dtype=np.float64)
        for capture_id in np.unique(capture):
            mask = capture == capture_id
            out[mask] = _per_message_dt_id(timestamp[mask], can_id[mask])
        return out

    order = np.argsort(can_id, kind="stable")
    ids_sorted = can_id[order]
    ts_sorted = timestamp[order]

    dt = np.empty(len(timestamp), dtype=np.float64)
    dt[:] = np.nan
    same_as_prev = np.empty(len(ids_sorted), dtype=bool)
    same_as_prev[0] = False
    same_as_prev[1:] = ids_sorted[1:] == ids_sorted[:-1]

    diffs = np.empty(len(ts_sorted), dtype=np.float64)
    diffs[:] = np.nan
    diffs[1:] = ts_sorted[1:] - ts_sorted[:-1]
    diffs[~same_as_prev] = np.nan

    dt[order] = diffs
    return dt


def _nan_stats(block: np.ndarray) -> Dict[str, np.ndarray]:
    """mean/std/min/max over axis 1, tolerating all-NaN rows."""
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(block).sum(axis=1)
        mean = np.where(valid > 0, np.nanmean(np.where(np.isfinite(block), block, np.nan), axis=1), 0.0)
        std = np.where(valid > 1, np.nanstd(np.where(np.isfinite(block), block, np.nan), axis=1), 0.0)
        mn = np.where(valid > 0, np.nanmin(np.where(np.isfinite(block), block, np.inf), axis=1), 0.0)
        mx = np.where(valid > 0, np.nanmax(np.where(np.isfinite(block), block, -np.inf), axis=1), 0.0)
    mn = np.where(np.isfinite(mn), mn, 0.0)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return {"mean": mean, "std": std, "min": mn, "max": mx}


def extract_features(
    df: pd.DataFrame,
    starts: np.ndarray,
    window_size: int,
    groups: Sequence[str] = DEFAULT_GROUPS,
    chunk_size: int = 50_000,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build the feature matrix for the windows beginning at `starts`.

    Returns (X, y). A window is labelled attack (1) if ANY message inside
    it is an attack message -- the standard convention for window-level
    CAN IDS, and the one ACHILLES uses.
    """
    groups = list(groups)
    if "dlc" in groups and not (df["dlc"] >= 0).any():
        LOG.warning("dataset has no DLC; dropping the 'dlc' feature group")
        groups = [g for g in groups if g != "dlc"]

    names = resolve_feature_names(groups)
    if not names:
        raise ValueError("no feature groups selected")

    timestamp = df["timestamp"].to_numpy(np.float64)
    can_id = df["can_id"].to_numpy(np.int64)
    dlc = df["dlc"].to_numpy(np.float64)
    label = df["label"].to_numpy(np.int8)
    capture = (df["capture"].to_numpy(np.int64) if "capture" in df.columns
               else np.zeros(len(df), dtype=np.int64))

    need_dt_id = "temporal_id" in groups
    dt_id_all = (_per_message_dt_id(timestamp, can_id, capture)
                 if need_dt_id else None)

    offsets = np.arange(window_size, dtype=np.int64)
    n_windows = len(starts)
    columns: Dict[str, np.ndarray] = {n: np.empty(n_windows, np.float64) for n in names}
    y = np.empty(n_windows, dtype=np.int8)

    for lo in range(0, n_windows, chunk_size):
        hi = min(lo + chunk_size, n_windows)
        idx = starts[lo:hi, None] + offsets[None, :]        # (B, w)
        sl = slice(lo, hi)

        w_ts = timestamp[idx]
        w_id = can_id[idx]
        y[sl] = (label[idx].max(axis=1) > 0).astype(np.int8)

        duration = w_ts[:, -1] - w_ts[:, 0]

        # ---- inter-frame timing -------------------------------------
        if "temporal" in groups or "basic" in groups:
            iat = np.diff(w_ts, axis=1)
            iat = np.where(np.isfinite(iat), iat, 0.0)

        if "basic" in groups:
            columns["msg_count"][sl] = float(window_size)
            columns["window_duration"][sl] = duration

        if "temporal" in groups:
            mean = iat.mean(axis=1)
            std = iat.std(axis=1)
            columns["iat_mean"][sl] = mean
            columns["iat_std"][sl] = std
            columns["iat_min"][sl] = iat.min(axis=1)
            columns["iat_max"][sl] = iat.max(axis=1)
            columns["iat_median"][sl] = np.median(iat, axis=1)
            columns["iat_cv"][sl] = std / (np.abs(mean) + EPS)
            columns["iat_range"][sl] = iat.max(axis=1) - iat.min(axis=1)
            columns["msg_rate_hz"][sl] = window_size / (duration + EPS)

        # ---- per-ID recurrence timing --------------------------------
        if need_dt_id:
            block = dt_id_all[idx]
            stats = _nan_stats(block)
            columns["dt_id_mean"][sl] = stats["mean"]
            columns["dt_id_std"][sl] = stats["std"]
            columns["dt_id_min"][sl] = stats["min"]
            columns["dt_id_max"][sl] = stats["max"]
            columns["dt_id_cv"][sl] = stats["std"] / (np.abs(stats["mean"]) + EPS)

        # ---- CAN-ID frequency statistics ------------------------------
        if "frequency" in groups or "basic" in groups:
            eq = (w_id[:, :, None] == w_id[:, None, :])     # (B, w, w) bool
            counts = eq.sum(axis=2).astype(np.float64)      # multiplicity per message
            inv = 1.0 / counts
            n_unique = inv.sum(axis=1)

        if "basic" in groups:
            columns["n_unique_ids_basic"][sl] = n_unique

        if "frequency" in groups:
            p = counts / float(window_size)
            entropy = -(p * np.log(p + EPS) * inv).sum(axis=1)
            cmafid = counts.max(axis=1)
            cmifid = counts.min(axis=1)
            columns["n_unique_ids"][sl] = n_unique
            columns["id_diversity"][sl] = n_unique / float(window_size)
            columns["cmafid"][sl] = cmafid
            columns["cmifid"][sl] = cmifid
            columns["id_entropy"][sl] = entropy
            columns["n_repeated_ids"][sl] = ((counts > 1) * inv).sum(axis=1)
            columns["max_id_share"][sl] = cmafid / float(window_size)

        # ---- DLC -----------------------------------------------------
        if "dlc" in groups:
            w_dlc = dlc[idx]
            w_dlc = np.where(w_dlc >= 0, w_dlc, np.nan)
            stats = _nan_stats(w_dlc)
            columns["dlc_mean"][sl] = stats["mean"]
            columns["dlc_std"][sl] = stats["std"]
            columns["dlc_max"][sl] = stats["max"]

    X = pd.DataFrame({name: columns[name] for name in names})
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y


def build_split_features(
    df: pd.DataFrame,
    split_blocks: Dict[str, Sequence],
    window_size: int,
    step: int,
    groups: Sequence[str],
    max_windows: Dict[str, int] | int | None = None,
) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    """Feature matrices for each split, pooled across that split's captures."""
    from ..data.windowing import split_window_starts

    out = {}
    for split_name, blocks in split_blocks.items():
        cap = (max_windows.get(split_name) if isinstance(max_windows, dict)
               else max_windows)
        starts = split_window_starts(blocks, window_size, step, max_windows=cap)
        if len(starts) == 0:
            raise ValueError(f"split '{split_name}' yielded zero windows")
        X, y = extract_features(df, starts, window_size, groups)
        LOG.info("split '%s': %d windows, %d features, attack rate %.4f",
                 split_name, len(X), X.shape[1], float(y.mean()))
        out[split_name] = (X, y)
    return out
