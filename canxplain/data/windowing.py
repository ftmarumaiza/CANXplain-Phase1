"""Sliding-window segmentation and the split strategy.

WHY THIS FILE EXISTS
--------------------
With window_size=10 and step=1, two adjacent windows share 9 of their 10
messages. If those windows are split randomly into train and test, the
test set is a near-copy of the training set and every reported score is
inflated. This is the single easiest way to produce a meaningless
automotive-IDS result, and a lot of published work does exactly it.

THE SPLIT STRATEGY USED HERE (documented so it can be cited)
------------------------------------------------------------
Public CAN datasets arrive as several independent captures (one per
attack type). Two constraints have to hold at once:

  (a) no window may share a message with a window in another split, and
  (b) every split must contain both normal and attack traffic.

Splitting the concatenated stream globally satisfies (a) but breaks (b):
whole captures land on one side of the cut, so the training split can end
up entirely normal and the test split entirely attack. Splitting randomly
satisfies (b) but breaks (a).

So the split is applied WITHIN EACH CAPTURE:

    for each capture c:
        [ train_c | purge | val_c | purge | test_c ]
    train = concat(train_c), val = concat(val_c), test = concat(test_c)

  1. Messages are ordered by (capture, timestamp).
  2. Each capture is cut into three contiguous blocks by position, using
     the fractions from the config (default 0.6 / 0.2 / 0.2).
  3. A purge gap of (window_size - 1) messages is removed at each
     boundary -- exactly the overlap radius, so no window in one split
     can share a single message with a window in another split.
  4. Windows are generated inside each block only, so a window never
     straddles a split boundary OR a capture boundary. The latter matters
     because an inter-arrival time computed across two different
     recordings is not a physical quantity.

Consequence: within every capture, training data precedes test data in
time. This is stricter than a random split and scores will be lower than
a random split would give. That is the point -- it is the honest number.

Cross-validation inside the training split uses the same idea
(BlockedKFold below): contiguous validation blocks with purge gaps, never
shuffled folds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger

LOG = get_logger()

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitBlock:
    """A contiguous slice of one capture within the ordered stream."""
    name: str
    start: int          # inclusive index into the ordered message stream
    stop: int           # exclusive
    capture: int = 0
    n_messages: int = field(init=False)

    def __post_init__(self):
        self.n_messages = max(0, self.stop - self.start)

    def as_dict(self):
        return {"name": self.name, "capture": int(self.capture),
                "start": int(self.start), "stop": int(self.stop),
                "n_messages": int(self.n_messages)}


def capture_ranges(df: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """(capture_id, start, stop) for each capture in the ordered stream."""
    captures = df["capture"].to_numpy()
    if len(captures) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(captures)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(captures)]])
    return [(int(captures[s]), int(s), int(e)) for s, e in zip(starts, stops)]


def chronological_split(
    df: pd.DataFrame,
    window_size: int,
    train: float = 0.6,
    val: float = 0.2,
    test: float = 0.2,
) -> Dict[str, List[SplitBlock]]:
    """Per-capture chronological split with purge gaps.

    Returns {"train": [blocks], "val": [blocks], "test": [blocks]}.
    """
    total = train + val + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    purge = window_size - 1
    blocks: Dict[str, List[SplitBlock]] = {name: [] for name in SPLIT_NAMES}
    skipped = []

    for capture_id, lo, hi in capture_ranges(df):
        n = hi - lo
        usable = n - 2 * purge
        if usable < 3 * window_size:
            skipped.append((capture_id, n))
            continue

        n_train = int(usable * train)
        n_val = int(usable * val)

        train_stop = lo + n_train
        val_start = train_stop + purge
        val_stop = val_start + n_val
        test_start = val_stop + purge

        blocks["train"].append(SplitBlock("train", lo, train_stop, capture_id))
        blocks["val"].append(SplitBlock("val", val_start, val_stop, capture_id))
        blocks["test"].append(SplitBlock("test", test_start, hi, capture_id))

    if skipped:
        LOG.warning("%d capture(s) too short to split and were dropped: %s",
                    len(skipped), skipped)
    if not blocks["train"]:
        raise ValueError(
            f"no capture in this stream is long enough to split with "
            f"window_size={window_size}"
        )

    for name in SPLIT_NAMES:
        LOG.info("split '%s': %d capture blocks, %d messages (purge=%d)",
                 name, len(blocks[name]),
                 sum(b.n_messages for b in blocks[name]), purge)
    return blocks


def window_starts(
    block: SplitBlock,
    window_size: int,
    step: int = 1,
    max_windows: int | None = None,
) -> np.ndarray:
    """Start indices of every window fully contained in `block`.

    If there are more windows than `max_windows`, an EVENLY SPACED subset
    is taken rather than a random one. Even spacing keeps coverage of the
    whole block and reduces overlap between retained windows; random
    sampling would keep many near-duplicate neighbours.
    """
    last_start = block.stop - window_size
    if last_start < block.start:
        return np.empty(0, dtype=np.int64)

    starts = np.arange(block.start, last_start + 1, step, dtype=np.int64)
    if max_windows is not None and len(starts) > max_windows:
        keep = np.unique(np.linspace(0, len(starts) - 1, max_windows).astype(np.int64))
        starts = starts[keep]
    return starts


def split_window_starts(
    blocks: List[SplitBlock],
    window_size: int,
    step: int = 1,
    max_windows: int | None = None,
) -> np.ndarray:
    """Window starts for a whole split, budget shared across its captures."""
    if not blocks:
        return np.empty(0, dtype=np.int64)

    per_block = None
    if max_windows is not None:
        per_block = max(1, max_windows // len(blocks))

    parts = [window_starts(b, window_size, step, per_block) for b in blocks]
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(parts))


class BlockedKFold:
    """Contiguous, purged K-fold for overlapping temporal windows.

    Fold k uses contiguous block k as validation. Training indices are
    everything outside that block MINUS a purge margin of
    (window_size - 1) windows on each side, so no training window shares
    a message with a validation window.

    This replaces StratifiedKFold everywhere in the meta-learning loop.
    Using shuffled folds there would leak information into the SHAP
    consistency score, which is precisely the quantity this project
    claims as a contribution.
    """

    def __init__(self, n_splits: int = 3, purge: int = 9):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.purge = max(0, purge)

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        all_idx = np.arange(n)

        for k in range(self.n_splits):
            v_start, v_stop = bounds[k], bounds[k + 1]
            val_idx = all_idx[v_start:v_stop]

            mask = np.ones(n, dtype=bool)
            mask[max(0, v_start - self.purge): min(n, v_stop + self.purge)] = False
            train_idx = all_idx[mask]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def describe_stream(df: pd.DataFrame, name: str) -> dict:
    """Row for Table 1 (dataset statistics)."""
    durations, medians = [], []
    timestamps = df["timestamp"].to_numpy()
    for _, lo, hi in capture_ranges(df):
        ts = timestamps[lo:hi]
        if len(ts) > 1:
            durations.append(float(ts[-1] - ts[0]))
            gaps = np.diff(ts)
            gaps = gaps[np.isfinite(gaps)]
            if gaps.size:
                medians.append(float(np.median(gaps)))

    total_duration = float(np.sum(durations)) if durations else np.nan
    return {
        "dataset": name,
        "n_messages": int(len(df)),
        "n_captures": int(df["capture"].nunique()),
        "n_unique_can_ids": int(df["can_id"].nunique()),
        "attack_message_rate": float(df["label"].mean()),
        "total_duration_s": total_duration,
        "median_inter_arrival_s": float(np.median(medians)) if medians else np.nan,
        "mean_msg_rate_hz": (float(len(df) / total_duration)
                             if total_duration and total_duration > 0 else np.nan),
        "has_dlc": bool((df["dlc"] >= 0).any()),
    }
