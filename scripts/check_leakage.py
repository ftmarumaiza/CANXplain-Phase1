"""Assertions that the evaluation setup is fair.

These are the invariants the results depend on. They are cheap to check
and expensive to get wrong, so they are checked explicitly rather than
asserted in prose:

  1. Windows never cross a split boundary or a capture boundary.
     A purge gap of window_size - 1 messages sits at every split cut.
  2. The scaler is fitted on the training block only. Val, test and any
     cross-dataset target are transformed with that same frozen scaler.
  3. The cross-dataset target shares no feature-scaling statistics with
     the source: transforming the target must not change the scaler.
  4. Train/val/test window index sets are disjoint.
  5. Feature columns are identical and identically ordered between the
     source training matrix and the cross-dataset target matrix, so the
     model is not silently fed permuted inputs.

    python scripts/check_leakage.py --config configs/debug.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canxplain.config import load_config, parse_cli_overrides
from canxplain.experiments.pipeline import prepare_dataset, prepare_target_dataset
from canxplain.utils import get_logger

LOG = get_logger()

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        LOG.info("PASS  %s", label)
    else:
        LOG.error("FAIL  %s %s", label, detail)
        FAILURES.append(label)


def check_dataset(cfg, name: str, groups: list[str], window_size: int) -> None:
    LOG.info("-" * 70)
    LOG.info("dataset: %s", name)
    data = prepare_dataset(cfg, name, groups, fit_scaler=True)

    # 1. purge gap and capture containment
    gap_ok = True
    detail = ""
    ordered = ["train", "val", "test"]
    per_capture: dict[int, list] = {}
    for split in ordered:
        for block in data.blocks.get(split, []):
            per_capture.setdefault(block.capture, []).append((split, block))
    for capture, blocks in per_capture.items():
        blocks.sort(key=lambda sb: sb[1].start)
        for (_, a), (_, b) in zip(blocks, blocks[1:]):
            if b.start - a.stop < window_size - 1:
                gap_ok = False
                detail = (f"capture {capture}: gap {b.start - a.stop} "
                          f"< purge {window_size - 1}")
    check(gap_ok, "purge gap >= window_size - 1 at every split boundary", detail)

    # 4. disjoint source messages.
    # Each split's feature frame carries its own RangeIndex, so the row
    # labels say nothing about provenance. The meaningful check is that
    # the (capture, message) spans the splits are drawn from do not
    # overlap.
    spans: dict[str, set] = {}
    for split in ordered:
        seen = set()
        for block in data.blocks.get(split, []):
            seen.update((block.capture, i) for i in range(block.start, block.stop))
        spans[split] = seen
    overlap = (len(spans["train"] & spans["val"])
               + len(spans["train"] & spans["test"])
               + len(spans["val"] & spans["test"]))
    check(overlap == 0,
          "train / val / test draw from disjoint (capture, message) spans",
          f"{overlap} shared messages")

    # 2. scaler fitted on train only
    mean_diff = float(np.max(np.abs(
        data.X_train.to_numpy().mean(axis=0))))
    check(mean_diff < 0.05,
          "training features are centred (scaler was fitted on train)",
          f"max |mean| = {mean_diff:.4f}")

    val_mean = float(np.max(np.abs(data.X_val.to_numpy().mean(axis=0))))
    check(val_mean > 1e-6,
          "validation features are NOT re-centred (scaler was not refitted)",
          f"max |mean| = {val_mean:.6f}")

    # 3 + 5. cross-dataset target uses the frozen source scaler
    for target in cfg.data.datasets.keys():
        if target == name:
            continue
        before = data.scaler.mean_.copy()
        X_t, y_t, _ = prepare_target_dataset(
            cfg, target, groups, data.scaler, data.feature_names)
        after = data.scaler.mean_
        check(np.allclose(before, after),
              f"scaler unchanged after transforming target '{target}'")
        check(list(X_t.columns) == list(data.X_train.columns),
              f"feature columns and order match for target '{target}'")
        check(len(X_t) == len(y_t) and len(X_t) > 0,
              f"target '{target}' produced usable windows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default="configs/debug.yaml")
    parser.add_argument("--set", "-s", nargs="*", action="extend",
                        default=None, dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_cli_overrides(args.overrides or []))
    window_size = int(cfg.data.window_size)
    groups = list(cfg.features.groups)

    for name in cfg.data.datasets.keys():
        check_dataset(cfg, name, groups, window_size)

    LOG.info("=" * 70)
    if FAILURES:
        LOG.error("%d leakage check(s) FAILED: %s", len(FAILURES),
                  ", ".join(FAILURES))
        return 1
    LOG.info("all leakage checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
