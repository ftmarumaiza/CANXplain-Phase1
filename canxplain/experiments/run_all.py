"""One-command reproduction of the entire Phase 1 experiment.

Runs, in order:
  1. the data/feature check (Tables 1-2, Figure 1)
  2. the full ablation study incl. cross-dataset (Tables 3-8, Figures 2-7)
  3. the fitness weight sensitivity sweep (Table 9, Figure 8)

Everything lands under run.output_dir from the config.

    python -m canxplain.experiments.run_all --config configs/debug.yaml
    python -m canxplain.experiments.run_all --config configs/full.yaml
"""
from __future__ import annotations

import sys
import time

from ..utils import get_logger

LOG = get_logger()


def _invoke(module_main, argv):
    saved = sys.argv
    sys.argv = argv
    try:
        module_main()
    finally:
        sys.argv = saved


def main():
    from . import run_ablation, run_pipeline, run_sensitivity

    argv = sys.argv[1:]
    start = time.perf_counter()

    stages = [
        ("data and feature pipeline", run_pipeline.main, ["run_pipeline"] + argv),
        ("ablation study", run_ablation.main, ["run_ablation"] + argv),
        ("fitness sensitivity", run_sensitivity.main, ["run_sensitivity"] + argv),
    ]

    for i, (label, entry, stage_argv) in enumerate(stages, start=1):
        LOG.info("=" * 70)
        LOG.info("STAGE %d/%d — %s", i, len(stages), label)
        LOG.info("=" * 70)
        try:
            _invoke(entry, stage_argv)
        except SystemExit:
            raise
        except Exception as exc:
            LOG.error("stage '%s' failed: %s", label, exc, exc_info=True)
            LOG.error("continuing to the next stage; results will be partial")

    LOG.info("=" * 70)
    LOG.info("all stages finished in %.1f minutes", (time.perf_counter() - start) / 60)
    LOG.info("=" * 70)


if __name__ == "__main__":
    main()
