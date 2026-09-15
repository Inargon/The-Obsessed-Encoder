#!/usr/bin/env python3
"""Run cosine-aligned training on clean PushT without a pixel tag."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from . import run_control
except ImportError:
    import run as base_runner
    import run_control


def load_pusht_clean_aligned_configs() -> dict:
    """Remove only the watermark from the standard cosine-aligned arm."""
    config = copy.deepcopy(run_control.load_control_configs())
    arm = copy.deepcopy(config["arms"]["control_aligned_pred1"])

    config["common_overrides"] = ["data.dataset.name=pusht_expert_train.h5"]
    config["common_eval_overrides"] = []
    config["arms"] = {"pusht_clean_aligned": arm}
    return config


def main() -> int:
    base_runner.CASE = "lewm-pusht-clean-aligned"
    base_runner.load_configs = load_pusht_clean_aligned_configs
    os.environ.setdefault("RESULTS_DIR", "./results/pusht-clean-aligned-seed0")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
