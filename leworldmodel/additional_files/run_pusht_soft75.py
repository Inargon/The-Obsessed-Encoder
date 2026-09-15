#!/usr/bin/env python3
"""Run the tagged PushT soft-retention guardrail experiment."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from . import run_control
except ImportError:
    import run as base_runner
    import run_control


def load_pusht_soft75_configs() -> dict:
    """Clone the completed hard-aligned arm and change only its retention floor."""
    config = copy.deepcopy(run_control.load_control_configs())
    soft75 = copy.deepcopy(config["arms"]["control_aligned_pred1"])
    soft75["overrides"].append(
        "+loss.aligned_gradient_routing.minimum_retention=0.75"
    )
    config["arms"] = {"pusht_tagged_soft75_pred1": soft75}
    return config


def main() -> int:
    base_runner.CASE = "lewm-pusht-soft75"
    base_runner.load_configs = load_pusht_soft75_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; SR and colour-pair geometry are in metrics.jsonl"
    )
    os.environ.setdefault("RESULTS_DIR", "./results/pusht-tagged-soft75-seed0")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
