#!/usr/bin/env python3
"""Run the paper-length clean-Reacher soft75 guardrail."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from .run_reacher_control import load_reacher_configs
except ImportError:
    import run as base_runner
    from run_reacher_control import load_reacher_configs


def load_reacher_soft75_full_configs() -> dict:
    config = copy.deepcopy(load_reacher_configs())
    arm = copy.deepcopy(config["arms"]["reacher_clean_aligned"])
    arm["overrides"].append(
        "+loss.aligned_gradient_routing.minimum_retention=0.75"
    )
    config["arms"] = {"reacher_clean_soft75_full": arm}
    return config


def main() -> int:
    base_runner.CASE = "lewm-reacher-soft75-full"
    base_runner.load_configs = load_reacher_soft75_full_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; calibrated historical checkpoint evaluation is required"
    )
    os.environ.setdefault("RESULTS_DIR", "./results/reacher-soft75-full-seed0")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
