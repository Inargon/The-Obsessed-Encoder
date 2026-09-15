#!/usr/bin/env python3
"""Run the clean-Reacher orthogonal-drop mechanism comparator."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from .run_reacher_control import load_reacher_configs
except ImportError:
    import run as base_runner
    from run_reacher_control import load_reacher_configs


def load_reacher_parallel_only_configs() -> dict:
    """Change only the orthogonal admission rule of clean Reacher aligned."""
    config = copy.deepcopy(load_reacher_configs())
    arm = copy.deepcopy(config["arms"]["reacher_clean_aligned"])
    arm["overrides"].append(
        "+loss.aligned_gradient_routing.orthogonal_mode=drop"
    )
    config["arms"] = {"reacher_clean_parallel_only": arm}
    return config


def main() -> int:
    base_runner.CASE = "lewm-reacher-parallel-only"
    base_runner.load_configs = load_reacher_parallel_only_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; calibrated historical checkpoint evaluation is required"
    )
    os.environ.setdefault("RESULTS_DIR", "./results/reacher-parallel-only-seed0")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
