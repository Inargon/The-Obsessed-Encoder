#!/usr/bin/env python3
"""Run the paper-length Reacher grid under names distinct from the 10k pilot."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from .run_reacher_control import load_reacher_configs
except ImportError:
    import run as base_runner
    from run_reacher_control import load_reacher_configs


def load_full_configs() -> dict:
    """Return the matched clean/tagged JEPA/aligned grid with isolated names."""
    cfg = copy.deepcopy(load_reacher_configs())
    cfg["arms"] = {
        f"{name}_full": spec for name, spec in cfg["arms"].items()
    }
    return cfg


def main() -> int:
    base_runner.CASE = "lewm-reacher-control-full"
    base_runner.load_configs = load_full_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped for Reacher full grid; metrics/checkpoints are the source of truth"
    )
    os.environ.setdefault("RESULTS_DIR", "./results/reacher-control-full")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
