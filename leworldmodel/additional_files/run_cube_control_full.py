#!/usr/bin/env python3
"""Run the 10-epoch Cube grid under names distinct from the 10k pilot."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from .run_cube_control import load_cube_configs
except ImportError:
    import run as base_runner
    from run_cube_control import load_cube_configs


def load_full_configs() -> dict:
    cfg = copy.deepcopy(load_cube_configs())
    cfg["arms"] = {
        f"{name}_full": spec for name, spec in cfg["arms"].items()
    }
    return cfg


def main() -> int:
    base_runner.CASE = "lewm-cube-control-full"
    base_runner.load_configs = load_full_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped for Cube full grid; metrics.jsonl is the source of truth"
    )
    os.environ.setdefault("RESULTS_DIR", "./results/cube-control-full")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
