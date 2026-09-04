#!/usr/bin/env python3
"""Run the three capacity-allocation methods on RandGoal without baselines."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

try:
    from . import run as base_runner
except ImportError:  # direct execution from leworldmodel/additional_files
    import run as base_runner


CONFIG_PATH = Path(__file__).with_name("allocation_configs.yaml")


def load_allocation_configs() -> dict:
    with CONFIG_PATH.open() as handle:
        raw = yaml.safe_load(handle)

    arms = {}
    for name, method in raw["arms"].items():
        allocation = method["allocation"]
        overrides = list(raw["common_overrides"])
        overrides.append("+loss.allocation.enabled=true")
        for key, value in allocation.items():
            hydra_value = str(value).lower() if isinstance(value, bool) else value
            overrides.append(f"+loss.allocation.{key}={hydra_value}")
        arms[name] = {
            "overrides": overrides,
            "pair_suites": ["t_position"],
            "eval_overrides": list(raw["common_eval_overrides"]),
        }
    return {**raw, "arms": arms}


def main() -> int:
    base_runner.CONFIGS_YAML = CONFIG_PATH
    base_runner.CASE = "lewm-allocation"
    base_runner.load_configs = load_allocation_configs
    os.environ.setdefault("RESULTS_DIR", "./results/allocation")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
