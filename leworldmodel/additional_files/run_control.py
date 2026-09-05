#!/usr/bin/env python3
"""Run the two control-grounded allocation methods without old baselines."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

try:
    from . import run as base_runner
except ImportError:
    import run as base_runner


CONFIG_PATH = Path(__file__).with_name("control_configs.yaml")


def load_control_configs() -> dict:
    with CONFIG_PATH.open() as handle:
        raw = yaml.safe_load(handle)

    arms = {}
    for name, method in raw["arms"].items():
        overrides = list(raw["common_overrides"])
        overrides.append("+loss.control.enabled=true")
        for key, value in method["control"].items():
            hydra_value = str(value).lower() if isinstance(value, bool) else value
            overrides.append(f"+loss.control.{key}={hydra_value}")
        overrides.extend(method.get("overrides", []))
        arms[name] = {
            "overrides": overrides,
            "pair_suites": ["colour"],
            "eval_overrides": list(raw["common_eval_overrides"]),
        }
    return {**raw, "arms": arms}


def main() -> int:
    base_runner.CONFIGS_YAML = CONFIG_PATH
    base_runner.CASE = "lewm-control"
    base_runner.load_configs = load_control_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped for control grid; metrics.jsonl is the source of truth"
    )
    os.environ.setdefault("RESULTS_DIR", "./results/control")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
