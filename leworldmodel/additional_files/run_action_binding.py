#!/usr/bin/env python3
"""Run the three-arm counterfactual action-binding diagnostic."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

try:
    from . import run as base_runner
except ImportError:
    import run as base_runner


CONFIG_PATH = Path(__file__).with_name("action_binding_configs.yaml")


def _hydra(prefix: str, values: dict) -> list[str]:
    result = []
    for key, value in values.items():
        rendered = str(value).lower() if isinstance(value, bool) else value
        result.append(f"+loss.{prefix}.{key}={rendered}")
    return result


def load_action_binding_configs() -> dict:
    with CONFIG_PATH.open() as handle:
        raw = yaml.safe_load(handle)
    arms = {}
    for name, method in raw["arms"].items():
        overrides = list(raw["common_overrides"])
        overrides += ["+loss.control.enabled=true"]
        overrides += _hydra("control", method["control"])
        overrides += ["+loss.action_binding.enabled=true"]
        overrides += _hydra("action_binding", method["action_binding"])
        arms[name] = {
            "overrides": overrides,
            "pair_suites": ["colour"],
            "eval_overrides": list(raw["common_eval_overrides"]),
        }
    return {**raw, "arms": arms}


def main() -> int:
    base_runner.CONFIGS_YAML = CONFIG_PATH
    base_runner.CASE = "lewm-counterfactual-action-binding"
    base_runner.load_configs = load_action_binding_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; metrics.jsonl is the source of truth"
    )
    os.environ.setdefault("RESULTS_DIR", "./results/action-binding")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
