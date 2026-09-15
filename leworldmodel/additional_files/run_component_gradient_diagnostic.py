#!/usr/bin/env python3
"""Run short PushT/Reacher jobs with per-control-component diagnostics."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from . import run_control
    from .run_reacher_control import load_reacher_configs
except ImportError:
    import run as base_runner
    import run_control
    from run_reacher_control import load_reacher_configs


DIAGNOSTIC_OVERRIDE = "+loss.component_gradient_diagnostics.enabled=true"
DIAGNOSTIC_FREQUENCY = "+loss.component_gradient_diagnostics.every_n_steps=100"


def _diagnostic_arm(source: dict) -> dict:
    arm = copy.deepcopy(source)
    arm["overrides"].extend([DIAGNOSTIC_OVERRIDE, DIAGNOSTIC_FREQUENCY])
    arm.pop("pair_suites", None)
    return arm


def load_component_gradient_configs() -> dict:
    pusht = run_control.load_control_configs()
    reacher = load_reacher_configs()
    config = copy.deepcopy(pusht)
    config["arms"] = {
        "pusht_tagged_component_grad_diag": _diagnostic_arm(
            pusht["arms"]["control_aligned_pred1"]
        ),
        "reacher_clean_component_grad_diag": _diagnostic_arm(
            reacher["arms"]["reacher_clean_aligned"]
        ),
    }
    return config


def main() -> int:
    base_runner.CASE = "lewm-component-gradient-diagnostic"
    base_runner.load_configs = load_component_gradient_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; component-gradient metrics are in metrics.jsonl"
    )
    os.environ.setdefault(
        "RESULTS_DIR", "./results/component-gradient-diagnostic-seed0"
    )
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
