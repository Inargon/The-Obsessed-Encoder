#!/usr/bin/env python3
"""Run tagged-PushT prediction-gradient component ablations."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from . import run_control
except ImportError:
    import run as base_runner
    import run_control


def _arm_with_mode(source: dict, mode: str) -> dict:
    arm = copy.deepcopy(source)
    arm["overrides"].append(
        f"+loss.aligned_gradient_routing.routing_mode={mode}"
    )
    return arm


def load_pusht_orthogonal_component_configs() -> dict:
    """Split the aligned update into raw and cosine-gated orthogonal arms."""
    config = copy.deepcopy(run_control.load_control_configs())
    aligned = config["arms"]["control_aligned_pred1"]
    config["arms"] = {
        "pusht_tagged_orthogonal_only_raw_pred1": _arm_with_mode(
            aligned, "orthogonal_only"
        ),
        "pusht_tagged_orthogonal_only_gated_pred1": _arm_with_mode(
            aligned, "gated_orthogonal_only"
        ),
    }
    return config


def main() -> int:
    base_runner.CASE = "lewm-pusht-orthogonal-components"
    base_runner.load_configs = load_pusht_orthogonal_component_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; SR and gradient geometry are in metrics.jsonl"
    )
    os.environ.setdefault(
        "RESULTS_DIR", "./results/pusht-orthogonal-components-seed0"
    )
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
