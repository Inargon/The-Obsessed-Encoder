#!/usr/bin/env python3
"""Run clean-Reacher prediction-gradient component ablations."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from .run_reacher_control import load_reacher_configs
except ImportError:
    import run as base_runner
    from run_reacher_control import load_reacher_configs


def _arm_with_mode(source: dict, mode: str) -> dict:
    arm = copy.deepcopy(source)
    arm["overrides"].append(
        f"+loss.aligned_gradient_routing.routing_mode={mode}"
    )
    return arm


def load_reacher_orthogonal_component_configs() -> dict:
    """Build clean-only raw and cosine-gated orthogonal prediction arms."""
    config = copy.deepcopy(load_reacher_configs())
    aligned = config["arms"]["reacher_clean_aligned"]
    config["arms"] = {
        "reacher_clean_orthogonal_only_raw": _arm_with_mode(
            aligned, "orthogonal_only"
        ),
        "reacher_clean_orthogonal_only_gated": _arm_with_mode(
            aligned, "gated_orthogonal_only"
        ),
    }
    return config


def main() -> int:
    base_runner.CASE = "lewm-reacher-orthogonal-components"
    base_runner.load_configs = load_reacher_orthogonal_component_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; historical-protocol evaluation is required"
    )
    os.environ.setdefault(
        "RESULTS_DIR", "./results/reacher-orthogonal-components-seed0"
    )
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
