#!/usr/bin/env python3
"""Run matched clean-Reacher arms that isolate aligned-routing failures."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from .run_reacher_control import load_reacher_configs
except ImportError:
    import run as base_runner
    from run_reacher_control import load_reacher_configs


def _with_override(arm: dict, override: str) -> dict:
    result = copy.deepcopy(arm)
    result["overrides"].append(override)
    return result


def load_diagnostic_configs() -> dict:
    """Build a clean-only grid separating loss, norm, and direction effects."""
    base = copy.deepcopy(load_reacher_configs())
    jepa = base["arms"]["reacher_clean_jepa"]
    aligned = base["arms"]["reacher_clean_aligned"]

    control_only = copy.deepcopy(aligned)
    control_only["overrides"] = [
        value
        for value in control_only["overrides"]
        if value != "+loss.aligned_gradient_routing.enabled=true"
    ]

    arms = {
        "reacher_clean_jepa_diag": copy.deepcopy(jepa),
        "reacher_clean_control_only_diag": control_only,
        "reacher_clean_norm_scalar_diag": _with_override(
            aligned,
            "+loss.aligned_gradient_routing.routing_mode=norm_matched_scalar",
        ),
        "reacher_clean_aligned_diag": copy.deepcopy(aligned),
    }
    for label, retention in (("soft25", 0.25), ("soft50", 0.50), ("soft75", 0.75)):
        arms[f"reacher_clean_{label}_diag"] = _with_override(
            aligned,
            f"+loss.aligned_gradient_routing.minimum_retention={retention}",
        )

    base["arms"] = arms
    return base


def main() -> int:
    base_runner.CASE = "lewm-reacher-routing-diagnostic"
    base_runner.load_configs = load_diagnostic_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; historical-protocol checkpoint evaluation is required"
    )
    os.environ.setdefault(
        "RESULTS_DIR", "./results/reacher-routing-diagnostic-seed0"
    )
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
