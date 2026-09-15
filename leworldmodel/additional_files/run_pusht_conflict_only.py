#!/usr/bin/env python3
"""Run the tagged PushT negative-conflict-only routing comparator."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from . import run_control
except ImportError:
    import run as base_runner
    import run_control


def load_pusht_conflict_only_configs() -> dict:
    """Clone hard aligned and preserve every control-orthogonal component."""
    config = copy.deepcopy(run_control.load_control_configs())
    arm = copy.deepcopy(config["arms"]["control_aligned_pred1"])
    arm["overrides"].append(
        "+loss.aligned_gradient_routing.routing_mode=conflict_only"
    )
    config["arms"] = {"pusht_tagged_conflict_only_pred1": arm}
    return config


def main() -> int:
    base_runner.CASE = "lewm-pusht-conflict-only"
    base_runner.load_configs = load_pusht_conflict_only_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; SR and colour-pair geometry are in metrics.jsonl"
    )
    os.environ.setdefault(
        "RESULTS_DIR", "./results/pusht-tagged-conflict-only-seed0"
    )
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
