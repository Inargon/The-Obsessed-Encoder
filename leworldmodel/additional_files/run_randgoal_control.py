#!/usr/bin/env python3
"""Run control-guided methods on the randomized-destination PushT data."""

from __future__ import annotations

import os

try:
    from . import run as base_runner
    from . import run_control
except ImportError:
    import run as base_runner
    import run_control


DATASET = "pusht_scripted_goal_train.lance"


def load_randgoal_control_configs() -> dict:
    """Retarget the control grid without carrying over the watermark setup."""
    cfg = run_control.load_control_configs()
    for arm in cfg["arms"].values():
        arm["overrides"] = [
            f"data.dataset.name={DATASET}",
            *(
                override
                for override in arm["overrides"]
                if not override.startswith("data.dataset.name=")
                and not override.startswith("+pixel_tag.")
            ),
        ]
        arm["eval_overrides"] = [f"+eval.dataset_name={DATASET}"]
        arm["pair_suites"] = ["t_position"]
    return cfg


def main() -> int:
    base_runner.CASE = "lewm-randgoal-control"
    base_runner.load_configs = load_randgoal_control_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped for randgoal control grid; metrics.jsonl is the source of truth"
    )
    os.environ.setdefault("RESULTS_DIR", "./results/randgoal-control")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
