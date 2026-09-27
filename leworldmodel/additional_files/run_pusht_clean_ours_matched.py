#!/usr/bin/env python3
"""Train full Ours on clean PushT under the matched JEPA protocol."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from . import run_control
except ImportError:
    import run as base_runner
    import run_control


def load_pusht_clean_ours_matched_configs() -> dict:
    """Keep the full control objective/router and remove only the pixel tag."""
    config = copy.deepcopy(run_control.load_control_configs())
    arm = copy.deepcopy(config["arms"]["control_aligned_pred1"])
    arm["overrides"] = [
        value
        for value in arm["overrides"]
        if not value.startswith("+pixel_tag.")
    ]
    arm["eval_overrides"] = [
        value
        for value in arm.get("eval_overrides", [])
        if not value.startswith("+eval.tag_")
    ]
    arm.pop("pair_suites", None)

    arm_name = os.environ.get(
        "PUSHT_CLEAN_OURS_ARM", "pusht_clean_ours_matched_full"
    )
    config["arms"] = {arm_name: arm}
    return config


def main() -> int:
    base_runner.CASE = "lewm-pusht-clean-ours-matched"
    base_runner.load_configs = load_pusht_clean_ours_matched_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; fixed-protocol evaluation is submitted separately"
    )
    os.environ.setdefault(
        "RESULTS_DIR", "./results/pusht-clean-ours-matched-20260927"
    )
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
