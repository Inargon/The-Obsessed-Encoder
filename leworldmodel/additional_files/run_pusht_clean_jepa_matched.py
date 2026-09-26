#!/usr/bin/env python3
"""Train a clean PushT LeWM/JEPA baseline matched to the local Ours run.

Only the method differs: this arm uses the upstream LeWM prediction + SIGReg
objective and carries no pixel tag, control objective, or gradient router.
"""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
except ImportError:
    import run as base_runner


# ``main`` installs our loader on ``base_runner``.  Keep the original callable
# before doing so; looking it up through ``base_runner`` later would recurse.
UPSTREAM_LOAD_CONFIGS = base_runner.load_configs


def load_pusht_clean_jepa_matched_configs() -> dict:
    config = copy.deepcopy(UPSTREAM_LOAD_CONFIGS())
    baseline = copy.deepcopy(config["arms"]["baseline"])

    # Give smoke and full runs distinct checkpoint namespaces.  This prevents
    # a short validation run (or any historical `baseline_seed0` run) from
    # contaminating the full campaign through the global checkpoint cache.
    arm_name = os.environ.get(
        "PUSHT_CLEAN_JEPA_ARM", "pusht_clean_jepa_matched_full"
    )
    config["arms"] = {arm_name: baseline}
    return config


def main() -> int:
    base_runner.CASE = "lewm-pusht-clean-jepa-matched"
    base_runner.load_configs = load_pusht_clean_jepa_matched_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped; fixed-protocol evaluation is submitted separately"
    )
    os.environ.setdefault(
        "RESULTS_DIR", "./results/pusht-clean-jepa-matched-20260926"
    )
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
