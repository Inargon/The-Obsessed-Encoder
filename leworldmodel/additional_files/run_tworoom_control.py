#!/usr/bin/env python3
"""Run matched clean/tagged JEPA and aligned-gradient arms on TwoRoom."""

from __future__ import annotations

import copy
import os

try:
    from . import run as base_runner
    from . import run_control
except ImportError:
    import run as base_runner
    import run_control


def load_tworoom_configs() -> dict:
    """Build a four-arm grid without carrying PushT-only pair diagnostics."""
    control_grid = run_control.load_control_configs()
    aligned_source = control_grid["arms"]["control_aligned_pred1"]

    def aligned_overrides(tagged: bool) -> list[str]:
        overrides = [
            value
            for value in aligned_source["overrides"]
            if not value.startswith("data.dataset.name=")
            and not value.startswith("+pixel_tag.")
        ]
        out = ["data=tworoom", *overrides]
        if tagged:
            out.extend(["+pixel_tag.mode=video", "+pixel_tag.size=5"])
        return out

    def eval_overrides(tagged: bool) -> list[str]:
        out = ["+eval.config=tworoom"]
        if tagged:
            out.extend(["+eval.tag_mode=video", "+eval.tag_size=5"])
        return out

    arms = {
        "tworoom_clean_jepa": {
            "overrides": ["data=tworoom"],
            "eval_overrides": eval_overrides(False),
        },
        "tworoom_clean_aligned": {
            "overrides": aligned_overrides(False),
            "eval_overrides": eval_overrides(False),
        },
        "tworoom_tagged_jepa": {
            "overrides": [
                "data=tworoom",
                "+pixel_tag.mode=video",
                "+pixel_tag.size=5",
            ],
            "eval_overrides": eval_overrides(True),
        },
        "tworoom_tagged_aligned": {
            "overrides": aligned_overrides(True),
            "eval_overrides": eval_overrides(True),
        },
    }
    cfg = copy.deepcopy(control_grid)
    cfg["arms"] = arms
    return cfg


def main() -> int:
    base_runner.CASE = "lewm-tworoom-control"
    base_runner.load_configs = load_tworoom_configs
    base_runner.render_figures = lambda args: print(
        "[figures] skipped for TwoRoom grid; metrics.jsonl is the source of truth"
    )
    os.environ.setdefault("RESULTS_DIR", "./results/tworoom-control")
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
