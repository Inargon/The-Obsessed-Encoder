#!/usr/bin/env python3
"""Render the fixed five-task Ours CEM pilot as a paper-ready summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


RESULTS = (
    ("Clean PushT", 46, 50),
    ("Tagged PushT", 44, 50),
    ("Reacher", 44, 50),
    ("TwoRoom", 48, 50),
    ("Cube", 40, 50),
)


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z / denominator * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return center - radius, center + radius


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row[0] for row in RESULTS]
    rates = np.asarray([row[1] / row[2] for row in RESULTS])
    intervals = np.asarray([wilson(row[1], row[2]) for row in RESULTS])
    errors = np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates))
    colors = ["#6F4E7C", "#B23A76", "#3B75AF", "#3E9C76", "#D18732"]

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    x = np.arange(len(labels))
    bars = ax.bar(x, 100 * rates, color=colors, width=0.68, zorder=2)
    ax.errorbar(
        x, 100 * rates, yerr=100 * errors, fmt="none", ecolor="#252525",
        elinewidth=1.4, capsize=4, zorder=3,
    )
    for bar, rate, (_, successes, trials) in zip(bars, rates, RESULTS):
        ax.text(
            bar.get_x() + bar.get_width() / 2, 100 * rate + 1.6,
            f"{100 * rate:.0f}%\n({successes}/{trials})",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Planning success rate (%)")
    ax.set_title("Ours with online CEM planning", fontweight="bold")
    fig.text(
        0.99, 0.015,
        "One fixed evaluation group, seed 42; bars show Wilson 95% intervals",
        ha="right", va="bottom", fontsize=8.5, color="#555555",
    )
    ax.grid(axis="y", alpha=0.20, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.89, bottom=0.18)
    fig.savefig(args.out_dir / "ours-five-task-cem-summary.png", dpi=240)
    plt.close(fig)

    payload = {
        "method_display": "Ours",
        "planner": "online CEM",
        "protocol": {
            "evaluation_seed": 42,
            "episodes_per_task": 50,
            "uncertainty": "Wilson score interval for evaluation episodes only",
            "warning": "single training run and single fixed evaluation group; not a multi-seed estimate",
            "excluded": "GC-IDM results",
        },
        "results": [
            {
                "task": label,
                "successes": successes,
                "episodes": trials,
                "success_rate": successes / trials,
                "wilson_ci95": list(wilson(successes, trials)),
            }
            for label, successes, trials in RESULTS
        ],
    }
    (args.out_dir / "ours-five-task-cem-summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OURS_RESULT_SUMMARY_COMPLETE {args.out_dir}")


if __name__ == "__main__":
    main()
