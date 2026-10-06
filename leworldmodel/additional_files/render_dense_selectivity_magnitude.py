#!/usr/bin/env python3
"""Render magnitude-aware paper figures from dense selectivity JSON.

The original allocation maps normalize every clip to unit mass.  That is useful
for showing *where* the remaining sensitivity is allocated, but it hides large
between-model differences in absolute intervention response.  This renderer
keeps the allocation view and adds shared-scale and absolute-magnitude views.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KEYS = ("context_latent", "context_cost", "goal_cost")
TITLES = {
    "context_latent": "Context latent",
    "context_cost": "Context planning cost",
    "goal_cost": "Goal planning cost",
}
COLORS = {"jepa": "#E69F63", "ours": "#CC79A7"}
DISPLAY = {"jepa": "JEPA", "ours": "Ours"}


def bootstrap_mean(values: np.ndarray, seed: int, samples: int = 20_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    draws = values[indices].mean(axis=1)
    return float(values.mean()), np.quantile(draws, [0.025, 0.975])


def absolute_tag_values(run: dict, key: str) -> np.ndarray:
    fraction = np.asarray(run["per_clip"]["tag_region_mass"][key], dtype=float)
    total = np.asarray(run["per_clip"]["total_map_mass"][key], dtype=float)
    if fraction.shape != total.shape:
        raise ValueError(f"mismatched tag fraction and total mass for {key}")
    return fraction * total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text())
    out_dir = args.out_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [label for label in ("jepa", "ours") if label in data["runs"]]
    if labels != ["jepa", "ours"]:
        raise ValueError("expected matched jepa and ours runs")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Shared absolute scale within each metric.  Unlike the unit-mass map,
    # darkness here genuinely means a smaller intervention response.
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), squeeze=False)
    for col, key in enumerate(KEYS):
        aggregates = {
            label: np.asarray(data["runs"][label]["maps"]["mean"][key], dtype=float).mean(axis=0)
            for label in labels
        }
        vmax = max(float(aggregates[label].max()) for label in labels)
        images = []
        for row, label in enumerate(labels):
            axis = axes[row, col]
            images.append(
                axis.imshow(
                    aggregates[label], cmap="magma", vmin=0.0, vmax=max(vmax, 1e-12),
                    interpolation="nearest",
                )
            )
            axis.add_patch(
                Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="#00C853", linewidth=2)
            )
            if row == 0:
                axis.set_title(TITLES[key])
            if col == 0:
                axis.set_ylabel(DISPLAY[label], fontsize=12, fontweight="bold")
            axis.set_xticks([])
            axis.set_yticks([])
        figure.colorbar(images[-1], ax=axes[:, col].tolist(), fraction=0.035, pad=0.025)
    figure.suptitle("Absolute matched intervention sensitivity (shared scale per column)", fontsize=14)
    figure.subplots_adjust(left=0.07, right=0.94, bottom=0.06, top=0.90, wspace=0.28, hspace=0.12)
    figure.savefig(out_dir / "dense-selectivity-aggregate-absolute.png", dpi=240)
    plt.close(figure)

    # Pair allocation fraction with absolute response.  The left panel answers
    # where sensitivity is allocated; the right answers how large it is.
    x = np.arange(len(KEYS), dtype=float)
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    summaries = {"fraction": {}, "absolute": {}}
    for index, label in enumerate(labels):
        offset = (index - 0.5) * width
        fraction_means, fraction_errors = [], [[], []]
        absolute_means, absolute_errors = [], [[], []]
        summaries["fraction"][label] = {}
        summaries["absolute"][label] = {}
        for key_index, key in enumerate(KEYS):
            fraction = np.asarray(
                data["runs"][label]["per_clip"]["tag_region_mass"][key], dtype=float
            )
            absolute = absolute_tag_values(data["runs"][label], key)
            for name, values, means, errors in (
                ("fraction", fraction, fraction_means, fraction_errors),
                ("absolute", absolute, absolute_means, absolute_errors),
            ):
                mean, interval = bootstrap_mean(values, args.seed + key_index)
                means.append(mean)
                errors[0].append(mean - interval[0])
                errors[1].append(interval[1] - mean)
                summaries[name][label][key] = {
                    "mean": mean,
                    "bootstrap_clip_ci95": interval.tolist(),
                }
        axes[0].bar(
            x + offset, fraction_means, width, yerr=np.asarray(fraction_errors),
            capsize=3, color=COLORS[label], label=DISPLAY[label], alpha=0.92,
        )
        bars = axes[1].bar(
            x + offset, absolute_means, width, yerr=np.asarray(absolute_errors),
            capsize=3, color=COLORS[label], label=DISPLAY[label], alpha=0.92,
        )
        for bar, value in zip(bars, absolute_means):
            axes[1].annotate(
                f"{value:.2g}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8,
            )

    tick_labels = ["Latent/context", "Cost/context", "Cost/goal"]
    axes[0].set_xticks(x, tick_labels)
    axes[1].set_xticks(x, tick_labels)
    axes[0].set_ylabel("Fraction of sensitivity in tag cell")
    axes[0].set_ylim(0, 1.02)
    grid_size = int(data["protocol"]["grid_size"])
    uniform = 1.0 / (grid_size * grid_size)
    axes[0].axhline(uniform, color="0.35", linestyle="--", linewidth=1.2)
    axes[0].text(
        0.02, uniform + 0.025, f"uniform area baseline = {100 * uniform:.2f}%",
        transform=axes[0].get_yaxis_transform(), fontsize=8, color="0.3",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Absolute tag intervention sensitivity (log scale)")
    axes[1].text(
        0.02, 0.02, "Compare methods within each metric",
        transform=axes[1].transAxes, fontsize=8, color="0.35",
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_title("Where is sensitivity allocated?")
    axes[1].set_title("How large is the tag effect?")
    figure.tight_layout()
    figure.savefig(out_dir / "dense-tag-sensitivity-summary.png", dpi=240)
    plt.close(figure)

    (out_dir / "dense-selectivity-magnitude-summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n"
    )
    print(f"DENSE_SELECTIVITY_MAGNITUDE_RENDER_COMPLETE {out_dir}")


if __name__ == "__main__":
    main()
