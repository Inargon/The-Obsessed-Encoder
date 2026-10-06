#!/usr/bin/env python3
"""Plot the training-time gradient geometry recorded by Bloop runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = {
    "projection": "bloop_projection_ratio",
    "retained": "bloop_auxiliary_retained_fraction",
    "main_ema": "bloop_main_ema_cosine",
    "projected_ema": "bloop_projected_ema_cosine",
}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=/path/to/metrics.jsonl")
    label, path = value.split("=", 1)
    return label, Path(path)


def read_records(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            if line_number == len(lines):
                break
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return records


def extract(records: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    keys = (f"fit/{metric}", metric)
    steps, values = [], []
    for record in records:
        key = next((candidate for candidate in keys if candidate in record), None)
        if key is None:
            continue
        value = float(record[key])
        if not np.isfinite(value):
            continue
        steps.append(int(record.get("step", len(steps))))
        values.append(value)
    if not values:
        raise ValueError(f"metric {metric!r} is absent")
    return np.asarray(steps), np.asarray(values)


def bin_series(
    steps: np.ndarray,
    values: np.ndarray,
    bins: int = 160,
    reducer=np.mean,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(steps, kind="stable")
    steps, values = steps[order], values[order]
    groups = np.array_split(np.arange(len(values)), min(bins, len(values)))
    return (
        np.asarray([steps[group].mean() for group in groups]),
        np.asarray([reducer(values[group]) for group in groups]),
    )


def late_summary(records: list[dict]) -> dict[str, float]:
    series = {name: extract(records, metric)[1] for name, metric in METRICS.items()}
    count = min(len(values) for values in series.values())
    start = max(0, int(0.8 * count))
    projection = series["projection"][:count]
    return {
        "points": int(count),
        "late_conflict_fraction": float(np.mean(projection[start:] < 0)),
        "late_projection_coefficient": float(np.mean(projection[start:])),
        "late_auxiliary_retained_fraction": float(
            np.mean(series["retained"][:count][start:])
        ),
        "late_main_ema_cosine": float(np.mean(series["main_ema"][:count][start:])),
        "late_projected_ema_cosine": float(
            np.mean(series["projected_ema"][:count][start:])
        ),
    }


def render(runs: dict[str, list[dict]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    panels = (
        ("projection", "EMA-conflicting auxiliary updates", "%", True),
        ("retained", "Auxiliary gradient norm retained", "fraction", False),
        ("main_ema", "Current control / EMA cosine", "cosine", False),
        ("projected_ema", "Projected auxiliary / EMA cosine", "cosine", False),
    )
    for run_index, (label, records) in enumerate(runs.items()):
        color = colors(run_index)
        for axis, (name, title, ylabel, is_conflict) in zip(axes.flat, panels):
            steps, values = extract(records, METRICS[name])
            if is_conflict:
                values = (values < 0).astype(float) * 100.0
            x, y = bin_series(steps, values)
            axis.plot(x, y, color=color, lw=1.8, label=label)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.set_xlabel("training step")
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylim(-2, 102)
    axes[0, 1].axhline(1.0, color="0.35", ls="--", lw=1)
    axes[1, 1].axhline(0.0, color="0.35", ls="--", lw=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=min(5, len(labels)),
        frameon=False,
    )
    fig.suptitle("EMA-guided gradient geometry throughout training", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.84,
                        wspace=0.24, hspace=0.34)
    appendix = out.with_name(out.stem + "-appendix" + out.suffix)
    fig.savefig(appendix, dpi=220)
    plt.close(fig)

    # Main-paper figure: retain only the two quantities needed to explain the
    # routing mechanism.  Cosine diagnostics remain available in the appendix.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))
    main_panels = (
        ("projection", "EMA-conflicting auxiliary updates", "%", True),
        ("retained", "Auxiliary gradient norm retained", "fraction", False),
    )
    for run_index, (label, records) in enumerate(runs.items()):
        color = colors(run_index)
        for axis, (name, title, ylabel, is_conflict) in zip(axes, main_panels):
            steps, values = extract(records, METRICS[name])
            if is_conflict:
                values = (values < 0).astype(float) * 100.0
            x, y = bin_series(steps, values, bins=100)
            axis.plot(x, y, color=color, lw=1.8, label=label)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.set_xlabel("training step")
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(-2, 102)
    axes[1].set_ylim(0.58, 1.01)
    axes[1].axhline(1.0, color="0.35", ls="--", lw=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("EMA-guided auxiliary-gradient routing", y=0.995, fontsize=15)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.74, wspace=0.22)
    fig.savefig(out, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    missing = [str(path) for _, path in args.run if not path.exists()]
    if missing:
        parser.error("missing metrics: " + ", ".join(missing))
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")
    runs = {label: read_records(path) for label, path in args.run}
    args.out_dir.mkdir(parents=True)
    render(runs, args.out_dir / "bloop-training-mechanism.png")
    payload = {
        "runs": {
            label: {
                "metrics_jsonl": str(path),
                **late_summary(runs[label]),
            }
            for label, path in args.run
        },
        "protocol": {
            "curves": "160 equal-count bins over all recorded diagnostic points",
            "summary": "mean over the final 20% of diagnostic points",
            "conflict": "negative auxiliary dot product with control-gradient EMA",
        },
    }
    (args.out_dir / "bloop-training-mechanism.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    print(f"BLOOP_TRAINING_MECHANISM_COMPLETE {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
