"""The metric-curve figures: the fig 8 crossover and the fig 9 pair cosine.

Built on the shared style-v2 factory in common/plotting, the same one the
dinov3 and lejepa cases use, so all three read as one system: compact panel
rows, one horizontal legend, shared axis titles, and a mean +/- std band over
the seeds present under RESULTS_DIR. Reads exclusively from
RESULTS_DIR/<arm>_seed<k>/metrics.jsonl; no wandb, no GPU.

    python additional_files/graphs/training_curves.py --results-dir results --out-dir figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from common.plotting import (  # noqa: E402
    C_CLEAN, C_CONTROL, C_RANDGOAL, C_WATERMARKED, curve_row_figure,
    history_frame, pair_row_figure,
)

# The crossover arms, in legend order. Roles map onto the shared palette:
# baseline blue, the matched control green dashed, the predictable arm red,
# the task-semantic randgoal arm purple.
CROSSOVER_ARMS = (
    "baseline", "colored_square_frame", "colored_square_episode", "randgoal",
)
ARM_STYLES = {
    "baseline": dict(color=C_CLEAN, dash="solid", width=2.0),
    "colored_square_frame": dict(color=C_CONTROL, dash="dash", width=2.0),
    "colored_square_episode": dict(color=C_WATERMARKED, dash="solid", width=2.2),
    "randgoal": dict(color=C_RANDGOAL, dash="solid", width=2.2),
}
ARM_LABELS = {
    "baseline": "baseline",
    "colored_square_frame": "per-frame control",
    "colored_square_episode": "per-episode square",
    "randgoal": "RandGoal",
}

# pred_loss lands every 50 steps and is noisy enough to hide the arm ordering;
# 25 points is ~1250 steps of rolling mean, enough to read the curves apart
# while the red arm's late spikes survive. The eval series is already sparse,
# so it is drawn raw.
CROSSOVER_PANELS = [
    dict(metric="fit/pred_loss", label="prediction loss", log=True, smooth=25),
    dict(metric="eval/success_rate", label="planner success rate"),
]

# Pair panels: every perturbed arm on one row, each probing the signal its own
# data carries (corner colour for the squares, destination pose for randgoal).
# The metric keys carry no suite name, so a panel is only meaningful for the
# arm that probed it.
PAIR_PANELS = [
    ("colored_square_frame", "per-frame square (control)"),
    ("colored_square_episode", "per-episode square"),
    ("randgoal", "randgoal"),
]
# (pairing, label, colour, dash); "baseline" is the ~0 null reference.
PAIRINGS = [
    ("same_content", "same content, different tag", C_CLEAN, "solid"),
    ("same_tag", "different content, same tag", C_WATERMARKED, "solid"),
    ("baseline", "both different", C_CONTROL, "dot"),
]

# The pair series is logged far more densely than the eval; 9 points of rolling
# mean settles the late-training jitter without moving where the curves sit.
PAIR_SMOOTH = 9

REP_LABELS = {"backbone": "backbone", "projection": "projection"}


def fig_crossover(history, out: Path) -> None:
    curve_row_figure(
        history, CROSSOVER_PANELS, arms=CROSSOVER_ARMS, styles=ARM_STYLES,
        labels=ARM_LABELS, title="LeWM, colored-square and RandGoal arms",
        xlabel="training step", out_path=str(out),
    )
    print(f"wrote {out}")


def fig_pair(history, rep: str, out: Path) -> None:
    lines = [(f"pair/{rep}/{pairing}/cos_mean", label, color, dash)
             for pairing, label, color, dash in PAIRINGS]
    pair_row_figure(
        history, PAIR_PANELS, lines,
        title=f"LeWM pair cosine ({REP_LABELS[rep]})", xlabel="training step",
        ylabel="centered pair cosine", out_path=str(out), smooth=PAIR_SMOOTH,
    )
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = history_frame(args.results_dir)
    if history.empty:
        raise SystemExit(
            f"no metrics under {args.results_dir} — train the arms first "
            "(additional_files/run.py)"
        )

    fig_crossover(history, out / "f8_lewm_crossover.png")
    # The blog carries the backbone read; the projection read is the same
    # measurement one layer up, kept for inspection.
    fig_pair(history, "backbone", out / "f9_lewm_pair.png")
    fig_pair(history, "projection", out / "f9_lewm_pair_projection.png")


if __name__ == "__main__":
    main()
