"""Figure 10 — what the collapsed RandGoal encoder responds to.

Draws the projection-space encoding path for two sweeps that differ in one thing
only. On the left the goal-T follows the wander, on the right the grey block-T
follows the same wander. The collapsed encoder sweeps a wide, folded path when
the goal (the useless episode key) moves and barely stirs when the real state
moves. additional_files/readout.py is the measurement behind it.

    python additional_files/graphs/encoder_readout.py \
        --run-name randgoal_seed0 --epoch 5 --out-dir figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # leworldmodel/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from additional_files.readout import (
    PATH_POINTS, RANDGOAL_DATASET, SEED, fit_pca_basis, load_checkpoint,
    make_encoder, readout_paths, wander_pose,
)
from additional_files.stimuli import StimulusRenderer
from common.plotting import apply_style

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

STRIDE, FPS, HOLD_S, DPI = 4, 20, 1.0, 140
INK, MUTE, GRID, EDGE = "#1a1a1f", "#8a8780", "#eae7df", "#c9c6bd"
# reversed magma with the pale head clipped, so start/middle/end read at a glance
CMAP = ListedColormap(plt.get_cmap("magma_r")(np.linspace(0.18, 0.98, 256)))
START = dict(marker="o", color=CMAP(0.0), ms=9, mec="white", mew=1.4, zorder=8)
END = dict(marker="s", color=CMAP(1.0), ms=8.5, mec="white", mew=1.4, zorder=8)

HEADS = [("The goal-T moves", "block and agent held fixed"),
         ("The block-T moves", "agent and goal held fixed")]
TITLE = "The collapsed encoder follows the goal, not the world"
SUBTITLE = "the encoding path, drawn in the top two PC directions of the RandGoal dataset"
RC = {"figure.facecolor": "white", "axes.facecolor": "white",
      "axes.edgecolor": EDGE, "axes.linewidth": 1.1, "axes.grid": True,
      "grid.color": GRID, "grid.linewidth": 1.0, "grid.alpha": 1.0,
      "xtick.color": MUTE, "ytick.color": MUTE, "xtick.labelsize": 8.5,
      "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False}


def _panel(fig, ax, data: dict, col: int, xlim, ylim) -> dict:
    """One panel's static furniture; returns the artists the animation updates."""
    pts = data["scores"]
    lc = LineCollection([], lw=1.5, capstyle="round")
    ax.add_collection(lc)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))
    ax.set_xlabel("PC 1", color=MUTE, labelpad=3)
    if col == 0:
        ax.set_ylabel("PC 2", color=MUTE, labelpad=3)
    ax.tick_params(length=0)
    ax.plot(*pts[0], ls="", **START)
    end = ax.plot(*pts[-1], ls="", visible=False, **END)[0]
    (tip,) = ax.plot([], [], marker="o", ls="", ms=6.5, mec="white", mew=1.1, zorder=9)

    inset = ax.inset_axes([0.0, 1.03, 0.235, 0.235])  # above the grid, never on the path
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set(edgecolor=EDGE, linewidth=1.0)

    # heads placed from the axes box: set_title would be lifted above the inset
    pos = ax.get_position()
    cx = pos.x0 + pos.width / 2
    fig.text(cx, pos.y1 + 0.185, HEADS[col][0], ha="center", va="bottom",
             fontsize=13.5, fontweight="bold", color=INK)
    fig.text(cx, pos.y1 + 0.157, HEADS[col][1], ha="center", va="bottom",
             fontsize=10, color=MUTE)
    fig.text(pos.x0 + 0.118 * pos.width, pos.y1 + 0.135, "stimulus", ha="center",
             va="bottom", fontsize=7.5, color=MUTE)

    return dict(im=inset.imshow(data["insets"][0]), lc=lc, tip=tip, end=end, pts=pts,
                segs=np.stack([pts[:-1], pts[1:]], axis=1), insets=data["insets"])


def fig_anim(paths: list[dict], idx: np.ndarray, out: Path) -> None:
    apply_style()
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(10.4, 6.7))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.63, bottom=0.14, wspace=0.13)
    fig.text(0.5, 0.975, TITLE, ha="center", va="top", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.5, 0.928, SUBTITLE, ha="center", va="top", fontsize=11, color=MUTE, style="italic")

    # one square box holding both paths, so the two panels stay comparable
    allsc = np.concatenate([p["scores"] for p in paths])
    lo, hi = allsc.min(axis=0), allsc.max(axis=0)
    centre, radius = (lo + hi) / 2, 1.04 * (hi - lo).max() / 2
    xlim = (centre[0] - radius, centre[0] + radius)
    ylim = (centre[1] - radius, centre[1] + radius)

    axs = [fig.add_subplot(1, 2, c + 1) for c in range(2)]
    panels = [_panel(fig, ax, d, c, xlim, ylim) for c, (ax, d) in enumerate(zip(axs, paths))]
    axs[0].legend(handles=[plt.Line2D([], [], ls="", label="start", **START),
                           plt.Line2D([], [], ls="", label="end", **END)],
                  loc="upper right", frameon=True, framealpha=0.9, edgecolor=EDGE,
                  fontsize=9, handletextpad=0.3, borderpad=0.5)

    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), orientation="horizontal",
                      cax=fig.add_axes([0.34, 0.045, 0.32, 0.02]))
    cb.outline.set_edgecolor(EDGE)
    cb.outline.set_linewidth(0.8)
    cb.set_ticks([0.02, 0.98])
    cb.set_ticklabels(["start", "end"])
    cb.ax.tick_params(length=0, labelsize=9, colors=MUTE, pad=3)

    progress = np.linspace(0.0, 1.0, len(paths[0]["scores"]) - 1)
    order = np.concatenate([np.arange(len(idx)), np.full(int(FPS * HOLD_S), len(idx) - 1)])

    def draw(step: int):
        k = order[step]
        n, done = idx[k], k == len(idx) - 1
        for p in panels:
            # only the number of drawn segments grows; each keeps its own colour
            p["im"].set_data(p["insets"][k])
            p["lc"].set_segments(p["segs"][:n])
            p["lc"].set_colors(CMAP(progress[:n]))
            p["tip"].set_data([p["pts"][n, 0]], [p["pts"][n, 1]])
            p["tip"].set_markerfacecolor(CMAP(progress[min(n, len(progress) - 1)]))
            p["tip"].set_visible(not done)
            p["end"].set_visible(done)

    out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, draw, frames=len(order)).save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="randgoal_seed0")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--dataset", default=RANDGOAL_DATASET, help="dataset for the PC basis")
    p.add_argument("--points", type=int, default=PATH_POINTS, help="points per path")
    p.add_argument("--step", type=int, default=None, help="load weights_step_<N>.pt")
    p.add_argument("--epoch", type=int, default=None, help="load weights_epoch_<N>.pt")
    args = p.parse_args()

    poses = wander_pose(SEED, args.points)
    idx = np.arange(0, args.points, STRIDE)  # GIF frames, ending on the last point
    if idx[-1] != args.points - 1:
        idx = np.append(idx, args.points - 1)

    encode = make_encoder(load_checkpoint(args.run_name, step=args.step, epoch=args.epoch))
    mean, comps = fit_pca_basis(encode, args.dataset)
    paths = readout_paths(encode, StimulusRenderer().render, poses, mean, comps, idx)
    fig_anim(paths, idx, Path(args.out_dir) / "f10_encoder_readout.gif")


if __name__ == "__main__":
    main()
