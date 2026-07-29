"""Figure 7 — the datasets as real frames: fixed-goal vs random-goal episodes
(top), and one baseline episode tagged per-episode vs per-frame (bottom),
stamped by the same tag code training uses.

Also writes an animated GIF companion of four panels in two side-by-side
couples. Each panel plays three episodes back to back with an episode counter:
the goal couple contrasts the fixed goal with the random one across episode
cuts, the square couple contrasts the per-episode colour (changes only at
cuts, shown in a zoom inset) with the per-frame colour (changes constantly).

    python additional_files/graphs/dataset_montage.py --out figures/f7_lewm_datasets.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # leworldmodel/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from additional_files.pixel_tag import PixelTag
from common.plotting import apply_style, save_fig

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import ConnectionPatch

# animation pacing: every ANIM_STRIDE-th step at ANIM_FPS; panels whose
# episodes end early hold their last frame, then everything holds ANIM_HOLD_S
# before the loop
ANIM_STRIDE = 3
ANIM_FPS = 12
ANIM_HOLD_S = 1.0
ANIM_DPI = 100
# side of the top-left corner region the square panels magnify in their inset
ZOOM_PX = 14
# red used for the zoom outline and its leader line
TAG_RED = "#d62728"


def _frames(dataset_name: str, ep: int, steps: list[int]) -> list[np.ndarray]:
    """Native uint8 (H, W, 3) frames from a dataset episode."""
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    ds = swm.data.load_dataset(dataset_name, cache_dir=cache_dir, keys_to_load=["pixels"])
    hi = max(steps) + 1
    clip = ds._load_slice(ep, 0, hi)["pixels"]  # (T, C, H, W) uint8
    return [clip[s].permute(1, 2, 0).numpy() for s in steps]


def _episode_steps(dataset_name: str, ep: int, stride: int = ANIM_STRIDE) -> list[int]:
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    ds = swm.data.load_dataset(dataset_name, cache_dir=cache_dir, keys_to_load=["pixels"])
    return list(range(0, int(ds.lengths[ep]), stride))


def fig_dataset_animation(
    groups: list[tuple[str, list[tuple[list[list[np.ndarray]], str, bool]]]],
    out: Path,
) -> None:
    """Panels in titled couples, each playing its episodes back to back.

    groups: (group_title, panels) pairs; each panel is (episodes, title, zoom)
    where episodes is a list of frame clips. A counter inside the panel names
    the episode being played, panels that finish early hold their last frame,
    and zoom panels magnify the tagged corner in a framed inset.
    """
    apply_style()
    fig = plt.figure(figsize=(11.6, 4.2), constrained_layout=True)
    # keep the auto layout below this line so the top band is free for the
    # group titles, which sit above the insets
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))

    # one shared grid so all panels get identical axes and the frames line up
    # exactly; a narrow spacer column between groups keeps the couples reading
    # as groups. Every panel carries a same-sized inset so the reserved
    # headroom (and thus each frame's size) matches across all four.
    ratios: list[float] = []
    cols_per_group: list[list[int]] = []
    for gi, (_, group) in enumerate(groups):
        if gi > 0:
            ratios.append(0.3)  # spacer between groups
        start = len(ratios)
        ratios.extend([1.0] * len(group))
        cols_per_group.append(list(range(start, start + len(group))))
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.06)

    panels = []
    group_axes: list[tuple[str, list, list]] = []
    for (group_title, group), cols in zip(groups, cols_per_group):
        axs, insets = [], []
        for (episodes, title, zoom), col in zip(group, cols):
            ax = fig.add_subplot(gs[0, col])
            frames = [frame for clip in episodes for frame in clip]
            cuts = np.cumsum([len(clip) for clip in episodes])
            im = ax.imshow(frames[0])
            # panel title placed by hand just above the frame; set_title would
            # be lifted above the inset by constrained_layout and clash with
            # the group title
            ax.text(0.5, 1.01, title, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=10)
            ax.set_axis_off()
            counter = ax.text(
                0.04, 0.04, "", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            )
            axins = ax.inset_axes([0.03, 1.06, 0.20, 0.20])
            axins.set_xticks([])
            axins.set_yticks([])
            inset_im = None
            if zoom:
                # small zoom of the tagged corner, parked in the margin above
                # the square so it covers none of the scene, with one thin
                # leader line from the square up to the zoom
                inset_im = axins.imshow(frames[0])
                axins.set_xlim(-0.5, ZOOM_PX - 0.5)
                axins.set_ylim(ZOOM_PX - 0.5, -0.5)
                # all four edges back on (the house style hides top/right),
                # thick enough to survive GIF palette quantization
                for spine in axins.spines.values():
                    spine.set_visible(True)
                    spine.set_color(TAG_RED)
                    spine.set_linewidth(1.8)
                ax.add_artist(ConnectionPatch(
                    xyA=(2.5, -0.5), coordsA=ax.transData,
                    xyB=(0.5, 0.0), coordsB=axins.transAxes,
                    color=TAG_RED, linewidth=1.1, zorder=5, clip_on=False,
                ))
                # dot sits in the gap just above the square, on the leader line,
                # so it marks the square without covering its colour
                ax.plot(2.5, -4.5, marker="o", color=TAG_RED, markersize=2.5,
                        zorder=6, clip_on=False)
            else:
                # invisible placeholder: reserves the same headroom so this
                # frame ends up the same size as the zoom panels'
                for spine in axins.spines.values():
                    spine.set_visible(False)
                axins.patch.set_alpha(0)
            axs.append(ax)
            insets.append(axins)
            panels.append((im, inset_im, counter, frames, cuts))
        group_axes.append((group_title, axs, insets))

    # group titles centred over each couple, just above the insets. Positions
    # come from the finalised layout so the titles clear the zooms.
    fig.canvas.draw()
    title_y = max(ins.get_position().y1 for _, _, insets in group_axes
                  for ins in insets) + 0.02
    for group_title, axs, _ in group_axes:
        spans = [ax.get_position() for ax in axs]
        cx = (min(p.x0 for p in spans) + max(p.x1 for p in spans)) / 2
        fig.text(cx, min(title_y, 0.99), group_title, ha="center", va="bottom",
                 fontsize=12)

    total = max(len(frames) for *_, frames, _ in panels) + int(ANIM_FPS * ANIM_HOLD_S)

    def draw(step: int):
        for im, inset_im, counter, frames, cuts in panels:
            t = min(step, len(frames) - 1)
            im.set_data(frames[t])
            if inset_im is not None:
                inset_im.set_data(frames[t])
            ep = int(np.searchsorted(cuts, t, side="right"))
            counter.set_text(f"episode {ep + 1}/{len(cuts)}")

    anim = FuncAnimation(fig, draw, frames=total)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=ANIM_FPS), dpi=ANIM_DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dataset", default="pusht_expert_train.h5")
    p.add_argument("--randgoal-dataset", default="pusht_scripted_goal_train.lance")
    p.add_argument("--tag-size", type=int, default=5)
    p.add_argument("--tag-seed", type=int, default=3072)
    p.add_argument("--out", default="figures/f7_lewm_datasets.png")
    args = p.parse_args()

    base_ep0, base_ep1 = (_frames(args.baseline_dataset, ep, [0])[0] for ep in (0, 1))
    rand_ep0, rand_ep1 = (_frames(args.randgoal_dataset, ep, [0])[0] for ep in (0, 1))

    # Two frames of a baseline episode, tagged. Each tag mode gets its own
    # episode (distinct from the top row's) so no scene appears twice in the
    # figure. Stamping expects a (T, C, H, W) clip; go through torch to reuse
    # the exact training stamp with the episode's true index as its colour key.
    def tagged(mode: str, ep: int) -> list[np.ndarray]:
        f0, f1 = _frames(args.baseline_dataset, ep, [0, 60])
        clip = torch.from_numpy(np.stack([f0, f1])).permute(0, 3, 1, 2).contiguous()
        PixelTag(mode=mode, size=args.tag_size, seed=args.tag_seed).stamp(clip, ep, 0, 60)
        return [clip[t].permute(1, 2, 0).numpy() for t in range(2)]

    vid0, vid1 = tagged("video", ep=2)
    frm0, frm1 = tagged("frame", ep=3)

    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    panels = [
        (base_ep0, "fixed goal — episode 1"),
        (base_ep1, "fixed goal — episode 2"),
        (rand_ep0, "random goal — episode 1"),
        (rand_ep1, "random goal — episode 2"),
        (vid0, "per-episode colour — frame 1"),
        (vid1, "per-episode colour — frame 61"),
        (frm0, "per-frame colour — frame 1"),
        (frm1, "per-frame colour — frame 61"),
    ]
    for ax, (img, title) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    save_fig(fig, Path(args.out))

    def episode(dataset_name: str, ep: int) -> list[np.ndarray]:
        return _frames(dataset_name, ep, _episode_steps(dataset_name, ep))

    # tag a whole subsampled episode; frameskip=stride keeps each frame's
    # episode-local step, so frame-mode colours match what training would stamp
    def tagged_episode(mode: str, ep: int) -> list[np.ndarray]:
        frames = episode(args.baseline_dataset, ep)
        clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
        PixelTag(mode=mode, size=args.tag_size, seed=args.tag_seed).stamp(
            clip, ep, 0, ANIM_STRIDE
        )
        return [clip[t].permute(1, 2, 0).numpy() for t in range(clip.shape[0])]

    # left couple: three episodes per arm so the goal contrast plays out at the
    # cuts; right couple: the same three baseline episodes under both tag modes
    # (disjoint from the left couple's episodes), so only the square differs
    fig_dataset_animation(
        [
            (
                "the goal",
                [
                    ([episode(args.baseline_dataset, ep) for ep in (0, 1, 2)],
                     "fixed goal (original)", False),
                    ([episode(args.randgoal_dataset, ep) for ep in (0, 1, 2)],
                     "random goal", False),
                ],
            ),
            (
                "the corner square",
                [
                    ([tagged_episode("video", ep) for ep in (3, 4, 5)],
                     "per-episode colour", True),
                    ([tagged_episode("frame", ep) for ep in (3, 4, 5)],
                     "per-frame colour", True),
                ],
            ),
        ],
        Path(args.out).with_suffix(".gif"),
    )


if __name__ == "__main__":
    main()
