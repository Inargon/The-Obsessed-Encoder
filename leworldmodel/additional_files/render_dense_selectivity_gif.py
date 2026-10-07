#!/usr/bin/env python3
"""Render compact JEPA-versus-Ours dense-intervention GIFs.

The animation cycles through matched diagnostic samples; it is deliberately
not presented as a rollout.  Every frame uses the same scene, goal, candidate
bank and spatial interventions for both models.  Heat is shown relative to
uniform spatial allocation, while tag share and absolute tag effect remain
visible so within-model normalization cannot exaggerate a small response.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


DISPLAY = {"context_cost": "context", "goal_cost": "goal"}


def relative_to_uniform(values: np.ndarray, floor: float = 0.25) -> np.ndarray:
    """Log2 sensitivity relative to a spatially uniform allocation."""
    heat = np.asarray(values, dtype=float)
    if heat.ndim != 2 or (heat < 0).any():
        raise ValueError("heatmap must be a nonnegative matrix")
    allocation = heat / max(float(heat.sum()), 1e-12)
    relative = allocation * allocation.size
    return np.clip(np.log2(np.maximum(relative, floor)), -2.0, 2.0)


def absolute_tag_values(run: dict, key: str) -> np.ndarray:
    share = np.asarray(run["per_clip"]["tag_region_mass"][key], dtype=float)
    total = np.asarray(run["per_clip"]["total_map_mass"][key], dtype=float)
    if share.shape != total.shape:
        raise ValueError(f"mismatched tag share and total mass for {key}")
    return share * total


def select_positions(data: dict, key: str, count: int) -> list[int]:
    """Choose strong, disclosed JEPA-minus-Ours allocation contrasts."""
    jepa = np.asarray(data["runs"]["jepa"]["per_clip"]["tag_region_mass"][key])
    ours = np.asarray(data["runs"]["ours"]["per_clip"]["tag_region_mass"][key])
    if jepa.shape != ours.shape or jepa.ndim != 1:
        raise ValueError("expected matched per-clip tag shares")
    order = np.argsort(-(jepa - ours), kind="stable")
    return [int(value) for value in order[: min(max(count, 1), len(order))]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.samples, args.fps, args.dpi) < 1:
        raise SystemExit("samples, fps and dpi must be positive")
    data = json.loads(args.input.read_text())
    if not all(label in data["runs"] for label in ("jepa", "ours")):
        raise ValueError("input must contain matched jepa and ours runs")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import ConnectionPatch, Rectangle
    import stable_worldmodel as swm
    import torch

    from additional_files.pixel_tag import PixelTag

    out_dir = args.out_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = data["protocol"]
    history = int(protocol["history"])
    horizon = int(protocol["horizon"])
    frameskip = int(protocol["frameskip"])
    tag_size = int(protocol["tag_size"])
    dataset = swm.data.load_dataset(
        protocol["dataset"],
        cache_dir=os.environ["LOCAL_DATASET_DIR"],
        num_steps=history + horizon,
        frameskip=frameskip,
        keys_to_cache=["action"],
    )
    dataset.transform = None
    tagger = PixelTag(mode="video", size=tag_size, seed=0)

    for key in ("context_cost", "goal_cost"):
        positions = select_positions(data, key, args.samples)
        raw_images: list[np.ndarray] = []
        jepa_maps: list[np.ndarray] = []
        ours_maps: list[np.ndarray] = []
        notes: list[tuple[float, float, float, float]] = []
        jepa_abs = absolute_tag_values(data["runs"]["jepa"], key)
        ours_abs = absolute_tag_values(data["runs"]["ours"], key)
        jepa_share = np.asarray(
            data["runs"]["jepa"]["per_clip"]["tag_region_mass"][key]
        )
        ours_share = np.asarray(
            data["runs"]["ours"]["per_clip"]["tag_region_mass"][key]
        )
        for position in positions:
            sample = dataset[int(protocol["indices"][position])]
            raw = torch.as_tensor(sample["pixels"]).clone()
            tagger.stamp(raw, ep_idx=position, start=0, frameskip=frameskip)
            time_index = history - 1 if key == "context_cost" else -1
            raw_images.append(raw[time_index].permute(1, 2, 0).numpy())
            jepa_maps.append(relative_to_uniform(
                np.asarray(data["runs"]["jepa"]["maps"]["mean"][key][position])
            ))
            ours_maps.append(relative_to_uniform(
                np.asarray(data["runs"]["ours"]["maps"]["mean"][key][position])
            ))
            notes.append((
                float(jepa_share[position]), float(jepa_abs[position]),
                float(ours_share[position]), float(ours_abs[position]),
            ))

        figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.75), constrained_layout=True)
        input_image = axes[0].imshow(raw_images[0])
        axes[0].add_patch(Rectangle(
            (0, 0), tag_size, tag_size, fill=False, edgecolor="#00C853", linewidth=2.2
        ))
        inset = axes[0].inset_axes([0.03, 0.72, 0.26, 0.26])
        inset_image = inset.imshow(raw_images[0])
        inset.set_xlim(-0.5, 13.5)
        inset.set_ylim(13.5, -0.5)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("#d62728")
            spine.set_linewidth(1.8)
        axes[0].add_artist(ConnectionPatch(
            xyA=(2.5, -0.5), coordsA=axes[0].transData,
            xyB=(0.5, 0.0), coordsB=inset.transAxes,
            color="#d62728", linewidth=1.1, clip_on=False,
        ))

        map_images = [
            axes[1].imshow(jepa_maps[0], cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest"),
            axes[2].imshow(ours_maps[0], cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest"),
        ]
        for axis in axes[1:]:
            axis.add_patch(Rectangle(
                (-0.5, -0.5), 1, 1, fill=False, edgecolor="#00C853", linewidth=2.2
            ))
        axes[0].set_title("Matched input", fontweight="bold")
        axes[1].set_title("JEPA", fontweight="bold")
        axes[2].set_title("Ours", fontweight="bold")
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        texts = [
            axes[1].text(0.5, -0.08, "", transform=axes[1].transAxes,
                         ha="center", va="top", fontsize=9),
            axes[2].text(0.5, -0.08, "", transform=axes[2].transAxes,
                         ha="center", va="top", fontsize=9),
        ]
        colorbar = figure.colorbar(
            ScalarMappable(norm=Normalize(-2, 2), cmap="RdBu_r"),
            ax=axes[1:].tolist(), orientation="horizontal", fraction=0.07, pad=0.16,
        )
        colorbar.set_ticks([-2, -1, 0, 1, 2])
        colorbar.set_ticklabels(["¼×", "½×", "1×", "2×", "4×"])
        colorbar.set_label("Spatial intervention sensitivity relative to uniform allocation")
        title = figure.suptitle("", fontsize=13)

        def draw(frame: int):
            input_image.set_data(raw_images[frame])
            inset_image.set_data(raw_images[frame])
            map_images[0].set_data(jepa_maps[frame])
            map_images[1].set_data(ours_maps[frame])
            js, ja, os_, oa = notes[frame]
            texts[0].set_text(f"tag share={100*js:.1f}%  abs={ja:.2f}")
            texts[1].set_text(f"tag share={100*os_:.1f}%  abs={oa:.2f}")
            ratio = ja / max(oa, 1e-12)
            title.set_text(
                f"Tagged PushT {DISPLAY[key]}-cost selectivity — matched sample {frame + 1}/{len(positions)}\n"
                f"largest disclosed JEPA−Ours tag-allocation gaps; Ours tag effect {ratio:.1f}× lower"
            )
            return [input_image, inset_image, *map_images, *texts, title]

        animation = FuncAnimation(figure, draw, frames=len(positions), interval=1000 / args.fps)
        output = out_dir / f"dense-{DISPLAY[key]}-selectivity.gif"
        animation.save(output, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
        plt.close(figure)
        print(f"DENSE_SELECTIVITY_GIF_COMPLETE {output}")


if __name__ == "__main__":
    main()
