#!/usr/bin/env python3
"""Enigma-style feature-allocation visuals comparing JEPA with repair.

The experiment holds either physical content or the small RGB tag fixed and
changes the other factor.  It reports mean-centered embedding similarity and
contrasts smooth content and tag-only trajectories in each frozen encoder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


MODEL_ORDER = ("jepa", "bloop")
DISPLAY = {"jepa": "JEPA", "bloop": "Ours"}
COLORS = {"jepa": "#d55e00", "bloop": "#cc79a7"}


def path_length(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())


def normalized_path_metrics(
    content: np.ndarray, tag: np.ndarray, gallery: np.ndarray, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(gallery), size=min(4096, len(gallery) * 2))
    right = rng.integers(0, len(gallery), size=len(left))
    scale = float(np.median(np.linalg.norm(gallery[left] - gallery[right], axis=1)))
    scale = max(scale, 1e-12)
    content_length = path_length(content)
    tag_length = path_length(tag)
    return {
        "content_path_length_over_gallery_median": content_length / scale,
        "tag_path_length_over_gallery_median": tag_length / scale,
        "tag_to_content_path_length_ratio": tag_length / max(content_length, 1e-12),
        "content_tortuosity": content_length
        / max(float(np.linalg.norm(content[-1] - content[0])), 1e-12),
        "tag_tortuosity": tag_length
        / max(float(np.linalg.norm(tag[-1] - tag[0])), 1e-12),
    }


def batched_path_metrics(
    content: np.ndarray,
    tag: np.ndarray,
    gallery: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """Full-dimensional path allocation for a matched trajectory batch."""
    content = np.asarray(content, dtype=np.float64)
    tag = np.asarray(tag, dtype=np.float64)
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(gallery), size=min(4096, len(gallery) * 2))
    right = rng.integers(0, len(gallery), size=len(left))
    scale = float(np.median(np.linalg.norm(gallery[left] - gallery[right], axis=1)))
    scale = max(scale, 1e-12)
    content_length = np.linalg.norm(np.diff(content, axis=1), axis=2).sum(axis=1)
    tag_length = np.linalg.norm(np.diff(tag, axis=1), axis=2).sum(axis=1)
    return {
        "content_path_length_over_gallery_median": content_length / scale,
        "tag_path_length_over_gallery_median": tag_length / scale,
        "tag_to_content_path_length_ratio": tag_length
        / np.maximum(content_length, 1e-12),
    }


def summarize_paths(values: dict[str, np.ndarray], seed: int) -> dict:
    summary = {}
    for offset, (key, raw) in enumerate(values.items()):
        raw = np.asarray(raw, dtype=np.float64)
        if key == "tag_to_content_path_length_ratio":
            log_values = np.log(np.maximum(raw, 1e-12))
            stats = mean_ci(log_values, seed + offset)
            summary[key] = {
                "geometric_mean": float(np.exp(stats["mean"])),
                "geometric_mean_ci95": [float(np.exp(x)) for x in stats["ci95"]],
                "median": float(np.median(raw)),
                "fraction_below_one": float(np.mean(raw < 1.0)),
            }
        else:
            summary[key] = mean_ci(raw, seed + offset)
    return summary


def similarity_samples(passes):
    import torch

    reference = passes["own"]["projection"]
    center = reference.mean(dim=0)
    own = reference - center
    result = {}
    for name in ("same_content", "same_tag"):
        result[name] = torch.nn.functional.cosine_similarity(
            own, passes[name]["projection"] - center, dim=1
        ).cpu().numpy()
    return result


def mean_ci(values, seed: int, samples: int = 20_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)]}


def _rgb(frame):
    value = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    if value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    return value


def render_input_panel(out_dir, raw, tag_a, tag_b, tag_size):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    clean = raw.clone()
    reference = raw.clone()
    changed = raw.clone()
    reference[:, :tag_size, :tag_size] = torch.as_tensor(tag_a).view(3, 1, 1)
    changed[:, :tag_size, :tag_size] = torch.as_tensor(tag_b).view(3, 1, 1)
    images = (clean, reference, changed)
    titles = ("Physical content", "Reference predictable tag", "Counterfactual tag")
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.2), constrained_layout=True)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(_rgb(image))
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if title != "Physical content":
            axis.add_patch(
                plt.Rectangle(
                    (-0.5, -0.5),
                    tag_size,
                    tag_size,
                    fill=False,
                    ec="crimson",
                    lw=2.5,
                )
            )
            inset = axis.inset_axes([0.62, 0.62, 0.33, 0.33])
            inset.imshow(_rgb(image)[:tag_size, :tag_size], interpolation="nearest")
            inset.set_title("5×5 tag", fontsize=9)
            inset.set_xticks([])
            inset.set_yticks([])
    fig.suptitle("Matched intervention: physical scene fixed, only the RGB tag changes")
    fig.savefig(out_dir / "matched-tag-intervention.png", dpi=220)
    plt.close(fig)


def render_similarity(out_dir, similarity):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), sharey=True, constrained_layout=True)
    conditions = ("same_content", "same_tag")
    labels = ("Same content\ndifferent tag", "Different content\nsame tag")
    for axis, name in zip(axes, MODEL_ORDER):
        means = np.array(
            [similarity[name][condition]["mean"] for condition in conditions]
        )
        cis = np.array([similarity[name][condition]["ci95"] for condition in conditions])
        error = np.stack([means - cis[:, 0], cis[:, 1] - means])
        axis.bar([0, 1], means, color=["#0072b2", "#e69f00"], alpha=0.9)
        axis.errorbar([0, 1], means, yerr=error, fmt="none", ecolor="black", capsize=4)
        axis.axhline(0, color="0.35", lw=1)
        axis.set_xticks([0, 1], labels)
        axis.set_title(DISPLAY[name])
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean-centered cosine similarity")
    fig.suptitle("What organizes the representation: physical content or predictable tag?")
    fig.savefig(out_dir / "similarity-crossover.png", dpi=220)
    plt.close(fig)


def render_trajectory_figure(out_dir, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    for row, name in enumerate(MODEL_ORDER):
        gallery = records[name]["gallery_2d"]
        paths = (records[name]["content_2d"], records[name]["tag_2d"])
        titles = ("Physical-content sweep", "Tag-only sweep")
        for column, (path, title) in enumerate(zip(paths, titles)):
            axis = axes[row, column]
            axis.scatter(gallery[:, 0], gallery[:, 1], s=4, color="0.72", alpha=0.22)
            progress = np.linspace(0, 1, len(path))
            axis.plot(path[:, 0], path[:, 1], color=COLORS[name], lw=1.5, alpha=0.75)
            points = axis.scatter(
                path[:, 0],
                path[:, 1],
                c=progress,
                cmap="viridis",
                s=30,
                zorder=3,
            )
            axis.scatter(*path[0], color="#22aa44", s=85, zorder=4, label="start")
            axis.scatter(*path[-1], color="#dd2222", marker="*", s=130, zorder=4, label="end")
            axis.set_title(f"{DISPLAY[name]} — {title}")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal", adjustable="datalim")
            if column == 0:
                axis.set_ylabel(DISPLAY[name])
            if row == 0 and column == 1:
                axis.legend(frameon=False, loc="best")
    fig.colorbar(points, ax=axes, label="sweep progress", shrink=0.72)
    fig.suptitle("Smooth physical motion versus a smooth 5×5 RGB-tag change")
    fig.savefig(out_dir / "content-vs-tag-latent-trajectories.png", dpi=220)
    plt.close(fig)


def render_path_lengths(out_dir, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(2)
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    content = np.asarray([
        summaries[name]["content_path_length_over_gallery_median"]["mean"]
        for name in MODEL_ORDER
    ])
    tag = np.asarray([
        summaries[name]["tag_path_length_over_gallery_median"]["mean"]
        for name in MODEL_ORDER
    ])
    content_ci = np.array([
        summaries[name]["content_path_length_over_gallery_median"]["ci95"]
        for name in MODEL_ORDER
    ])
    tag_ci = np.array([
        summaries[name]["tag_path_length_over_gallery_median"]["ci95"]
        for name in MODEL_ORDER
    ])
    axis.bar(x - width / 2, content, width, label="Physical-content sweep", color="#0072b2")
    axis.bar(x + width / 2, tag, width, label="Tag-only sweep", color="#e69f00")
    axis.errorbar(
        x - width / 2,
        content,
        yerr=np.stack([content - content_ci[:, 0], content_ci[:, 1] - content]),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axis.errorbar(
        x + width / 2,
        tag,
        yerr=np.stack([tag - tag_ci[:, 0], tag_ci[:, 1] - tag]),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axis.set_xticks(x, [DISPLAY[name] for name in MODEL_ORDER])
    axis.set_ylabel("Latent path length / median gallery distance")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    axis.set_title("Which change consumes more representational distance?")
    fig.savefig(out_dir / "feature-allocation-path-length.png", dpi=220)
    plt.close(fig)


def render_ratio_distribution(out_dir, per_trajectory, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    rng = np.random.default_rng(947)
    x = np.arange(len(MODEL_ORDER), dtype=float)
    values = [
        np.asarray(
            per_trajectory[name]["tag_to_content_path_length_ratio"], dtype=float
        )
        for name in MODEL_ORDER
    ]
    count = min(map(len, values))
    if any(len(value) != count for value in values):
        raise ValueError("matched trajectory arrays have different lengths")
    # One shared horizontal offset per matched trajectory.  Both scatter
    # endpoints and the connecting segment use these exact coordinates, so
    # the visual pairing is faithful rather than merely index-aligned.
    pair_jitter = rng.uniform(-0.08, 0.08, size=count)
    for index in range(count):
        axis.plot(
            x + pair_jitter[index],
            [values[0][index], values[1][index]],
            color="0.80",
            lw=0.55,
            alpha=0.30,
        )
    for model_index, (name, raw) in enumerate(zip(MODEL_ORDER, values)):
        jitter = pair_jitter
        axis.scatter(
            np.full(len(raw), x[model_index]) + jitter,
            raw,
            s=13,
            alpha=0.42,
            color=COLORS[name],
            edgecolors="none",
        )
        stats = summaries[name]["tag_to_content_path_length_ratio"]
        mean = stats["geometric_mean"]
        low, high = stats["geometric_mean_ci95"]
        axis.errorbar(
            x[model_index],
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            color="black",
            markerfacecolor=COLORS[name],
            capsize=5,
            ms=7,
            zorder=5,
        )
    axis.axhline(
        1.0,
        color="0.25",
        ls="--",
        lw=1.2,
        label="parity threshold (tag / physical = 1)",
    )
    axis.set_yscale("log")
    axis.set_xticks(x, [DISPLAY[name] for name in MODEL_ORDER])
    axis.set_ylabel("Tag-only / physical-content latent path length")
    axis.set_title("Matched trajectory-level feature allocation")
    axis.grid(axis="y", alpha=0.2, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(out_dir / "feature-allocation-ratio-distribution.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--cache-dir", default=os.environ.get("LOCAL_DATASET_DIR"))
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--sweep-steps", type=int, default=32)
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tag-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")
    if not args.cache_dir:
        raise SystemExit("set LOCAL_DATASET_DIR or pass --cache-dir")

    import torch
    import stable_worldmodel as swm
    from sklearn.decomposition import PCA

    from additional_files.diagnose_frozen_representation import load_probe_model, parse_checkpoint
    from additional_files.diagnose_generic_tag_geometry import (
        build_loader,
        collect_frames,
        encode,
        make_interventions,
        normalized_colors,
        stamp,
    )
    from additional_files.pixel_tag import PixelTag
    from utils import get_img_preprocessor

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    preprocess = get_img_preprocessor("pixels", "pixels", img_size=224)
    loader = build_loader(
        args.dataset, args.cache_dir, args.frames, args.seed, args.batch_size
    )
    frames = collect_frames(loader)
    interventions = make_interventions(frames, args.seed, 0, args.tag_size, preprocess)

    # Choose one real trajectory with substantial physical displacement so the
    # content sweep is not an easy, nearly static example.
    trajectory_dataset = swm.data.load_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        num_steps=args.sweep_steps,
        frameskip=1,
        keys_to_cache=["state"],
    )
    trajectory_dataset.transform = None
    trajectory_ids = rng.choice(
        len(trajectory_dataset),
        size=min(args.trajectories, len(trajectory_dataset)),
        replace=False,
    )
    trajectory_records = []
    for index in trajectory_ids:
        sample = trajectory_dataset[int(index)]
        state = np.asarray(sample["state"])
        score = float(np.linalg.norm(np.diff(state[:, :4], axis=0), axis=1).sum())
        trajectory_records.append((score, int(index), sample))
    _, trajectory_index, trajectory_sample = max(trajectory_records, key=lambda row: row[0])
    raw_trajectory = torch.as_tensor(trajectory_sample["pixels"]).clone()

    tag = PixelTag(mode="video", size=args.tag_size, seed=0)
    colors_u8 = np.stack([tag.color_for(0), tag.color_for(args.frames + 17)])
    colors_norm = normalized_colors(preprocess, colors_u8)
    alpha = np.linspace(0, 1, args.sweep_steps)[:, None]
    smooth_u8 = np.rint((1 - alpha) * colors_u8[0] + alpha * colors_u8[1]).astype(np.uint8)
    smooth_norm = normalized_colors(preprocess, smooth_u8)
    content_sweeps, tag_sweeps = [], []
    for _, _, sample in trajectory_records:
        raw = torch.as_tensor(sample["pixels"]).clone()
        processed = preprocess({"pixels": raw})["pixels"]
        content_sweeps.append(
            stamp(
                processed,
                colors_norm[0].repeat(args.sweep_steps, 1),
                args.tag_size,
            )
        )
        fixed = processed[0:1].repeat(args.sweep_steps, 1, 1, 1)
        tag_sweeps.append(stamp(fixed, smooth_norm, args.tag_size))
    content_sweep = torch.cat(content_sweeps)
    tag_sweep = torch.cat(tag_sweeps)

    specs = [parse_checkpoint(value) for value in args.checkpoint]
    spec_by_name = {label: (run, filename) for label, run, filename in specs}
    missing = [name for name in MODEL_ORDER if name not in spec_by_name]
    if missing:
        raise ValueError(f"missing checkpoint labels: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    similarity, summaries, per_trajectory, records = {}, {}, {}, {}
    for model_index, name in enumerate(MODEL_ORDER):
        run, filename = spec_by_name[name]
        print(f"loading {name}: {run}/{filename}", flush=True)
        model = load_probe_model(f"{run}/{filename}").to(device).eval().requires_grad_(False)
        passes = {
            key: encode(model, value, device, args.batch_size)
            for key, value in interventions.items()
        }
        samples = similarity_samples(passes)
        similarity[name] = {
            condition: mean_ci(values, args.seed + model_index)
            for condition, values in samples.items()
        }
        gallery_embedding = passes["own"]["projection"].cpu().numpy()
        content_embedding_flat = encode(
            model, content_sweep, device, args.batch_size
        )["projection"].cpu().numpy()
        tag_embedding_flat = encode(
            model, tag_sweep, device, args.batch_size
        )["projection"].cpu().numpy()
        shape = (len(trajectory_records), args.sweep_steps, -1)
        content_embedding = content_embedding_flat.reshape(shape)
        tag_embedding = tag_embedding_flat.reshape(shape)
        raw_metrics = batched_path_metrics(
            content_embedding,
            tag_embedding,
            gallery_embedding,
            args.seed + model_index,
        )
        per_trajectory[name] = {key: value.tolist() for key, value in raw_metrics.items()}
        summaries[name] = summarize_paths(raw_metrics, args.seed + 100 * model_index)
        visual_position = next(
            i for i, (_, index, _) in enumerate(trajectory_records)
            if index == trajectory_index
        )
        pca = PCA(n_components=2).fit(gallery_embedding)
        records[name] = {
            "gallery_2d": pca.transform(gallery_embedding),
            "content_2d": pca.transform(content_embedding[visual_position]),
            "tag_2d": pca.transform(tag_embedding[visual_position]),
        }
        del model, passes
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    render_input_panel(
        args.out_dir, raw_trajectory[0], colors_u8[0], colors_u8[1], args.tag_size
    )
    render_similarity(args.out_dir, similarity)
    render_trajectory_figure(args.out_dir, records)
    render_path_lengths(args.out_dir, summaries)
    render_ratio_distribution(args.out_dir, per_trajectory, summaries)
    payload = {
        "protocol": {
            "dataset": args.dataset,
            "gallery_frames": len(frames),
            "sweep_steps": args.sweep_steps,
            "trajectories": len(trajectory_records),
            "trajectory_dataset_indices": [index for _, index, _ in trajectory_records],
            "trajectory_dataset_index": trajectory_index,
            "trajectory_selection": (
                "aggregate metrics use all seeded random trajectories; "
                "PCA illustration uses the largest physical path among them"
            ),
            "similarity": "mean-centered cosine on matched interventions",
            "path_normalization": "median pairwise distance in each model's gallery",
            "warning": "PCA is illustrative; path-length metrics are computed in full projector space",
            "seed": args.seed,
            "device": str(device),
        },
        "checkpoints": {
            name: f"{spec_by_name[name][0]}/{spec_by_name[name][1]}" for name in MODEL_ORDER
        },
        "tag_rgb_endpoints": colors_u8.tolist(),
        "similarity": similarity,
        "path_summary": summaries,
        "path_metrics_per_trajectory": per_trajectory,
    }
    (args.out_dir / "feature-allocation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    print(f"FEATURE_ALLOCATION_VISUALS_COMPLETE {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
