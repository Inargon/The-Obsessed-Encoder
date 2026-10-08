"""Controlled RGB-tag sweeps for auditing apparent high-dimensional folding.

For each sampled physical frame, this script holds every pixel fixed except the
5x5 evaluation tag.  It then compares JEPA and Ours using absolute displacement,
global PCA dimension, local TwoNN dimension, and an independent random-projection
Gaussian-shape audit.  No model is trained and no simulator state is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .tag_folding_probe_metrics import summarize_tag_sweep
except ImportError:  # direct execution: python additional_files/tag_folding_probe.py
    from tag_folding_probe_metrics import summarize_tag_sweep


DEFAULT_CHECKPOINTS = {
    "jepa": "colored_square_episode_seed0/weights_epoch_10.pt",
    "ours": (
        "interface-cycle-20261002-064431_tagged_pusht_bloop_cycle_full_seed0/"
        "weights_epoch_10.pt"
    ),
}


def sample_rgb(count: int, seed: int) -> np.ndarray:
    """Return deterministic iid RGB samples in [0, 255].

    TwoNN assumes locally random sampling.  A regular colour lattice or a
    low-discrepancy sequence would give prettier coverage but bias that local
    intrinsic-dimension estimator.
    """

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(count, 3), dtype=np.uint8)


def stamp_tag(image: np.ndarray, rgb: np.ndarray, size: int = 5) -> np.ndarray:
    """Copy an HWC uint8 image and replace only its upper-left tag cell."""

    output = np.asarray(image).copy()
    if output.ndim != 3 or output.shape[-1] not in (3, 4):
        raise ValueError(f"expected HWC image, got {output.shape}")
    output[:size, :size, :3] = np.asarray(rgb, dtype=output.dtype)
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _extract_pixels(sample: dict[str, Any]) -> np.ndarray:
    pixels = np.asarray(sample["pixels"])
    if pixels.ndim == 4:
        pixels = pixels[0]
    if pixels.ndim != 3:
        raise ValueError(f"expected one native image, got {pixels.shape}")
    if pixels.shape[0] in (3, 4) and pixels.shape[-1] not in (3, 4):
        pixels = np.moveaxis(pixels, 0, -1)
    if np.issubdtype(pixels.dtype, np.floating) and float(pixels.max()) <= 1.0:
        pixels = np.rint(255.0 * pixels)
    return np.clip(pixels, 0, 255).astype(np.uint8)


def _load_dataset(dataset_name: str, cache_dir: str | None):
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        num_steps=1,
        frameskip=1,
        keys_to_cache=[],
    )
    dataset.transform = None
    return dataset


@torch.inference_mode()
def encode_sweep(
    model,
    base_image: np.ndarray,
    colors: np.ndarray,
    *,
    preprocess,
    device: str,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(colors), batch_size):
        selected = colors[start : start + batch_size]
        native = np.stack([stamp_tag(base_image, color) for color in selected])
        native = torch.from_numpy(np.moveaxis(native, -1, 1).copy())
        pixels = preprocess({"pixels": native})["pixels"].to(device)
        cls = model.encoder(
            pixels, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0].float()
        embeddings = model.projector(cls).float()
        chunks.append(embeddings.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _load_model(checkpoint: str, device: str):
    from additional_files.diagnose_frozen_representation import load_probe_model

    model = load_probe_model(checkpoint)
    model = model.to(device).eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    return model


def _aggregate_scene_metrics(scene_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_paths = {
        "centroid_rms": lambda row: row["absolute_tag_sensitivity"]["centroid_rms"],
        "pairwise_rms": lambda row: row["absolute_tag_sensitivity"]["pairwise_rms"],
        "global_participation_dimension": lambda row: row["global_participation_dimension"],
        "local_twonn_dimension": lambda row: row["local_twonn_dimension"],
        "folding_ratio": lambda row: row["folding_ratio"],
        "projection_quantile_mse": lambda row: row["heldout_projection_gaussian_gap"][
            "mean_quantile_mse"
        ],
    }
    output: dict[str, Any] = {}
    for name, getter in scalar_paths.items():
        values = np.asarray([getter(row) for row in scene_metrics], dtype=np.float64)
        values = values[np.isfinite(values)]
        output[name] = {
            "mean": float(values.mean()) if len(values) else float("nan"),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return output


def _pca2(embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * singular_values[:2]


def render_summary(
    output_path: Path,
    colors: np.ndarray,
    representative: dict[str, np.ndarray],
    aggregate: dict[str, dict[str, Any]],
) -> None:
    fig = plt.figure(figsize=(13.8, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.15, 0.85))
    model_colors = {"jepa": "#2f6fad", "ours": "#b1357a"}

    for column, model_name in enumerate(("jepa", "ours")):
        axis = fig.add_subplot(grid[0, column])
        points = _pca2(representative[model_name])
        display_colors = colors.astype(np.float64) / 255.0
        axis.scatter(points[:, 0], points[:, 1], c=display_colors, s=14, alpha=0.78, linewidths=0)
        axis.set_title("JEPA" if model_name == "jepa" else "Ours", fontweight="bold")
        axis.set_xlabel("tag-sweep PC1")
        axis.set_ylabel("tag-sweep PC2")
        axis.spines[["top", "right"]].set_visible(False)

    explanation = fig.add_subplot(grid[0, 2])
    explanation.axis("off")
    explanation.text(
        0.02,
        0.98,
        "Controlled intervention",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    explanation.text(
        0.02,
        0.82,
        "Physical pixels fixed\nOnly the 5×5 RGB tag changes\nNo retraining or simulator labels",
        va="top",
        fontsize=12,
        linespacing=1.5,
    )
    explanation.text(
        0.02,
        0.45,
        "Folding signature",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    explanation.text(
        0.02,
        0.32,
        "large global PCA dimension\n÷ small local TwoNN dimension",
        va="top",
        fontsize=12,
        linespacing=1.45,
    )

    metric_specs = [
        ("centroid_rms", "Absolute tag effect\n(latent RMS)"),
        ("global_participation_dimension", "Global effective\ndimension"),
        ("folding_ratio", "Folding ratio\n(global / local)"),
    ]
    for column, (metric, label) in enumerate(metric_specs):
        axis = fig.add_subplot(grid[1, column])
        names = ("jepa", "ours")
        means = [aggregate[name][metric]["mean"] for name in names]
        errors = [aggregate[name][metric]["std"] for name in names]
        axis.bar(
            [0, 1],
            means,
            yerr=errors,
            width=0.62,
            color=[model_colors[name] for name in names],
            capsize=4,
        )
        axis.set_xticks([0, 1], ["JEPA", "Ours"])
        axis.set_title(label, fontsize=12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)

    fig.suptitle(
        "Does a low-dimensional tag spread across the latent space?",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or os.environ.get("LOCAL_DATASET_DIR")
    dataset = _load_dataset(args.dataset, cache_dir)
    from utils import get_img_preprocessor

    preprocess = get_img_preprocessor("pixels", "pixels", img_size=224)
    rng = np.random.default_rng(args.seed)
    if args.dataset_indices:
        indices = np.asarray(args.dataset_indices, dtype=np.int64)
    else:
        indices = np.sort(rng.choice(len(dataset), size=args.scenes, replace=False))
    colors = sample_rgb(args.colors, seed=args.seed)

    models = {
        "jepa": _load_model(args.jepa_checkpoint, args.device),
        "ours": _load_model(args.ours_checkpoint, args.device),
    }
    raw_embeddings: dict[str, list[np.ndarray]] = {"jepa": [], "ours": []}
    metrics: dict[str, list[dict[str, Any]]] = {"jepa": [], "ours": []}
    for scene_number, dataset_index in enumerate(indices, start=1):
        sample = dataset[int(dataset_index)]
        image = _extract_pixels(sample)
        for model_name, model in models.items():
            embeddings = encode_sweep(
                model,
                image,
                colors,
                preprocess=preprocess,
                device=args.device,
                batch_size=args.batch_size,
            )
            raw_embeddings[model_name].append(embeddings)
            metrics[model_name].append(
                summarize_tag_sweep(
                    embeddings,
                    projections=args.projections,
                    seed=args.seed + 1009 * scene_number,
                )
            )
        print(f"tag folding probe: {scene_number}/{len(indices)} scenes", flush=True)

    aggregate = {
        model_name: _aggregate_scene_metrics(model_metrics)
        for model_name, model_metrics in metrics.items()
    }
    representative_index = len(indices) // 2
    representative = {
        model_name: model_embeddings[representative_index]
        for model_name, model_embeddings in raw_embeddings.items()
    }
    render_summary(
        output_dir / "tag-folding-probe-summary.png",
        colors,
        representative,
        aggregate,
    )
    np.savez_compressed(
        output_dir / "tag-folding-probe-embeddings.npz",
        colors=colors,
        dataset_indices=indices,
        jepa=np.stack(raw_embeddings["jepa"]),
        ours=np.stack(raw_embeddings["ours"]),
    )
    result = {
        "protocol": {
            "dataset": args.dataset,
            "dataset_indices": indices.tolist(),
            "scenes": int(len(indices)),
            "colors_per_scene": int(len(colors)),
            "tag_size": 5,
            "intervention": "all physical pixels fixed; only upper-left RGB tag changed",
            "local_dimension": "TwoNN, estimated independently within each scene",
            "global_dimension": "PCA covariance participation ratio within each scene",
            "folding_ratio": "global participation dimension / local TwoNN dimension",
            "gaussian_audit": (
                "held-out random-projection standardized quantile MSE; not the training SIGReg"
            ),
            "scope_warning": (
                "comparative empirical audit; a high ratio supports but does not prove manifold folding"
            ),
            "seed": args.seed,
            "device": args.device,
        },
        "checkpoints": {
            "jepa": args.jepa_checkpoint,
            "ours": args.ours_checkpoint,
        },
        "runs": {
            name: {"scenes": metrics[name], "aggregate": aggregate[name]}
            for name in ("jepa", "ours")
        },
    }
    (output_dir / "tag-folding-probe.json").write_text(
        json.dumps(_json_safe(result), indent=2),
        encoding="utf-8",
    )
    print(f"TAG_FOLDING_PROBE_COMPLETE {output_dir}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--cache-dir")
    parser.add_argument("--jepa-checkpoint", default=DEFAULT_CHECKPOINTS["jepa"])
    parser.add_argument("--ours-checkpoint", default=DEFAULT_CHECKPOINTS["ours"])
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--colors", type=int, default=512)
    parser.add_argument("--dataset-indices", type=int, nargs="*")
    parser.add_argument("--projections", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
