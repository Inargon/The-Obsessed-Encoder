#!/usr/bin/env python3
"""Dense matched intervention maps for tagged-PushT representations/planners.

The diagnostic asks which image regions affect (1) the planner-facing latent
and (2) fixed-candidate planning costs.  It is deliberately interventional:
the same clips, goals, and candidate action banks are used for every model.
Each spatial cell is replaced either by a locally blurred version or by the
same cell from a matched real donor clip.  Agreement across these baselines is
reported as the primary map.  This is an offline sensitivity diagnostic, not
a closed-loop success-rate evaluation or a semantic causal certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


MAP_KEYS = (
    "context_latent",
    "goal_latent",
    "context_cost",
    "goal_cost",
    "context_rank",
    "goal_rank",
)


def grid_windows(height: int, width: int, grid_size: int) -> list[tuple[int, int, int, int]]:
    """Return a complete non-overlapping row-major spatial partition."""
    if min(height, width, grid_size) < 1:
        raise ValueError("height, width, and grid_size must be positive")
    if grid_size > min(height, width):
        raise ValueError("grid_size cannot exceed the shorter image dimension")
    ys = np.linspace(0, height, grid_size + 1, dtype=int)
    xs = np.linspace(0, width, grid_size + 1, dtype=int)
    return [
        (int(ys[row]), int(ys[row + 1]), int(xs[col]), int(xs[col + 1]))
        for row in range(grid_size)
        for col in range(grid_size)
    ]


def normalized_cost_metrics(reference: np.ndarray, changed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized RMSE, pair-order reversal, and selection change."""
    ref = np.asarray(reference, dtype=float)
    alt = np.asarray(changed, dtype=float)
    if ref.ndim == 1:
        ref = ref[None]
    if alt.ndim == 1:
        alt = alt[None]
    if ref.shape[-1] != alt.shape[-1] or ref.shape[-1] < 2:
        raise ValueError("reference and changed costs need matched candidate axes")
    if ref.shape[0] == 1 and alt.shape[0] != 1:
        ref = np.broadcast_to(ref, alt.shape)
    if ref.shape != alt.shape or not np.isfinite(ref).all() or not np.isfinite(alt).all():
        raise ValueError("cost arrays must be matched and finite")
    rmse = np.sqrt(np.mean(np.square(ref - alt), axis=1))
    normalized = rmse / np.maximum(ref.std(axis=1), 1e-12)
    left, right = np.triu_indices(ref.shape[1], 1)
    ref_delta = ref[:, left] - ref[:, right]
    alt_delta = alt[:, left] - alt[:, right]
    informative = (ref_delta != 0) & (alt_delta != 0)
    denominator = np.maximum(informative.sum(axis=1), 1)
    reversal = ((ref_delta * alt_delta < 0) & informative).sum(axis=1) / denominator
    selection = (ref.argmin(axis=1) != alt.argmin(axis=1)).astype(float)
    return normalized, reversal, selection


def tag_region_mass(values: np.ndarray, windows: list[tuple[int, int, int, int]], tag_size: int) -> float:
    """Fraction of nonnegative map mass assigned to cells touching the tag."""
    scores = np.asarray(values, dtype=float).reshape(-1)
    if len(scores) != len(windows) or tag_size < 1 or (scores < 0).any():
        raise ValueError("invalid map, windows, or tag size")
    total = scores.sum()
    if total <= 1e-12:
        return 0.0
    touches = np.asarray(
        [y0 < tag_size and x0 < tag_size for y0, _y1, x0, _x1 in windows]
    )
    return float(scores[touches].sum() / total)


def spatial_entropy(values: np.ndarray) -> float:
    scores = np.asarray(values, dtype=float).reshape(-1)
    total = scores.sum()
    if total <= 1e-12:
        return 0.0
    probability = scores / total
    probability = probability[probability > 0]
    return float(-(probability * np.log(probability)).sum() / np.log(len(scores)))


def nonnegative_cosine(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float).reshape(-1)
    second = np.asarray(right, dtype=float).reshape(-1)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return float(np.array_equal(first, second))
    return float(np.dot(first, second) / denominator)


def unit_mass(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    return scores / max(float(scores.sum()), 1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=RUN/FILE.pt")
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--num-clips", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--tag-size", type=int, default=5)
    parser.add_argument("--patch-batch-size", type=int, default=4)
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()
    if min(
        args.num_clips,
        args.candidates,
        args.history,
        args.horizon,
        args.frameskip,
        args.grid_size,
        args.tag_size,
        args.patch_batch_size,
    ) < 1:
        parser.error("all count and spatial arguments must be positive")
    if args.candidates < 2:
        parser.error("at least two candidates are required")
    return args


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import torch
    import torch.nn.functional as F
    import stable_worldmodel as swm

    from additional_files.diagnose_frozen_representation import parse_checkpoint
    from additional_files.diagnose_tag_intervention import choose_finite_action_indices
    from additional_files.evaluate_reacher_checkpoint import load_inference_model
    from additional_files.mechanism_metrics import normalize_action_blocks, summarize
    from additional_files.pixel_tag import PixelTag
    from utils import get_column_normalizer, get_img_preprocessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dataset = swm.data.load_dataset(
        args.dataset,
        cache_dir=os.environ["LOCAL_DATASET_DIR"],
        num_steps=args.history + args.horizon,
        frameskip=args.frameskip,
        keys_to_cache=["action"],
    )
    dataset.transform = None
    normalizer = get_column_normalizer(dataset, "action", "action")
    native_action_dim = int(dataset.get_dim("action"))
    preprocess = get_img_preprocessor("pixels", "pixels", img_size=224)
    selected = choose_finite_action_indices(
        dataset,
        rng.permutation(len(dataset)),
        min(args.num_clips, len(dataset)),
    )
    ids = np.asarray(selected, dtype=int)
    partners = np.roll(np.arange(len(ids)), 1)
    bank_ids = rng.integers(len(ids), size=(len(ids), args.candidates))
    tagger = PixelTag(mode="video", size=args.tag_size, seed=0)

    raw_clips: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    for position, index in enumerate(ids):
        sample = dataset[int(index)]
        raw = torch.as_tensor(sample["pixels"]).clone()
        if raw.dtype != torch.uint8 or raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError(f"Expected native uint8 TCHW input, got {raw.dtype}/{raw.shape}")
        if raw.shape[0] != args.history + args.horizon:
            raise ValueError("unexpected clip length")
        tagger.stamp(raw, ep_idx=position, start=0, frameskip=args.frameskip)
        raw_clips.append(raw)
        action = torch.as_tensor(sample["action"]).clone()
        if not torch.isfinite(action).all():
            raise AssertionError("finite-action sampler returned invalid actions")
        action = normalize_action_blocks(action, normalizer, native_action_dim)
        actions.append(action.reshape(args.history + args.horizon, -1))
    action_tensor = torch.stack(actions)
    height, width = map(int, raw_clips[0].shape[-2:])
    windows = grid_windows(height, width, args.grid_size)

    def preprocess_batch(raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 5:
            raise ValueError("expected BTCHW raw batch")
        batch, time = raw.shape[:2]
        flat = raw.reshape(batch * time, *raw.shape[2:])
        cooked = preprocess({"pixels": flat})["pixels"]
        return cooked.reshape(batch, time, *cooked.shape[1:]).to(device)

    def variants(raw: torch.Tensor, donor: torch.Tensor, subset) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(subset)
        blurred_source = F.avg_pool2d(
            raw.float(), kernel_size=7, stride=1, padding=3
        ).round().clamp(0, 255).to(torch.uint8)
        blur = raw.unsqueeze(0).repeat(count, 1, 1, 1, 1)
        matched = blur.clone()
        for row, window_index in enumerate(subset):
            y0, y1, x0, x1 = windows[window_index]
            blur[row, :, :, y0:y1, x0:x1] = blurred_source[:, :, y0:y1, x0:x1]
            matched[row, :, :, y0:y1, x0:x1] = donor[:, :, y0:y1, x0:x1]
        return blur, matched

    def encode_projection(model, frames: torch.Tensor) -> torch.Tensor:
        cls = model.encoder(
            frames, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0].float()
        return model.projector(cls).float()

    def plan_costs(model, context: torch.Tensor, goal: torch.Tensor, plans: torch.Tensor) -> torch.Tensor:
        # context: B,T,C,H,W; goal: B,1,C,H,W; plans: B,K,L,A
        batch = context.shape[0]
        candidates = plans.shape[1]
        info = {
            "pixels": context[:, None].expand(-1, candidates, -1, -1, -1, -1),
            "goal": goal[:, None].expand(-1, candidates, -1, -1, -1, -1),
            "action": plans,
        }
        return model.get_cost(info, plans.clone()).reshape(batch, candidates)

    checkpoint_root = Path(os.environ["STABLEWM_HOME"]) / "checkpoints"
    result: dict = {
        "protocol": {
            **vars(args),
            "out_dir": str(args.out_dir),
            "diagnostic": "offline dense matched intervention sensitivity; NOT closed-loop SR",
            "baselines": ["local_blur", "matched_real_donor_patch"],
            "primary_map": "arithmetic mean of the two intervention sensitivities",
            "context_intervention": "same cell replaced in all context frames; goal and candidates fixed",
            "goal_intervention": "same cell replaced in goal frame; context and candidates fixed",
            "scope_warning": "spatial sensitivity, not semantic causal identification",
            "indices": ids.tolist(),
            "candidate_source_indices": ids[bank_ids].tolist(),
            "partner_indices": ids[partners].tolist(),
            "tag_colors": [tagger.color_for(position).tolist() for position in range(len(ids))],
            "native_image_shape": [height, width],
            "windows": windows,
            "device": str(device),
        },
        "runs": {},
    }

    with torch.no_grad():
        for checkpoint_spec in args.checkpoint:
            label, run_name, filename = parse_checkpoint(checkpoint_spec)
            checkpoint = checkpoint_root / run_name / filename
            digest = hashlib.sha256()
            with checkpoint.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1_048_576), b""):
                    digest.update(chunk)
            model = load_inference_model(run_name, filename).to(device).eval().requires_grad_(False)
            maps = {baseline: {key: [] for key in MAP_KEYS} for baseline in ("blur", "donor")}
            maps["mean"] = {key: [] for key in MAP_KEYS}
            tag_masses = {key: [] for key in MAP_KEYS}
            entropies = {key: [] for key in MAP_KEYS}
            total_masses = {key: [] for key in MAP_KEYS}
            baseline_agreement = {key: [] for key in MAP_KEYS}

            for clip_index, raw in enumerate(raw_clips):
                donor = raw_clips[int(partners[clip_index])]
                own = preprocess_batch(raw[None])[0]
                plans = action_tensor[bank_ids[clip_index], : args.history + args.horizon - 1].clone()
                plans[:, : args.history - 1] = action_tensor[clip_index, : args.history - 1]
                plans = plans[None].to(device)
                reference_cost = plan_costs(
                    model,
                    own[: args.history][None],
                    own[-1:][None],
                    plans,
                ).cpu().numpy()
                reference_context_latent = encode_projection(
                    model, own[args.history - 1 : args.history]
                )
                reference_goal_latent = encode_projection(model, own[-1:])

                clip_maps = {
                    baseline: {key: np.zeros(len(windows), dtype=float) for key in MAP_KEYS}
                    for baseline in ("blur", "donor")
                }
                for start in range(0, len(windows), args.patch_batch_size):
                    subset = list(range(start, min(start + args.patch_batch_size, len(windows))))
                    raw_blur, raw_donor = variants(raw, donor, subset)
                    for baseline, raw_variants in (("blur", raw_blur), ("donor", raw_donor)):
                        cooked = preprocess_batch(raw_variants)
                        batch = len(subset)
                        context_latent = encode_projection(
                            model, cooked[:, args.history - 1]
                        )
                        goal_latent = encode_projection(model, cooked[:, -1])
                        context_distance = torch.linalg.vector_norm(
                            context_latent - reference_context_latent, dim=1
                        ) / torch.linalg.vector_norm(reference_context_latent, dim=1).clamp_min(1e-12)
                        goal_distance = torch.linalg.vector_norm(
                            goal_latent - reference_goal_latent, dim=1
                        ) / torch.linalg.vector_norm(reference_goal_latent, dim=1).clamp_min(1e-12)
                        expanded_plans = plans.expand(batch, -1, -1, -1)
                        context_cost = plan_costs(
                            model,
                            cooked[:, : args.history],
                            own[-1:][None].expand(batch, -1, -1, -1, -1),
                            expanded_plans,
                        ).cpu().numpy()
                        goal_cost = plan_costs(
                            model,
                            own[: args.history][None].expand(batch, -1, -1, -1, -1),
                            cooked[:, -1:],
                            expanded_plans,
                        ).cpu().numpy()
                        context_rmse, context_rank, _ = normalized_cost_metrics(reference_cost, context_cost)
                        goal_rmse, goal_rank, _ = normalized_cost_metrics(reference_cost, goal_cost)
                        values = {
                            "context_latent": context_distance.cpu().numpy(),
                            "goal_latent": goal_distance.cpu().numpy(),
                            "context_cost": context_rmse,
                            "goal_cost": goal_rmse,
                            "context_rank": context_rank,
                            "goal_rank": goal_rank,
                        }
                        for key, value in values.items():
                            clip_maps[baseline][key][subset] = value

                for key in MAP_KEYS:
                    mean_map = 0.5 * (
                        clip_maps["blur"][key] + clip_maps["donor"][key]
                    )
                    for baseline in ("blur", "donor"):
                        maps[baseline][key].append(
                            clip_maps[baseline][key].reshape(args.grid_size, args.grid_size).tolist()
                        )
                    maps["mean"][key].append(
                        mean_map.reshape(args.grid_size, args.grid_size).tolist()
                    )
                    tag_masses[key].append(tag_region_mass(mean_map, windows, args.tag_size))
                    entropies[key].append(spatial_entropy(mean_map))
                    total_masses[key].append(float(mean_map.sum()))
                    baseline_agreement[key].append(
                        nonnegative_cosine(
                            clip_maps["blur"][key], clip_maps["donor"][key]
                        )
                    )
                if (clip_index + 1) % 4 == 0:
                    print(f"{label}: {clip_index + 1}/{len(raw_clips)} clips", flush=True)

            result["runs"][label] = {
                "run_name": run_name,
                "checkpoint": filename,
                "sha256": digest.hexdigest(),
                "maps": maps,
                "per_clip": {
                    "tag_region_mass": tag_masses,
                    "spatial_entropy": entropies,
                    "total_map_mass": total_masses,
                    "blur_donor_cosine": baseline_agreement,
                },
                "summary": {
                    "tag_region_mass": {
                        key: summarize(values, args.seed) for key, values in tag_masses.items()
                    },
                    "spatial_entropy": {
                        key: summarize(values, args.seed) for key, values in entropies.items()
                    },
                    "total_map_mass": {
                        key: summarize(values, args.seed) for key, values in total_masses.items()
                    },
                    "blur_donor_cosine": {
                        key: summarize(values, args.seed)
                        for key, values in baseline_agreement.items()
                    },
                },
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    output_json = args.out_dir / "dense-planner-selectivity.json"
    output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    labels = list(result["runs"])
    display = {"jepa": "JEPA", "ours": "Ours"}
    map_titles = {
        "context_latent": "Context latent sensitivity",
        "context_cost": "Context planning-cost sensitivity",
        "goal_cost": "Goal planning-cost sensitivity",
    }
    # Aggregate maps expose systematic spatial allocation without cherry-picking.
    figure, axes = plt.subplots(len(labels), 3, figsize=(10.5, 3.1 * len(labels)), squeeze=False)
    for row, label in enumerate(labels):
        for col, key in enumerate(map_titles):
            stack = np.asarray(result["runs"][label]["maps"]["mean"][key])
            aggregate = np.stack([unit_mass(item) for item in stack]).mean(axis=0)
            axis = axes[row, col]
            image = axis.imshow(aggregate, cmap="magma", interpolation="nearest")
            axis.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="#00C853", linewidth=2))
            axis.set_title(map_titles[key])
            axis.set_ylabel(display.get(label, label))
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle("Dense matched intervention sensitivity (green = tag cell)", fontsize=15)
    figure.tight_layout()
    figure.savefig(args.out_dir / "dense-selectivity-aggregate.png", dpi=220)
    plt.close(figure)

    # Paired tag-mass summary, which is the quantitative counterpart of the maps.
    keys = ("context_latent", "context_cost", "goal_cost")
    x = np.arange(len(keys), dtype=float)
    width_bar = 0.34
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = {"jepa": "#E69F63", "ours": "#CC79A7"}
    for index, label in enumerate(labels):
        means = [result["runs"][label]["summary"]["tag_region_mass"][key]["mean"] for key in keys]
        intervals = [
            result["runs"][label]["summary"]["tag_region_mass"][key]["bootstrap_clip_ci95"]
            for key in keys
        ]
        errors = np.asarray([[mean - low for mean, (low, _high) in zip(means, intervals)],
                             [high - mean for mean, (_low, high) in zip(means, intervals)]])
        offset = (index - (len(labels) - 1) / 2) * width_bar
        axis.bar(
            x + offset,
            means,
            width=width_bar,
            yerr=errors,
            capsize=3,
            label=display.get(label, label),
            color=colors.get(label),
            alpha=0.9,
        )
    axis.set_xticks(x, ["Latent/context", "Cost/context", "Cost/goal"])
    axis.set_ylabel("Fraction of spatial sensitivity in tag cell")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.out_dir / "dense-tag-attribution-mass.png", dpi=220)
    plt.close(figure)

    # Select examples by a predeclared quantitative gap, not visual appearance.
    if "jepa" in result["runs"] and "ours" in result["runs"]:
        jepa_mass = np.asarray(result["runs"]["jepa"]["per_clip"]["tag_region_mass"]["context_cost"])
        ours_mass = np.asarray(result["runs"]["ours"]["per_clip"]["tag_region_mass"]["context_cost"])
        chosen = np.argsort(jepa_mass - ours_mass)[::-1][: min(args.examples, len(ids))]
    else:
        chosen = np.arange(min(args.examples, len(ids)))
    result["protocol"]["selected_example_positions"] = chosen.tolist()

    for rank, clip_index in enumerate(chosen, start=1):
        figure, axes = plt.subplots(len(labels), 4, figsize=(13.0, 3.0 * len(labels)), squeeze=False)
        context_raw = raw_clips[int(clip_index)][args.history - 1].permute(1, 2, 0).numpy()
        goal_raw = raw_clips[int(clip_index)][-1].permute(1, 2, 0).numpy()
        for row, label in enumerate(labels):
            axes[row, 0].imshow(context_raw)
            axes[row, 0].add_patch(Rectangle((0, 0), args.tag_size, args.tag_size, fill=False,
                                             edgecolor="#00C853", linewidth=2))
            axes[row, 0].set_title("Input context")
            for col, (key, base) in enumerate(
                (("context_latent", context_raw), ("context_cost", context_raw), ("goal_cost", goal_raw)),
                start=1,
            ):
                axis = axes[row, col]
                axis.imshow(base)
                heat = unit_mass(
                    np.asarray(result["runs"][label]["maps"]["mean"][key][int(clip_index)])
                )
                axis.imshow(
                    heat,
                    cmap="magma",
                    alpha=0.62,
                    interpolation="bilinear",
                    extent=(0, width, height, 0),
                )
                axis.add_patch(Rectangle((0, 0), args.tag_size, args.tag_size, fill=False,
                                         edgecolor="#00C853", linewidth=2))
                axis.set_title(map_titles[key])
            axes[row, 0].set_ylabel(display.get(label, label), fontsize=12, fontweight="bold")
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle(
            f"Matched dense selectivity — clip position {int(clip_index)}",
            fontsize=15,
        )
        figure.tight_layout()
        figure.savefig(args.out_dir / f"dense-example-{rank:02d}.png", dpi=220)
        plt.close(figure)

    # Rewrite after recording the deterministic example selection.
    output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"DENSE_PLANNER_SELECTIVITY_COMPLETE {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
