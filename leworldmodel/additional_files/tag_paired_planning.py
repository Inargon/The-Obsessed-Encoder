"""Frozen paired alignment audit on recorded Tagged PushT action banks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from additional_files.tag_candidate_audit import sample_episode_bank


TAG_MODES = ("context_tag", "goal_tag", "both_tags")
RANKS = (1, 4, 16)
CONDITIONS = {"reference": (0, 0), "context_tag": (1, 0), "goal_tag": (0, 1), "both_tags": (1, 1)}


@dataclass(frozen=True)
class LowRankAlignment:
    basis: np.ndarray
    source_center: np.ndarray
    shift: np.ndarray
    weights: np.ndarray

    def __call__(self, values):
        projected = values @ self.basis
        delta = self.shift + (projected - self.source_center) @ self.weights
        return values + delta @ self.basis.T


def fit_alignment(reference, shifted, rank, relative_ridge=0.1):
    """Fit a ridge-regularized residual map in the subspace of paired differences."""
    if (reference.shape != shifted.shape or reference.ndim != 2 or
            not 0 < rank < len(reference) or rank > reference.shape[1]):
        raise ValueError("Need matched matrices and sufficient anchor rows and dimensions")
    difference = reference - shifted
    _, _, directions = np.linalg.svd(difference, full_matrices=False)
    basis = directions[:rank].T
    source = shifted @ basis
    center = source.mean(axis=0)
    centered = source - center
    projected_delta = difference @ basis
    shift = projected_delta.mean(axis=0)
    gram = centered.T @ centered
    ridge = max(float(np.trace(gram) / rank) * relative_ridge, 1e-12)
    weights = np.linalg.solve(gram + ridge * np.eye(rank), centered.T @ (projected_delta - shift))
    return LowRankAlignment(basis, center, shift, weights)


def alignment_variants(reference, shifted):
    """Keep means fixed in the shuffled pairing control."""
    translation = (reference - shifted).mean(axis=0)
    ref_center, shifted_center = reference.mean(0), shifted.mean(0)
    ref_std, shifted_std = reference.std(0), shifted.std(0)
    ratios = np.divide(ref_std, shifted_std, out=np.ones_like(ref_std), where=shifted_std > 1e-8)
    return {
        "raw": lambda values: values,
        "mean_shift": lambda values: values + translation,
        "diagonal_moments": lambda values: ref_center + (values - shifted_center) * ratios,
        **{f"paired_rank_{rank}": fit_alignment(reference, shifted, rank) for rank in RANKS},
        "shuffled_rank_4": fit_alignment(reference, np.roll(shifted, 1, axis=0), 4),
    }


def measure_alignment(predictions, goals, reordered, recorded_index, maps):
    """Score each endpoint combination on the exact same candidate bank."""
    prediction_map, goal_map = maps
    corrected = (predictions[0], prediction_map(predictions[1]))
    target = (goals[0], goal_map(goals[1]))
    costs = {
        name: np.square(corrected[c] - target[g][:, None]).sum(-1)
        for name, (c, g) in CONDITIONS.items()
    }
    n, candidates = costs["reference"].shape
    if not all(np.isfinite(values).all() for values in costs.values()):
        raise ValueError("Nonfinite aligned candidate costs")
    rows = np.arange(n)
    reference = costs["reference"].argmin(axis=1)
    recorded = recorded_index[:, None] == np.arange(candidates)[None, :]
    strict_flips = {}
    recorded_rank = {}
    recorded_margin = {}
    for name, values in costs.items():
        minimum = values.min(axis=1)
        strict_flips[name] = minimum < values[rows, reference]
        expert = values[rows, recorded_index]
        recorded_rank[name] = 1 + (values < expert[:, None]).sum(axis=1)
        recorded_margin[name] = np.min(np.where(recorded, np.inf, values), axis=1) - expert
    changed_action_costs = np.square(prediction_map(reordered) - target[1][:, None]).sum(-1)
    action_order_winner = costs["both_tags"].argmin(axis=1)
    action_delta = changed_action_costs - costs["both_tags"]
    centered_delta = action_delta - action_delta.mean(axis=1, keepdims=True)
    reference_scale = costs["reference"].std(axis=1)
    action_response = np.sqrt(np.square(centered_delta).mean(axis=1))
    return {
        "reference_costs": costs["reference"],
        "strict_any_tag_flip": np.logical_or.reduce([strict_flips[name] for name in TAG_MODES]),
        "strict_tag_flips": {name: strict_flips[name] for name in TAG_MODES},
        "worst_recorded_rank": np.maximum.reduce(list(recorded_rank.values())),
        "recorded_rank": recorded_rank,
        "recorded_strictly_best_in_all_views": np.minimum.reduce(list(recorded_margin.values())) > 0,
        "recorded_margin": recorded_margin,
        "changed_tag_action_order_flip": changed_action_costs.min(1) < changed_action_costs[rows, action_order_winner],
        "changed_tag_relative_action_response": np.divide(
            action_response, reference_scale, out=np.zeros_like(action_response), where=reference_scale > 1e-12,
        ),
    }


def summarize_variants(predictions, goals, reordered, recorded_index, fitted, seed):
    rng = np.random.default_rng(seed)
    n = len(recorded_index)
    resample = rng.integers(n, size=(2000, n))
    results = {}
    baseline = None
    for name, maps in fitted.items():
        measured = measure_alignment(predictions, goals, reordered, recorded_index, maps)
        if baseline is None:
            if name != "raw":
                raise ValueError("Raw variant must be first")
            baseline = measured
        elif not np.array_equal(measured["reference_costs"], baseline["reference_costs"]):
            raise ValueError("An alignment modified the reference candidate costs")

        def paired(key):
            value = measured[key].astype(float)
            differences = value - baseline[key].astype(float)
            return {
                "mean": float(value.mean()),
                "difference_vs_raw": float(differences.mean()),
                "paired_episode_ci95": np.quantile(differences[resample].mean(1), [0.025, 0.975]).tolist(),
            }

        results[name] = {
            "strict_any_tag_flip": paired("strict_any_tag_flip"),
            "strict_tag_flip_counts": {key: int(value.sum()) for key, value in measured["strict_tag_flips"].items()},
            "worst_recorded_rank": paired("worst_recorded_rank"),
            "mean_recorded_rank": {key: float(value.mean()) for key, value in measured["recorded_rank"].items()},
            "median_recorded_margin": {key: float(np.median(value)) for key, value in measured["recorded_margin"].items()},
            "recorded_strictly_best_in_all_views": paired("recorded_strictly_best_in_all_views"),
            "changed_tag_action_order_flip": paired("changed_tag_action_order_flip"),
            "changed_tag_relative_action_response": paired("changed_tag_relative_action_response"),
            "per_episode": {
                key: measured[key].tolist()
                for key in ("strict_any_tag_flip", "worst_recorded_rank", "changed_tag_action_order_flip")
            },
        }
    return results


def native_cost_request(model, context, goal, actions, history):
    """Keep the single-request inference layout of the deployed planner."""
    n = actions.shape[1]
    info = {
        "pixels": context[:history][None, None].expand(1, n, -1, -1, -1, -1),
        "goal": goal[-1:][None, None].expand(1, n, -1, -1, -1, -1),
        "action": actions.clone(),
    }
    cost = model.get_cost(info, actions.clone())[0]
    return info, cost


def terminal_features(model, views, actions, history):
    """Read terminal features from the live planner call for each tag."""
    import torch

    predictions, goals = [], []
    for view in views:
        info, _ = native_cost_request(model, view, view, actions, history)
        predictions.append(info["predicted_emb"][0, :, -1])
        goals.append(info["goal_emb"][0, -1])
    return torch.stack(predictions), torch.stack(goals)


def check_cost_parity(model, views, actions, history, predictions, goals):
    import torch

    for context_tag, goal_tag in CONDITIONS.values():
        _, expected = native_cost_request(model, views[context_tag], views[goal_tag], actions, history)
        actual = (predictions[context_tag] - goals[goal_tag]).square().sum(-1)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def query_actions(recorded, bank, candidate_index, history):
    """Insert the recorded continuation and retain the query's observed history."""
    import torch

    plans = torch.cat((bank[:candidate_index], recorded[None], bank[candidate_index:]))
    plans = plans[:, :-1].clone()
    plans[:, :history - 1] = recorded[:history - 1]
    return plans.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--anchors", type=int, default=32)
    args = parser.parse_args()
    if args.out.exists() or args.anchors <= max(RANKS):
        parser.error("Use a new output and more than 16 calibration episodes")

    import torch
    import stable_worldmodel as swm
    from additional_files.evaluate_reacher_checkpoint import load_inference_model
    from additional_files.mechanism_metrics import normalize_action_blocks
    from utils import get_column_normalizer, get_img_preprocessor

    previous = json.loads(args.previous.read_text())
    protocol = previous["protocol"]
    history, horizon = protocol["history"], protocol["horizon"]
    frameskip, tag_size = protocol["frameskip"], protocol["tag_size"]
    num_steps = history + horizon
    queries, candidate_count = len(protocol["indices"]), protocol["candidates"]
    if queries < 2 or candidate_count < 3 or horizon < 2:
        raise ValueError("Need at least two queries, three candidates and two future blocks")
    rng = np.random.default_rng(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    dataset = swm.data.load_dataset(
        args.dataset, cache_dir=os.environ["LOCAL_DATASET_DIR"],
        num_steps=num_steps, frameskip=frameskip, keys_to_cache=["action"],
    )
    dataset.transform = None
    normalizer_source = dataset
    if args.dataset != "pusht_expert_train.h5":
        normalizer_source = swm.data.load_dataset(
            "pusht_expert_train.h5", cache_dir=os.environ["LOCAL_DATASET_DIR"],
            keys_to_load=["action"], keys_to_cache=["action"],
        )
    normalizer = get_column_normalizer(normalizer_source, "action", "action")
    action_dim = int(dataset.get_dim("action"))
    selected, bank = sample_episode_bank(
        dataset, rng, queries + args.anchors, candidate_count - 1, num_steps, frameskip, history,
    )
    calibration, query = selected[:args.anchors], selected[args.anchors:]
    recorded_indices = rng.integers(candidate_count, size=queries)
    palette = torch.as_tensor(rng.integers(256, size=(2, 3)), dtype=torch.uint8)
    if torch.equal(palette[0], palette[1]):
        palette[1, 0] = (int(palette[1, 0]) + 1) % 256
    preprocess = get_img_preprocessor("pixels", "pixels", img_size=224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def normalized(sample):
        result = normalize_action_blocks(
            torch.as_tensor(sample["action"]).clone(), normalizer, action_dim,
        ).reshape(num_steps, -1)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite normalized actions")
        return result

    bank_actions = torch.stack([normalized(dataset[item["clip"]]) for item in bank]).to(device)

    def collect(episode):
        sample = dataset[episode["clip"]]
        raw = torch.as_tensor(sample["pixels"])
        if raw.dtype != torch.uint8 or raw.ndim != 4 or raw.shape[:2] != (num_steps, 3):
            raise ValueError(f"Expected native uint8 TCHW pixels, got {raw.dtype} {raw.shape}")
        if tag_size > min(raw.shape[-2:]):
            raise ValueError("Tag exceeds native image size")
        frames = []
        for color in palette:
            tagged = raw.clone()
            tagged[:, :, :tag_size, :tag_size] = color.view(1, 3, 1, 1)
            frames.append(preprocess({"pixels": tagged})["pixels"])
        return torch.stack(frames), normalized(sample).to(device)

    report = {
        "protocol": {
            "dataset": args.dataset,
            "seed": protocol["seed"],
            "rgb": palette.tolist(),
            "calibration": calibration,
            "queries": query,
            "candidate_bank": bank,
            "recorded_candidate_indices": recorded_indices.tolist(),
            "history": history,
            "horizon": horizon,
            "frameskip": frameskip,
            "ranks": list(RANKS),
            "relative_ridge": 0.1,
            "paired_bootstrap": "query episode; conditional on fixed anchors, RGB pair and bank",
            "recorded_action_scope": "recorded continuation reaches the recorded goal frame; other candidates can also be valid",
            "action_response": "centered RMS change under future block reorder divided by unmodified reference cost std",
            "training_independent_episodes_verified": False,
        },
        "runs": {},
    }
    records = [collect(item) for item in calibration + query]
    checkpoint_root = Path(os.environ["STABLEWM_HOME"]) / "checkpoints"
    with torch.inference_mode():
        for name, old in previous["runs"].items():
            checkpoint = checkpoint_root / old["run_name"] / old["checkpoint"]
            digest = hashlib.sha256()
            with checkpoint.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1048576), b""):
                    digest.update(chunk)
            if digest.hexdigest() != old["sha256"]:
                raise ValueError(f"{name}: checkpoint SHA-256 changed")
            model = load_inference_model(old["run_name"], old["checkpoint"]).to(device).eval().requires_grad_(False)
            anchor_pred, anchor_goal = [], []
            for frames, recorded in records[:args.anchors]:
                pred, goal = terminal_features(model, frames.to(device), recorded[None, None, :-1], history)
                anchor_pred.append(pred[:, 0].float().cpu().numpy())
                anchor_goal.append(goal.float().cpu().numpy())
            pred_fit = alignment_variants(*np.stack(anchor_pred, axis=1))
            goal_fit = alignment_variants(*np.stack(anchor_goal, axis=1))
            predictions, goals, reordered = [], [], []
            for i, (frames, recorded) in enumerate(records[args.anchors:]):
                frames = frames.to(device)
                plans = query_actions(recorded, bank_actions, int(recorded_indices[i]), history)
                pred, goal = terminal_features(model, frames, plans, history)
                if i == 0:
                    check_cost_parity(model, frames, plans, history, pred, goal)
                order = plans.clone()
                order[:, :, history - 1:] = plans[:, :, history - 1:].roll(1, dims=2)
                action_info, _ = native_cost_request(model, frames[1], frames[1], order, history)
                predictions.append(pred.float().cpu().numpy())
                goals.append(goal.float().cpu().numpy())
                reordered.append(action_info["predicted_emb"][0, :, -1].float().cpu().numpy())
            report["runs"][name] = {
                "checkpoint_sha256": digest.hexdigest(),
                "raw_cost_parity_checked": True,
                "variants": summarize_variants(
                    np.stack(predictions, axis=1),
                    np.stack(goals, axis=1),
                    np.stack(reordered),
                    recorded_indices,
                    {key: (pred_fit[key], goal_fit[key]) for key in pred_fit},
                    protocol["seed"],
                ),
            }
            print(f"TAG_PAIRED_PLANNING_CHECKPOINT_COMPLETE {name}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"TAG_PAIRED_PLANNING_COMPLETE {args.out}", flush=True)


if __name__ == "__main__":
    main()
