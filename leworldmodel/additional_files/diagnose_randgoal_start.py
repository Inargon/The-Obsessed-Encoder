#!/usr/bin/env python3
"""Probe whether a frozen RandGoal encoder retains decision-relevant goals.

Only the first clip of each episode is used.  RandGoal starts the pusher and
block from (approximately) the same physical configuration in every episode,
so decoding the randomized destination or its initial expert action sequence
cannot be explained by later trajectory state alone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from additional_files.diagnose_frozen_representation import load_probe_model
from utils import get_img_preprocessor


def parse_checkpoint(value: str) -> tuple[str, str]:
    try:
        label, location = value.split("=", 1)
        run_name, filename = location.rsplit("/", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "checkpoint must be LABEL=RUN/weights_step_N.pt"
        ) from exc
    if not label or not run_name or not filename.endswith(".pt"):
        raise argparse.ArgumentTypeError(
            "checkpoint must be LABEL=RUN/weights_step_N.pt"
        )
    return label, f"{run_name}/{filename}"


def pose_target(pose: np.ndarray) -> np.ndarray:
    """Represent an x/y/angle pose without an angular discontinuity."""
    return np.column_stack(
        (pose[:, 0], pose[:, 1], np.sin(pose[:, 2]), np.cos(pose[:, 2]))
    ).astype(np.float32)


def relative_target(state: np.ndarray, goal_pose: np.ndarray) -> np.ndarray:
    """Initial block-to-goal displacement and relative orientation."""
    delta_angle = goal_pose[:, 2] - state[:, 4]
    return np.column_stack(
        (
            goal_pose[:, 0] - state[:, 2],
            goal_pose[:, 1] - state[:, 3],
            np.sin(delta_angle),
            np.cos(delta_angle),
        )
    ).astype(np.float32)


def probe_r2(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    alpha: float,
) -> float:
    probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    probe.fit(x[train], y[train])
    return float(
        r2_score(y[test], probe.predict(x[test]), multioutput="uniform_average")
    )


def build_loader(
    dataset_name: str,
    cache_dir: str | None,
    num_episodes: int,
    horizon: int,
    seed: int,
    batch_size: int,
):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        num_steps=horizon + 1,
        frameskip=1,
        keys_to_cache=["state", "action", "goal_pose"],
    )
    dataset.transform = get_img_preprocessor(
        "pixels", "pixels", img_size=224
    )
    step_idx = np.asarray(dataset.get_col_data("step_idx"))
    starts = np.flatnonzero(step_idx == 0)
    # Dataset indices correspond to starting rows.  Exclude any defensive
    # out-of-range entries should a backend trim invalid final windows.
    starts = starts[starts < len(dataset)]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        starts, size=min(num_episodes, len(starts)), replace=False
    )
    subset = torch.utils.data.Subset(dataset, np.sort(chosen).tolist())
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0
    )


@torch.no_grad()
def extract(model, loader, device: torch.device, horizon: int):
    values: dict[str, list[np.ndarray]] = {
        "backbone": [],
        "projection": [],
        "state": [],
        "goal": [],
        "actions": [],
    }
    for batch in loader:
        pixels = batch["pixels"]
        first = pixels[:, 0].to(device)
        cls = model.encoder(
            first, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0].float()
        projection = model.projector(cls).float()

        state = batch["state"]
        goal = batch["goal_pose"]
        actions = batch["action"].float()
        if state.ndim == 3:
            state = state[:, 0]
        if goal.ndim == 3:
            goal = goal[:, 0]
        if actions.ndim == 2:
            actions = actions[:, None, :]
        actions = actions[:, :horizon].reshape(actions.size(0), -1)

        values["backbone"].append(cls.cpu().numpy())
        values["projection"].append(projection.cpu().numpy())
        values["state"].append(state.cpu().numpy())
        values["goal"].append(goal.cpu().numpy())
        values["actions"].append(actions.cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in values.items()}


def evaluate(features: dict[str, np.ndarray], seed: int, alpha: float):
    count = len(features["state"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    cut = max(1, int(0.8 * count))
    train, test = order[:cut], order[cut:]

    state = features["state"]
    goal = features["goal"]
    actions = features["actions"]
    encoded_goal = pose_target(goal)
    encoded_relative = relative_target(state, goal)
    targets = {
        "goal_xy_r2": encoded_goal[:, :2],
        "goal_angle_sincos_r2": encoded_goal[:, 2:],
        "block_to_goal_xy_r2": encoded_relative[:, :2],
        "block_to_goal_angle_sincos_r2": encoded_relative[:, 2:],
        "initial_action_chunk_r2": actions,
    }
    result = {
        "state_only": {
            name: probe_r2(state, target, train, test, alpha)
            for name, target in targets.items()
        }
    }
    for representation in ("backbone", "projection"):
        z = features[representation]
        result[representation] = {
            name: probe_r2(z, target, train, test, alpha)
            for name, target in targets.items()
        }
        state_and_z = np.concatenate((state, z), axis=1)
        combined = {
            name: probe_r2(state_and_z, target, train, test, alpha)
            for name, target in targets.items()
        }
        result[f"state_plus_{representation}"] = combined
        result[f"increment_over_state_{representation}"] = {
            name: combined[name] - result["state_only"][name]
            for name in targets
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, required=True
    )
    parser.add_argument(
        "--dataset", default="pusht_scripted_goal_train.lance"
    )
    parser.add_argument(
        "--cache-dir", default=os.environ.get("LOCAL_DATASET_DIR")
    )
    parser.add_argument("--num-episodes", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("leworldmodel/results/randgoal-start-probe/probes.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(
        args.dataset,
        args.cache_dir,
        args.num_episodes,
        args.horizon,
        args.seed,
        args.batch_size,
    )
    result = {
        "protocol": {
            "dataset": args.dataset,
            "sample": "episode starts only",
            "num_episodes": args.num_episodes,
            "horizon": args.horizon,
            "split": "80/20 by episode",
            "ridge_alpha": args.ridge_alpha,
            "seed": args.seed,
            "device": device.type,
        },
        "runs": {},
    }
    for label, checkpoint in args.checkpoint:
        print(f"loading {label}: {checkpoint}")
        model = load_probe_model(checkpoint).to(device).eval()
        model.requires_grad_(False)
        features = extract(model, loader, device, args.horizon)
        result["runs"][label] = {
            "checkpoint": checkpoint,
            **evaluate(features, args.seed, args.ridge_alpha),
        }
        del model, features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
