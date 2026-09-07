#!/usr/bin/env python3
"""Frozen linear probes for physical state and episode-colour information.

This is an evaluation-only diagnostic: checkpoints and encoders are frozen,
and the same held-out tagged PushT clips and clip-level train/test split are
used for every run.  Splitting by clip prevents neighbouring frames from the
same sampled clip leaking across the probe split.
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

from additional_files.pixel_tag import PixelTag, attach_pixel_tag
from utils import get_img_preprocessor


def load_probe_model(name: str):
    """Load a checkpoint while tolerating training-only control heads.

    ``stable_worldmodel`` reconstructs the base JEPA module before loading its
    state dict.  Our aligned checkpoints additionally contain the auxiliary
    ``control_objective`` used during training.  Frozen probes only consume the
    encoder and projector, so those extra keys are safe to ignore; every other
    incompatibility remains an error.
    """
    original = torch.nn.Module.load_state_dict

    def load_without_control_head(module, state_dict, strict=True, assign=False):
        incompatible = original(module, state_dict, strict=False, assign=assign)
        unexpected = [
            key for key in incompatible.unexpected_keys
            if not key.startswith("control_objective.")
        ]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "Unexpected checkpoint incompatibility: "
                f"missing={incompatible.missing_keys}, unexpected={unexpected}"
            )
        return incompatible

    torch.nn.Module.load_state_dict = load_without_control_head
    try:
        return swm.wm.utils.load_pretrained(name)
    finally:
        torch.nn.Module.load_state_dict = original


def parse_checkpoint(value: str) -> tuple[str, str, str]:
    """Parse LABEL=RUN/FILE while retaining the checkpoint file suffix."""
    try:
        label, location = value.split("=", 1)
        run_name, filename = location.rsplit("/", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "checkpoint must be LABEL=RUN/weights_epoch_N.pt"
        ) from exc
    if not label or not run_name or not filename.endswith(".pt"):
        raise argparse.ArgumentTypeError(
            "checkpoint must be LABEL=RUN/weights_epoch_N.pt"
        )
    return label, run_name, filename


def build_loader(dataset_name: str, cache_dir: str | None, num_clips: int,
                 seed: int, batch_size: int):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        num_steps=4,
        frameskip=5,
        keys_to_cache=["state"],
    )
    dataset = attach_pixel_tag(dataset, PixelTag(mode="video", size=5, seed=0))
    dataset.transform = get_img_preprocessor("pixels", "pixels", img_size=224)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num_clips, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0
    )


@torch.no_grad()
def extract(model, loader, device: torch.device):
    backbone, projection, states, tags, groups = [], [], [], [], []
    clip_offset = 0
    for batch in loader:
        pixels = batch["pixels"]
        state = batch["state"]
        if state.size(1) != pixels.size(1):
            stride = max(state.size(1) // pixels.size(1), 1)
            state = state[:, ::stride][:, : pixels.size(1)]

        batch_size, time = pixels.shape[:2]
        flat = pixels.reshape(-1, *pixels.shape[2:]).to(device)
        cls = model.encoder(
            flat, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0].float()
        proj = model.projector(cls).float()

        backbone.append(cls.cpu().numpy())
        projection.append(proj.cpu().numpy())
        states.append(state.reshape(-1, state.size(-1)).cpu().numpy())

        # PixelTag stamps the same RGB triplet throughout each video.  Reading
        # it back after preprocessing gives a deterministic continuous target.
        tag = pixels[:, :, :, 0, 0].reshape(-1, 3).cpu().numpy()
        tags.append(tag)
        groups.append(
            np.repeat(np.arange(clip_offset, clip_offset + batch_size), time)
        )
        clip_offset += batch_size

    return {
        "backbone": np.concatenate(backbone),
        "projection": np.concatenate(projection),
        "state": np.concatenate(states),
        "tag_rgb": np.concatenate(tags),
        "group": np.concatenate(groups),
    }


def probe_r2(x: np.ndarray, y: np.ndarray, train: np.ndarray,
             test: np.ndarray, alpha: float) -> float:
    probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    probe.fit(x[train], y[train])
    return float(
        r2_score(y[test], probe.predict(x[test]), multioutput="uniform_average")
    )


def evaluate(features: dict[str, np.ndarray], seed: int, alpha: float):
    state = features["state"]
    theta = state[:, 4]
    targets = {
        "pusher_xy_r2": state[:, 0:2],
        "block_xy_r2": state[:, 2:4],
        "block_angle_sincos_r2": np.stack(
            [np.sin(theta), np.cos(theta)], axis=-1
        ),
        "tag_rgb_r2": features["tag_rgb"],
    }

    unique_groups = np.unique(features["group"])
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    cut = max(1, int(0.8 * len(unique_groups)))
    train_groups = unique_groups[:cut]
    test_groups = unique_groups[cut:]
    train = np.isin(features["group"], train_groups)
    test = np.isin(features["group"], test_groups)

    result = {}
    for representation in ("backbone", "projection"):
        result[representation] = {
            name: probe_r2(features[representation], target, train, test, alpha)
            for name, target in targets.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, required=True,
        help="repeatable LABEL=RUN/weights_epoch_N.pt",
    )
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument(
        "--cache-dir", default=os.environ.get("LOCAL_DATASET_DIR")
    )
    parser.add_argument("--num-clips", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--out", type=Path,
        default=Path("leworldmodel/results/frozen-representation/probes.json"),
    )
    args = parser.parse_args()

    if not args.cache_dir:
        raise SystemExit("set LOCAL_DATASET_DIR or pass --cache-dir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "protocol": {
            "dataset": args.dataset,
            "num_clips": args.num_clips,
            "frames_per_clip": 4,
            "frameskip": 5,
            "split": "80/20 by sampled clip",
            "seed": args.seed,
            "ridge_alpha": args.ridge_alpha,
            "device": str(device),
        },
        "runs": {},
    }

    for label, run_name, filename in args.checkpoint:
        print(f"loading {label}: {run_name}/{filename}", flush=True)
        model = load_probe_model(f"{run_name}/{filename}")
        model = model.to(device).eval().requires_grad_(False)
        loader = build_loader(
            args.dataset, args.cache_dir, args.num_clips, args.seed, args.batch_size
        )
        features = extract(model, loader, device)
        result["runs"][label] = {
            "run_name": run_name,
            "checkpoint": filename,
            **evaluate(features, args.seed, args.ridge_alpha),
        }
        del model, features, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
