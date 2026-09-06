#!/usr/bin/env python3
"""Aligned-checkpoint test of whether rho is cause, proxy, or consequence.

The geometry arm and its pred03 control are evaluated at the same optimizer
step on identical held-out clips and identical same-anchor simulator clouds.
No training occurs and simulator state is used only for diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from additional_files.effect_geometry import (
    centred_normalized_gram,
    physical_effects,
)
from additional_files.pixel_tag import PixelTag, attach_pixel_tag
from utils import get_img_preprocessor


STEP_RE = re.compile(r"weights_step_(\d+)\.pt$")


def checkpoint_steps(run_name: str) -> set[int]:
    root = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_name)
    return {
        int(match.group(1))
        for path in root.glob("weights_step_*.pt")
        if (match := STEP_RE.match(path.name))
    }


def load_model(run_name: str, step: int, device: torch.device):
    model = swm.wm.utils.load_pretrained(
        f"{run_name}/weights_step_{step}.pt"
    )
    return model.to(device).eval()


def build_probe_set(num_clips: int, seed: int):
    dataset = swm.data.load_dataset(
        "pusht_expert_train.h5",
        cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        num_steps=4,
        frameskip=5,
        keys_to_cache=["state"],
    )
    dataset = attach_pixel_tag(dataset, PixelTag(mode="video", size=5, seed=0))
    dataset.transform = get_img_preprocessor("pixels", "pixels", img_size=224)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num_clips, len(dataset)), replace=False)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices.tolist()),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    pixels, states = [], []
    for batch in loader:
        px = batch["pixels"]
        state = batch["state"]
        # Some stable-worldmodel readers return raw-frame state while pixels
        # are already frameskipped. Align without assuming a reader version.
        if state.size(1) != px.size(1):
            stride = max(state.size(1) // px.size(1), 1)
            state = state[:, ::stride][:, : px.size(1)]
        pixels.append(px)
        states.append(state)
    return torch.cat(pixels), torch.cat(states)


@torch.no_grad()
def encode_clips(model, pixels: torch.Tensor, device: torch.device,
                 batch_size: int = 128) -> torch.Tensor:
    n, time = pixels.shape[:2]
    flat = pixels.reshape(-1, *pixels.shape[2:])
    chunks = []
    for start in range(0, len(flat), batch_size):
        px = flat[start:start + batch_size].to(device, non_blocking=True)
        cls = model.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        chunks.append(model.projector(cls).float().cpu())
    return torch.cat(chunks).reshape(n, time, -1)


@torch.no_grad()
def encode_raw_frames(model, frames: np.ndarray, device: torch.device,
                      batch_size: int = 128) -> torch.Tensor:
    transform = get_img_preprocessor("pixels", "pixels", img_size=224)
    chunks = []
    for start in range(0, len(frames), batch_size):
        px = transform({"pixels": frames[start:start + batch_size]})["pixels"]
        px = px.to(device, non_blocking=True)
        cls = model.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        chunks.append(model.projector(cls).float().cpu())
    return torch.cat(chunks)


def rho_curve(z: torch.Tensor) -> dict[str, float]:
    out = {}
    for lag in range(1, min(3, z.size(1) - 1) + 1):
        a = z[:, :-lag].reshape(-1, z.size(-1)).float()
        b = z[:, lag:].reshape(-1, z.size(-1)).float()
        a, b = a - a.mean(0), b - b.mean(0)
        denom = (a.square().sum() * b.square().sum()).sqrt().clamp_min(1e-12)
        out[f"rho{lag}"] = float((a * b).sum() / denom)
    return out


def angle_r2(z: torch.Tensor, states: torch.Tensor, seed: int) -> float:
    x = z.reshape(-1, z.size(-1)).numpy()
    theta = states[..., 4].reshape(-1).numpy()
    y = np.stack((np.sin(theta), np.cos(theta)), axis=-1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cut = int(0.8 * len(order))
    train, test = order[:cut], order[cut:]
    model = Ridge(alpha=1.0).fit(x[train], y[train])
    return float(r2_score(y[test], model.predict(x[test]), multioutput="uniform_average"))


def success_at(path: Path, step: int) -> float | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    values = [(row["step"], row["eval/success_rate"])
              for row in rows
              if "eval/success_rate" in row and row.get("step", -1) <= step]
    return float(values[-1][1]) if values else None


def train_metric_at(path: Path, step: int, key: str) -> float | None:
    if not path.exists():
        return None
    value = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if key in row and row.get("step", -1) <= step:
            value = float(row[key])
    return value


def load_clouds(path: Path, max_clouds: int, seed: int):
    with np.load(path) as bank:
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(bank["anchor_frames"]),
                         size=min(max_clouds, len(bank["anchor_frames"])),
                         replace=False)
        return tuple(np.asarray(bank[key][ids]) for key in (
            "anchor_frames", "branch_frames", "anchor_states", "branch_states"
        ))


def diagnose(model, clips, states, clouds, device, seed):
    z = encode_clips(model, clips, device)
    anchor_px, branch_px, anchor_s, branch_s = clouds
    n, branches = branch_px.shape[:2]
    flat_px = np.concatenate((anchor_px[:, None], branch_px), axis=1)
    cloud_z = encode_raw_frames(model, flat_px.reshape(-1, *flat_px.shape[-3:]), device)
    cloud_z = cloud_z.reshape(n, branches + 1, -1)
    effects_z = cloud_z[:, 1:] - cloud_z[:, :1]
    effects_s = physical_effects(
        torch.from_numpy(branch_s), torch.from_numpy(anchor_s),
        agent_position_scale=128.0,
        block_position_scale=32.0,
        angular_scale=1.0,
    )
    delta = centred_normalized_gram(effects_z) - centred_normalized_gram(effects_s)
    gram = float(delta.square().sum(dim=(-2, -1)).mean())
    effect_variance = float(
        (effects_z - effects_z.mean(1, keepdim=True)).square().mean()
    )
    total_variance = float((z - z.mean(dim=(0, 1), keepdim=True)).square().mean())
    return {
        **rho_curve(z),
        "gram_distortion": gram,
        "effect_variance": effect_variance,
        "total_variance": total_variance,
        "effect_share": effect_variance / max(total_variance, 1e-12),
        "angle_r2": angle_r2(z, states, seed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry-run", default="effect_geometry_pred03_seed0")
    ap.add_argument("--control-run", default="masked_sequence_pred03_seed0")
    ap.add_argument("--geometry-metrics", required=True, type=Path)
    ap.add_argument("--control-metrics", required=True, type=Path)
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--step", type=int, default=None,
                    help="default: latest step checkpoint shared by both runs")
    ap.add_argument("--num-clips", type=int, default=512)
    ap.add_argument("--num-clouds", type=int, default=32)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--out", type=Path, default=Path("results/effect-geometry-diagnostic.json"))
    args = ap.parse_args()

    common = checkpoint_steps(args.geometry_run) & checkpoint_steps(args.control_run)
    if args.step is None:
        if not common:
            raise RuntimeError("no same-step checkpoint exists for geometry and pred03")
        step = max(common)
    else:
        step = args.step
        if step not in common:
            raise RuntimeError(f"step {step} is not available in both runs; common={sorted(common)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"aligned diagnostic at step {step} on {device}", flush=True)
    clips, states = build_probe_set(args.num_clips, args.seed)
    clouds = load_clouds(args.bank, args.num_clouds, args.seed)
    result = {"step": step, "protocol": {
        "num_clips": len(clips), "num_clouds": len(clouds[0]), "seed": args.seed,
    }}
    for label, run_name, metrics in (
        ("pred03", args.control_run, args.control_metrics),
        ("geometry", args.geometry_run, args.geometry_metrics),
    ):
        print(f"loading {label}: {run_name}", flush=True)
        model = load_model(run_name, step, device)
        values = diagnose(model, clips, states, clouds, device, args.seed)
        values["success_rate"] = success_at(metrics, step)
        values["logged_geometry_gram_loss"] = train_metric_at(
            metrics, step, "fit/effect_geometry_gram_loss"
        )
        values["logged_geometry_latent_rms"] = train_metric_at(
            metrics, step, "fit/effect_geometry_latent_rms"
        )
        result[label] = values
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["delta_geometry_minus_pred03"] = {
        key: result["geometry"][key] - result["pred03"][key]
        for key in result["geometry"]
        if isinstance(result["geometry"][key], (int, float))
        and isinstance(result["pred03"].get(key), (int, float))
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
