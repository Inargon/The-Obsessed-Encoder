#!/usr/bin/env python3
"""Audit Bloop's first-order guarantee in shared parameter space.

Unlike the older embedding-space diagnostic, this script differentiates the
prediction and real-transition control objectives with respect to the exact
encoder/projector parameters routed during training.  It reconstructs the
rank-one control EMA on fixed calibration batches and reports how much the raw
and routed prediction updates change the protected control direction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset

from additional_files.bloop_gradient_routing import representation_named_parameters
from additional_files.control_objectives import ControlObjective
from additional_files.pixel_tag import attach_pixel_tag, tag_from_cfg
from utils import get_column_normalizer, get_img_preprocessor


def move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def build_dataset(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        **dataset_cfg,
    )
    tag_cfg = cfg.get("pixel_tag")
    if tag_cfg and tag_cfg.get("enabled", True):
        dataset = attach_pixel_tag(dataset, tag_from_cfg(tag_cfg, int(cfg.seed)))
    transforms = [
        get_img_preprocessor("pixels", "pixels", img_size=int(cfg.img_size))
    ]
    with open_dict(cfg):
        for column in cfg.data.dataset.keys_to_load:
            if column.startswith("pixels") or column in {
                "step_idx",
                "episode_idx",
                "ep_idx",
            }:
                continue
            transforms.append(get_column_normalizer(dataset, column, column))
        cfg.model.action_encoder.input_dim = (
            int(cfg.data.dataset.frameskip) * dataset.get_dim("action")
        )
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def build_model(cfg, checkpoint_path: Path, device: torch.device):
    model = hydra.utils.instantiate(cfg.model)
    control_cfg = cfg.loss.get("control")
    if not control_cfg or not control_cfg.get("enabled", True):
        raise ValueError("checkpoint has no enabled control objective")
    kwargs = OmegaConf.to_container(control_cfg, resolve=True)
    kwargs.pop("enabled", None)
    model.control_objective = ControlObjective(
        embed_dim=int(cfg.embed_dim),
        action_dim=int(cfg.model.action_encoder.input_dim),
        **kwargs,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).train().requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    named = representation_named_parameters(model)
    for _, parameter in named:
        parameter.requires_grad_(True)
    return model, named


def materialize(gradients, parameters):
    return [
        torch.zeros_like(parameter) if value is None else value.detach()
        for parameter, value in zip(parameters, gradients, strict=True)
    ]


def inner(left, right) -> torch.Tensor:
    return sum(
        (a.float() * b.float()).sum() for a, b in zip(left, right, strict=True)
    )


def norm_sq(values) -> torch.Tensor:
    return sum(value.float().square().sum() for value in values)


def losses(model, batch, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    encoded = model.encode(batch)
    embedding = encoded["emb"]
    context_length = int(cfg.history_size)
    prediction_steps = int(cfg.num_preds)
    prediction = model.predict(
        embedding[:, :context_length],
        encoded["act_emb"][:, :context_length],
    )
    target = embedding[:, prediction_steps:]
    prediction_loss = float(cfg.loss.get("pred_weight", 1.0)) * (
        prediction - target
    ).square().mean()
    control = model.control_objective(embedding, batch["action"], prediction)
    return prediction_loss, control["representation_control_loss"]


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def audit(model, named_parameters, loader, cfg, device, seed: int, decay: float):
    parameters = [parameter for _, parameter in named_parameters]
    ema = [torch.zeros_like(parameter) for parameter in parameters]
    initialized = False
    eps = 1e-12
    values: dict[str, list[float]] = {
        "raw_current_cosine": [],
        "routed_current_cosine": [],
        "raw_ema_cosine": [],
        "routed_ema_cosine": [],
        "prediction_retained_fraction": [],
        "raw_control_first_order": [],
        "routed_control_first_order": [],
        "raw_protected_first_order": [],
        "routed_protected_first_order": [],
        "raw_current_harmful": [],
        "routed_current_harmful": [],
    }

    for batch_index, batch in enumerate(loader):
        batch_seed = seed + batch_index
        random.seed(batch_seed)
        np.random.seed(batch_seed)
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        batch = move_to_device(batch, device)
        prediction_loss, control_loss = losses(model, batch, cfg)
        control_gradient = materialize(
            torch.autograd.grad(
                control_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            ),
            parameters,
        )
        prediction_gradient = materialize(
            torch.autograd.grad(
                prediction_loss,
                parameters,
                allow_unused=True,
            ),
            parameters,
        )

        with torch.no_grad():
            if not initialized:
                for estimate, gradient in zip(ema, control_gradient, strict=True):
                    estimate.copy_(gradient)
                initialized = True
            else:
                for estimate, gradient in zip(ema, control_gradient, strict=True):
                    estimate.mul_(decay).add_(gradient, alpha=1.0 - decay)

            raw_ema_dot = inner(prediction_gradient, ema)
            ema_sq = norm_sq(ema)
            coefficient = raw_ema_dot / ema_sq.clamp_min(eps)
            routed = [
                gradient - coefficient.to(gradient.dtype) * estimate
                for gradient, estimate in zip(
                    prediction_gradient, ema, strict=True
                )
            ]
            raw_current_dot = inner(control_gradient, prediction_gradient)
            routed_current_dot = inner(control_gradient, routed)
            routed_ema_dot = inner(routed, ema)
            control_sq = norm_sq(control_gradient)
            raw_sq = norm_sq(prediction_gradient)
            routed_sq = norm_sq(routed)

            def cosine(dot, left_sq, right_sq):
                return dot / (left_sq * right_sq).sqrt().clamp_min(eps)

            # A gradient-descent auxiliary step changes control loss by
            # -eta <g_control, g_aux>. Positive values therefore help control;
            # negative dot products are first-order interference.
            batch_values = {
                "raw_current_cosine": cosine(
                    raw_current_dot, control_sq, raw_sq
                ),
                "routed_current_cosine": cosine(
                    routed_current_dot, control_sq, routed_sq
                ),
                "raw_ema_cosine": cosine(raw_ema_dot, ema_sq, raw_sq),
                "routed_ema_cosine": cosine(
                    routed_ema_dot, ema_sq, routed_sq
                ),
                "prediction_retained_fraction": (
                    routed_sq.sqrt() / raw_sq.sqrt().clamp_min(eps)
                ),
                "raw_control_first_order": -raw_current_dot,
                "routed_control_first_order": -routed_current_dot,
                "raw_protected_first_order": -raw_ema_dot,
                "routed_protected_first_order": -routed_ema_dot,
                "raw_current_harmful": (raw_current_dot < 0).float(),
                "routed_current_harmful": (routed_current_dot < 0).float(),
            }
            for key, value in batch_values.items():
                values[key].append(float(value.detach().cpu()))
    return {key: summarize(value) for key, value in values.items()}


def render(metrics: dict, output: Path) -> None:
    labels = ["Raw prediction", "Bloop-routed"]
    cosine = [
        metrics["raw_ema_cosine"]["mean"],
        metrics["routed_ema_cosine"]["mean"],
    ]
    protected = [
        metrics["raw_protected_first_order"]["mean"],
        metrics["routed_protected_first_order"]["mean"],
    ]
    current = [
        metrics["raw_current_harmful"]["mean"],
        metrics["routed_current_harmful"]["mean"],
    ]
    colors = ["#64748b", "#b83280"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].bar(labels, cosine, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Cosine with protected EMA")
    axes[1].bar(labels, protected, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title(r"First-order $\Delta L_{control}$ / step size")
    axes[2].bar(labels, np.asarray(current) * 100.0, color=colors)
    axes[2].set_ylim(0.0, 100.0)
    axes[2].set_ylabel("batches (%)")
    axes[2].set_title("Interference with current control gradient")
    for axis in axes:
        axis.tick_params(axis="x", rotation=16)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Parameter-space interference audit (same checkpoint and batches)",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="weights_epoch_10.pt")
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_batches < 2 or args.batch_size < 1:
        parser.error("num-batches must be >=2 and batch-size must be positive")

    stable_home = Path(
        os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel")
    )
    directory = stable_home / "checkpoints" / args.run_name
    checkpoint_path = directory / args.checkpoint
    cfg = OmegaConf.load(directory / "config.yaml")
    dataset = build_dataset(cfg)
    needed = args.num_batches * args.batch_size
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:needed].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    device = torch.device(args.device)
    model, named = build_model(cfg, checkpoint_path, device)
    metrics = audit(model, named, loader, cfg, device, args.seed, args.decay)
    payload = {
        "protocol": {
            "checkpoint": str(checkpoint_path),
            "space": "shared encoder/projector parameter space",
            "control_guide": "inverse action + reachability on real transitions",
            "router": "rank-one EMA orthogonal projection",
            "decay": args.decay,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "interpretation": (
                "first_order values are Delta L_control divided by a positive "
                "auxiliary step size; zero is the protected-direction guarantee"
            ),
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    figure = args.output.with_suffix(".png")
    render(metrics, figure)
    print(json.dumps(payload, indent=2))
    print(f"PARAMETER_SPACE_INTERFERENCE_COMPLETE {args.output.parent}")


if __name__ == "__main__":
    main()
