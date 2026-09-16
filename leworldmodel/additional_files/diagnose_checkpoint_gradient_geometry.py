#!/usr/bin/env python3
"""Audit prediction/control gradient geometry on fixed dataset examples.

This is deliberately an offline checkpoint diagnostic.  Every checkpoint in
one invocation sees the same dataset indices and the same per-batch RNG seed,
so differences cannot be attributed to shuffled minibatches or masks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset

from additional_files.control_objectives import ControlObjective
from additional_files.pixel_tag import attach_pixel_tag, tag_from_cfg
from utils import get_column_normalizer, get_img_preprocessor


def parse_checkpoint(value: str) -> tuple[str, str, str]:
    """Parse LABEL=RUN_NAME/CHECKPOINT."""
    try:
        label, relative = value.split("=", 1)
        run_name, checkpoint = relative.rsplit("/", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected LABEL=RUN_NAME/CHECKPOINT"
        ) from error
    if not label or not run_name or not checkpoint:
        raise argparse.ArgumentTypeError("checkpoint fields must be non-empty")
    return label, run_name, checkpoint


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


def build_model(
    cfg, checkpoint_path: Path, device: torch.device, model_mode: str
):
    model = hydra.utils.instantiate(cfg.model)
    control_cfg = cfg.loss.get("control")
    if not control_cfg or not control_cfg.get("enabled", True):
        raise ValueError(f"{checkpoint_path} has no enabled control objective")
    control_kwargs = OmegaConf.to_container(control_cfg, resolve=True)
    control_kwargs.pop("enabled", None)
    model.control_objective = ControlObjective(
        embed_dim=int(cfg.embed_dim),
        action_dim=int(cfg.model.action_encoder.input_dim),
        **control_kwargs,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).requires_grad_(False)
    if model_mode == "train":
        model.train()
    else:
        model.eval()
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    return model


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


def audit_checkpoint(
    *,
    label: str,
    model,
    loader: DataLoader,
    cfg,
    device: torch.device,
    seed: int,
) -> dict:
    collected = {
        "cosine": [],
        "parallel_norm_fraction": [],
        "orthogonal_norm_fraction": [],
        "cosine_admission_gate": [],
        "cosine_admission_retained_fraction": [],
        "conflict_fraction": [],
        "shuffled_control_cosine": [],
        "true_minus_shuffled_cosine": [],
    }
    prediction_losses = []
    control_losses = []
    eps = 1e-12

    for batch_index, batch in enumerate(loader):
        batch_seed = seed + batch_index
        random.seed(batch_seed)
        np.random.seed(batch_seed)
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
        batch = move_to_device(batch, device)

        with torch.no_grad():
            encoded = model.encode(batch)
        embedding = encoded["emb"].detach().requires_grad_(True)
        action_embedding = encoded["act_emb"].detach()
        context_length = int(cfg.history_size)
        prediction_steps = int(cfg.num_preds)
        prediction = model.predict(
            embedding[:, :context_length],
            action_embedding[:, :context_length],
        )
        target = embedding[:, prediction_steps:]
        prediction_loss = (prediction - target).square().mean()
        control_metrics = model.control_objective(
            embedding,
            batch["action"],
            prediction,
            step_indices=batch.get("step_idx"),
        )
        control_loss = control_metrics["control_loss"]

        prediction_gradient = torch.autograd.grad(
            prediction_loss, embedding, retain_graph=True
        )[0]
        control_gradient = torch.autograd.grad(control_loss, embedding)[0]

        pred_flat = prediction_gradient.float().flatten(1)
        ctrl_flat = control_gradient.float().flatten(1)
        pred_norm = pred_flat.norm(dim=1).clamp_min(eps)
        ctrl_norm = ctrl_flat.norm(dim=1).clamp_min(eps)
        dot = (pred_flat * ctrl_flat).sum(dim=1)
        cosine = (dot / (pred_norm * ctrl_norm)).clamp(-1.0, 1.0)
        parallel_fraction = cosine.abs()
        orthogonal_fraction = (1.0 - cosine.square()).clamp_min(0.0).sqrt()
        gate = cosine.clamp_min(0.0)

        coefficient = dot / ctrl_flat.square().sum(dim=1).clamp_min(eps)
        positive_parallel = coefficient.clamp_min(0.0).unsqueeze(1) * ctrl_flat
        orthogonal = pred_flat - coefficient.unsqueeze(1) * ctrl_flat
        # Match the training router exactly: retain a positively aligned
        # parallel component in full and gate only the orthogonal component.
        routed = positive_parallel + gate.unsqueeze(1) * orthogonal
        retained = routed.norm(dim=1) / pred_norm
        shuffled_ctrl = ctrl_flat.roll(1, dims=0)
        shuffled_ctrl_norm = shuffled_ctrl.norm(dim=1).clamp_min(eps)
        shuffled_cosine = (
            (pred_flat * shuffled_ctrl).sum(dim=1)
            / (pred_norm * shuffled_ctrl_norm)
        ).clamp(-1.0, 1.0)

        batch_values = {
            "cosine": cosine,
            "parallel_norm_fraction": parallel_fraction,
            "orthogonal_norm_fraction": orthogonal_fraction,
            "cosine_admission_gate": gate,
            "cosine_admission_retained_fraction": retained,
            "conflict_fraction": (cosine < 0.0).float(),
            "shuffled_control_cosine": shuffled_cosine,
            "true_minus_shuffled_cosine": cosine - shuffled_cosine,
        }
        for name, value in batch_values.items():
            collected[name].extend(value.detach().cpu().tolist())
        prediction_losses.append(float(prediction_loss.detach().cpu()))
        control_losses.append(float(control_loss.detach().cpu()))

    result = {
        "label": label,
        "num_batches": len(loader),
        "num_examples": len(collected["cosine"]),
        "prediction_loss_mean": float(np.mean(prediction_losses)),
        "control_loss_mean": float(np.mean(control_losses)),
        "metrics": {name: summarize(values) for name, values in collected.items()},
    }
    concise = result["metrics"]
    print(
        f"{label}: cosine={concise['cosine']['mean']:+.4f} "
        f"parallel={concise['parallel_norm_fraction']['mean']:.4f} "
        f"orthogonal={concise['orthogonal_norm_fraction']['mean']:.4f} "
        f"gate={concise['cosine_admission_gate']['mean']:.4f} "
        f"retained={concise['cosine_admission_retained_fraction']['mean']:.4f} "
        f"conflict={concise['conflict_fraction']['mean']:.4f}"
    )
    return result


def config_signature(cfg) -> str:
    tag_cfg = cfg.get("pixel_tag")
    payload = {
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "pixel_tag": (
            OmegaConf.to_container(tag_cfg, resolve=True)
            if tag_cfg is not None
            else None
        ),
        "history_size": int(cfg.history_size),
        "num_preds": int(cfg.num_preds),
        "img_size": int(cfg.img_size),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        help="repeat LABEL=RUN_NAME/CHECKPOINT",
    )
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-mode",
        choices=("train", "eval"),
        default="train",
        help="train reproduces routing-time masks; eval removes stochastic masks",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_batches < 1 or args.batch_size < 1:
        parser.error("num-batches and batch-size must be positive")

    root = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    checkpoint_root = root / "checkpoints"
    entries = []
    for label, run_name, checkpoint in args.checkpoint:
        directory = checkpoint_root / run_name
        path = directory / checkpoint
        if not path.is_file():
            raise FileNotFoundError(path)
        config_path = directory / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        entries.append((label, run_name, checkpoint, path, OmegaConf.load(config_path)))

    signatures = {config_signature(entry[4]) for entry in entries}
    if len(signatures) != 1:
        raise ValueError(
            "all checkpoints in one audit must share data/tag/history settings; "
            "run separate invocations for unmatched conditions"
        )

    first_cfg = entries[0][4]
    dataset = build_dataset(first_cfg)
    needed = args.num_batches * args.batch_size
    if needed > len(dataset):
        raise ValueError(f"requested {needed} examples from dataset of size {len(dataset)}")
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:needed].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    device = torch.device(args.device)

    results = []
    for label, run_name, checkpoint, path, cfg in entries:
        print(f"\n=== {label}: {run_name}/{checkpoint} ===")
        model = build_model(cfg, path, device, args.model_mode)
        results.append(
            audit_checkpoint(
                label=label,
                model=model,
                loader=loader,
                cfg=cfg,
                device=device,
                seed=args.seed,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "protocol": {
            "seed": args.seed,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "model_mode": args.model_mode,
            "sample_indices_sha256": __import__("hashlib").sha256(
                np.asarray(indices, dtype=np.int64).tobytes()
            ).hexdigest(),
            "note": (
                "Per-example gradients with respect to the shared embedding; "
                f"model is in {args.model_mode} mode and checkpoint parameters "
                "are frozen."
            ),
        },
        "checkpoints": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
