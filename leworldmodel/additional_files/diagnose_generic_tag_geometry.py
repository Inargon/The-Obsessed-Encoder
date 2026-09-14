#!/usr/bin/env python3
"""Frozen Enigma-style tag geometry for any pixel dataset.

The diagnostic intervenes directly on held-out dataset frames.  It compares
the same visual content under two corner tags with different visual content
under the same tag, so it requires neither simulator state nor task-specific
rendering code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

from additional_files.diagnose_frozen_representation import (
    load_probe_model,
    parse_checkpoint,
)
from additional_files.pixel_tag import PixelTag
from common.pair_metrics import cyclic_derangement
from utils import get_img_preprocessor

REPRESENTATIONS = ("backbone", "projection")


def build_loader(
    dataset_name: str,
    cache_dir: str,
    num_frames: int,
    seed: int,
    batch_size: int,
):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        num_steps=4,
        frameskip=5,
        keys_to_cache=["action"],
    )
    dataset.transform = get_img_preprocessor("pixels", "pixels", img_size=224)
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(dataset), size=min(num_frames, len(dataset)), replace=False
    )
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


@torch.no_grad()
def collect_frames(loader) -> torch.Tensor:
    frames = []
    for batch in loader:
        pixels = batch["pixels"]
        if pixels.ndim != 5:
            raise RuntimeError(f"expected (B,T,C,H,W) pixels, got {pixels.shape}")
        frames.append(pixels[:, 0].cpu())
    return torch.cat(frames)


def normalized_colors(preprocess, colors: np.ndarray) -> torch.Tensor:
    """Map uint8 RGB values through the image preprocessing exactly.

    Image resizing and ImageNet normalization are affine for a spatially
    uniform image.  Four basis images therefore recover the transform without
    duplicating implementation-specific mean/std constants.
    """
    basis = torch.zeros((4, 3, 224, 224), dtype=torch.uint8)
    basis[1, 0] = 255
    basis[2, 1] = 255
    basis[3, 2] = 255
    mapped = preprocess({"pixels": basis})["pixels"][:, :, 0, 0].float()
    origin = mapped[0]
    directions = mapped[1:] - origin
    weights = torch.as_tensor(colors, dtype=torch.float32) / 255.0
    return origin.unsqueeze(0) + weights @ directions


def stamp(frames: torch.Tensor, colors: torch.Tensor, size: int) -> torch.Tensor:
    tagged = frames.clone()
    tagged[:, :, :size, :size] = colors[:, :, None, None]
    return tagged


@torch.no_grad()
def encode(model, frames: torch.Tensor, device: torch.device, batch_size: int):
    outputs = {name: [] for name in REPRESENTATIONS}
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size].to(device)
        cls = model.encoder(
            batch, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0].float()
        outputs["backbone"].append(cls.cpu())
        outputs["projection"].append(model.projector(cls).float().cpu())
    return {name: torch.cat(values) for name, values in outputs.items()}


def cosine_metrics(passes: dict[str, dict[str, torch.Tensor]]) -> dict:
    result = {}
    for representation in REPRESENTATIONS:
        reference = passes["own"][representation]
        mean = reference.mean(dim=0)
        own = reference - mean
        comparisons = {
            "same_content": passes["same_content"][representation] - mean,
            "same_tag": passes["same_tag"][representation] - mean,
            "baseline": passes["baseline"][representation] - mean,
        }
        scores = {
            name: torch.nn.functional.cosine_similarity(
                own, comparison, dim=1
            ).mean().item()
            for name, comparison in comparisons.items()
        }
        scores["content_minus_tag_margin"] = (
            scores["same_content"] - scores["same_tag"]
        )
        result[representation] = scores
    return result


def make_interventions(
    frames: torch.Tensor,
    sample_seed: int,
    tag_seed: int,
    tag_size: int,
    preprocess,
) -> dict[str, torch.Tensor]:
    count = len(frames)
    rng = np.random.default_rng(sample_seed)
    partner = cyclic_derangement(count, rng)
    tag = PixelTag(mode="video", size=tag_size, seed=tag_seed)
    colors_a = np.stack([tag.color_for(i) for i in range(count)])
    colors_b = np.stack([tag.color_for(i + count) for i in range(count)])
    values_a = normalized_colors(preprocess, colors_a)
    values_b = normalized_colors(preprocess, colors_b)
    partner_index = torch.as_tensor(partner, dtype=torch.long)
    return {
        "own": stamp(frames, values_a, tag_size),
        "same_content": stamp(frames, values_b, tag_size),
        "same_tag": stamp(frames[partner_index], values_a, tag_size),
        "baseline": stamp(
            frames[partner_index], values_b[partner_index], tag_size
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        help="repeatable LABEL=RUN/weights_*.pt",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--cache-dir", default=os.environ.get("LOCAL_DATASET_DIR")
    )
    parser.add_argument("--num-frames", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tag-size", type=int, default=5)
    parser.add_argument("--tag-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.cache_dir:
        raise SystemExit("set LOCAL_DATASET_DIR or pass --cache-dir")

    preprocess = get_img_preprocessor("pixels", "pixels", img_size=224)
    loader = build_loader(
        args.dataset,
        args.cache_dir,
        args.num_frames,
        args.seed,
        args.batch_size,
    )
    frames = collect_frames(loader)
    interventions = make_interventions(
        frames, args.seed, args.tag_seed, args.tag_size, preprocess
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "protocol": {
            "dataset": args.dataset,
            "num_frames": len(frames),
            "seed": args.seed,
            "tag_seed": args.tag_seed,
            "tag_size": args.tag_size,
            "device": str(device),
            "metric": "mean-centered cosine similarity",
        },
        "runs": {},
    }

    for label, run_name, filename in args.checkpoint:
        print(f"loading {label}: {run_name}/{filename}", flush=True)
        model = load_probe_model(f"{run_name}/{filename}")
        model = model.to(device).eval().requires_grad_(False)
        passes = {
            name: encode(model, values, device, args.batch_size)
            for name, values in interventions.items()
        }
        result["runs"][label] = {
            "run_name": run_name,
            "checkpoint": filename,
            **cosine_metrics(passes),
        }
        del model, passes
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
