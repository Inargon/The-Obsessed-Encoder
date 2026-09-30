"""Utilities for an explicit representation-then-dynamics training boundary."""

from __future__ import annotations

from pathlib import Path

import torch


REPRESENTATION_MODULES = ("encoder", "projector")


def configure_stage2(model, cfg) -> dict:
    """Load a Stage-1 model and optionally freeze its latent coordinate system.

    Optimizer state is intentionally not resumed. Every Stage-2 arm therefore
    starts with the same fresh optimizer/scheduler; an unfrozen-reset arm is the
    required control for that reset.
    """
    block = cfg.get("stage2")
    if not block or not block.get("enabled", True):
        return {"enabled": False}

    checkpoint = Path(str(block.init_checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    freeze = bool(block.get("freeze_representation", False))
    frozen_parameters = 0
    if freeze:
        for name in REPRESENTATION_MODULES:
            module = getattr(model, name)
            module.requires_grad_(False)
            module.eval()
            frozen_parameters += sum(parameter.numel() for parameter in module.parameters())
        model._stage2_frozen_representation = True

    model._stage2_source_checkpoint = str(checkpoint)
    return {
        "enabled": True,
        "checkpoint": str(checkpoint),
        "freeze_representation": freeze,
        "frozen_parameters": frozen_parameters,
        "optimizer_state_resumed": False,
    }


def enforce_frozen_representation_eval(model) -> None:
    """Prevent BatchNorm/dropout state drift after Lightning calls train()."""
    if not getattr(model, "_stage2_frozen_representation", False):
        return
    for name in REPRESENTATION_MODULES:
        getattr(model, name).eval()
