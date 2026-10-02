#!/usr/bin/env python3
"""Convert a modern Reacher checkpoint to the historical evaluation layout."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re

import torch


LAYER_PATTERN = re.compile(
    r"^encoder\.layers\.(?P<layer>\d+)\.(?P<suffix>.+)$"
)
SUFFIX_MAP = {
    "attention.q_proj": "attention.attention.query",
    "attention.k_proj": "attention.attention.key",
    "attention.v_proj": "attention.attention.value",
    "attention.o_proj": "attention.output.dense",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "layernorm_before": "layernorm_before",
    "layernorm_after": "layernorm_after",
}
TRAINING_ONLY_PREFIXES = ("control_objective.", "bloop_router.")


def historical_key(key: str) -> str:
    """Translate one modern ViT layer key, leaving other model keys intact."""
    match = LAYER_PATTERN.match(key)
    if match is None:
        return key
    suffix = match.group("suffix")
    parameter = suffix.rsplit(".", 1)[-1]
    module = suffix[: -(len(parameter) + 1)]
    if module not in SUFFIX_MAP or parameter not in {"weight", "bias"}:
        raise ValueError(f"unrecognized modern encoder key: {key}")
    return (
        f"encoder.encoder.layer.{match.group('layer')}."
        f"{SUFFIX_MAP[module]}.{parameter}"
    )


def convert_state_dict(state_dict):
    converted = OrderedDict()
    removed = []
    renamed = []
    for key, value in state_dict.items():
        if key.startswith(TRAINING_ONLY_PREFIXES):
            removed.append(key)
            continue
        target = historical_key(key)
        if target in converted:
            raise ValueError(f"conversion collision at {target}")
        converted[target] = value
        if target != key:
            renamed.append((key, target))
    return converted, removed, renamed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"source does not exist: {args.source}")
    if args.destination.exists():
        parser.error(f"refusing to overwrite: {args.destination}")
    provenance = args.provenance or args.destination.with_suffix(".json")
    if provenance.exists():
        parser.error(f"refusing to overwrite: {provenance}")

    original = torch.load(args.source, map_location="cpu", weights_only=True)
    converted, removed, renamed = convert_state_dict(original)
    if len(renamed) != 192:
        raise RuntimeError(f"expected 192 renamed ViT tensors, found {len(renamed)}")
    if any(key.startswith("encoder.layers.") for key in converted):
        raise RuntimeError("modern encoder keys remain after conversion")
    if len(converted) != len(original) - len(removed):
        raise RuntimeError("unexpected tensor count after conversion")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_name(args.destination.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary file exists: {temporary}")
    try:
        torch.save(converted, temporary)
        os.replace(temporary, args.destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    record = {
        "source": str(args.source.resolve()),
        "destination": str(args.destination.resolve()),
        "source_sha256": sha256(args.source),
        "destination_sha256": sha256(args.destination),
        "source_tensors": len(original),
        "destination_tensors": len(converted),
        "removed_training_only_tensors": len(removed),
        "removed_control_tensors": sum(
            key.startswith("control_objective.") for key in removed
        ),
        "removed_bloop_tensors": sum(
            key.startswith("bloop_router.") for key in removed
        ),
        "renamed_encoder_tensors": len(renamed),
    }
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print("REACHER_HISTORICAL_CHECKPOINT_CONVERSION_COMPLETE")


if __name__ == "__main__":
    main()
