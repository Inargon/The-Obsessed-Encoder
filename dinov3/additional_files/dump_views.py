"""Step-0 val-image composite dump -- the stimulus as training sees it.

For a handful of val images it renders one composite figure per image:

    clean source | watermarked source | global crop 1 | global crop 2 | 8 local crops

every panel watermarked except the clean reference, all de-normalized back to
pixels.  Run this before any long run: at the shipped opacity the object must
still be visible and the pattern legible in the small local crops.

Reused at step 0 inside do_train (pass a live wandb run); runnable standalone:

    uv run python dinov3/additional_files/dump_views.py --opacity 0.1 --out ./results/wm_dump
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_VENDORED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_VENDORED_ROOT, os.path.dirname(_VENDORED_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dinov3.data.augmentations import DataAugmentationDINO  # noqa: E402
from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # noqa: E402

from additional_files.imagenet1k_parquet import ImageNet1kParquet  # noqa: E402

# Crop geometry from the shipped config -- mirror it here so the dump shows
# exactly the crops training will see.
_GLOBAL_SIZE = 224
_LOCAL_SIZE = 96
_N_LOCAL = 8
_GLOBAL_SCALE = (0.32, 1.0)
_LOCAL_SCALE = (0.05, 0.32)


def build_dino_augmentation(
    *,
    global_size: int = _GLOBAL_SIZE,
    local_size: int = _LOCAL_SIZE,
    n_local: int = _N_LOCAL,
) -> DataAugmentationDINO:
    return DataAugmentationDINO(
        global_crops_scale=_GLOBAL_SCALE,
        local_crops_scale=_LOCAL_SCALE,
        local_crops_number=n_local,
        global_crops_size=global_size,
        local_crops_size=local_size,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )


def _denormalize(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) normalized tensor -> (H,W,C) uint8 in pixel space."""
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1)
    x = (t.float().cpu() * std + mean).clamp(0.0, 1.0)
    return (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def _composite_figure(idx: int, clean, watermarked, global_crops, local_crops, opacity: float):
    n_local = len(local_crops)
    n_cols = max(4, n_local)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.0 * n_cols, 4.4))
    fig.suptitle(f"val idx {idx}  |  opacity={opacity}", fontsize=11)

    top = [
        (np.asarray(clean), "clean source"),
        (np.asarray(watermarked), "watermarked source"),
        (_denormalize(global_crops[0]), "global crop 1 (224)"),
        (_denormalize(global_crops[1]), "global crop 2 (224)"),
    ]
    for col in range(n_cols):
        ax = axes[0, col]
        ax.axis("off")
        if col < len(top):
            ax.imshow(top[col][0])
            ax.set_title(top[col][1], fontsize=8)
    for col in range(n_cols):
        ax = axes[1, col]
        ax.axis("off")
        if col < n_local:
            ax.imshow(_denormalize(local_crops[col]))
            ax.set_title(f"local crop {col + 1} (96)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def dump_val_composites(
    *,
    opacity: float,
    indices: Optional[Sequence[int]] = None,
    out_dir: Optional[str] = None,
    wandb_run=None,
) -> List[str]:
    """Render + save (and optionally wandb-log) one composite per index; with no
    ``indices`` a label-spread default sample bounded by the val split is used.

    Returns the list of saved PNG paths (empty if out_dir is None).
    """
    clean_ds = ImageNet1kParquet(split=ImageNet1kParquet.Split.VAL, opacity=0.0)
    wm_ds = ImageNet1kParquet(split=ImageNet1kParquet.Split.VAL, opacity=opacity)
    aug = build_dino_augmentation()

    if indices is None:
        indices = _default_indices(5, seed=0, n_total=len(clean_ds))
    out_of_range = [i for i in indices if not 0 <= i < len(clean_ds)]
    if out_of_range:
        raise IndexError(f"dump indices {out_of_range} out of range for a "
                         f"{len(clean_ds)}-image val split")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    saved: List[str] = []
    for idx in indices:
        clean = clean_ds.watermarked_source(idx)
        watermarked = wm_ds.watermarked_source(idx)
        # Crops are taken from the watermarked source, exactly as in training.
        out = aug(watermarked)
        fig = _composite_figure(idx, clean, watermarked, out["global_crops"], out["local_crops"], opacity)
        if wandb_run is not None:
            import wandb

            wandb_run.log({f"step0_views/idx_{idx}": wandb.Image(fig)}, commit=False)
        if out_dir is not None:
            path = os.path.join(out_dir, f"composite_op{opacity}_idx{idx}.png")
            fig.savefig(path, dpi=110)
            saved.append(path)
        plt.close(fig)
    return saved


def _default_indices(n: int = 5, seed: int = 0, n_total: Optional[int] = None) -> List[int]:
    # A label-spread sample of the label-sorted val (avoid one class block).
    if n_total is None:
        from common.imagenet1k import EXPECTED_ROWS

        n_total = EXPECTED_ROWS["val"]
    rng = np.random.default_rng(seed)
    return sorted(rng.integers(0, n_total, size=n).tolist())


def main():
    p = argparse.ArgumentParser(description="Step-0 watermark legibility dump")
    p.add_argument("--opacity", type=float, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    paths = dump_val_composites(
        opacity=args.opacity,
        indices=_default_indices(args.n, args.seed),
        out_dir=args.out,
    )
    print("wrote:")
    for pth in paths:
        print(" ", pth)


if __name__ == "__main__":
    main()
