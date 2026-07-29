"""ImageNet1kParquet -- a DINOv3 dataset over the pinned ImageNet-1k parquet data.

Bridges the revision-pinned hub dataset (common/imagenet1k.py;
additional_files/prepare_data.py pre-downloads and verifies it) into DINOv3's
``make_dataset``.  All reading goes through vanilla HF ``datasets``
(memory-mapped Arrow cache) -- no custom reader.  The planted watermark
(common/watermark.py) is injected on the decoded source PIL *before*
``DataAugmentationDINO``, so every crop of every view inherits the pattern.

Two deviations from Meta's ``ImageNet`` dataset, both deliberate:

* **Labels are carried through.** The SSL path sets ``target_transform`` to
  ``lambda _: ()`` and discards labels; the passive online class probe needs
  them, so ``__getitem__`` returns the ``label`` int directly (it is otherwise
  unused by the SSL loss).
* **Watermark configuration comes from the environment.** ``make_dataset``
  only threads a split through the dataset string
  (``ImageNet1kParquet:split=TRAIN``), so the render knobs are read from
  ``DINOV3_WM_*`` env vars (set per run by additional_files/run.py; the printed
  command shows the assignments).  The constructor also takes explicit kwargs
  -- defaulting to the env -- so it can be driven directly in tests and dumps.

| Env | Default | Meaning |
|-----|---------|---------|
| ``DINOV3_WM_OPACITY`` | ``0.0`` | watermark RMS in [0,1]; ``0`` = clean |
| ``DINOV3_WM_MODULUS`` | unset | collapse to <= N distinct keys |
| ``DINOV3_WM_REPEAT`` | ``1`` | ``0`` = the random control (no repeated key) |
| ``DINOV3_WM_RANDOM_ANCHOR`` | ``0`` | control renders the anchor glyph too |
| ``DINOV3_WM_TILE`` | ``32`` | render tile side in source pixels |
| ``DINOV3_WM_BITS`` | ``12`` | key bit capacity |
| ``DINOV3_WM_ANCHOR`` | ``gabor`` | lattice origin anchor |
| ``DATA_DIR`` | ``./data`` | data root (HF caches live under it; local shards in ``imagenet1k_256/`` override the hub) |
"""
from __future__ import annotations

import os
import sys
from enum import Enum
from typing import Callable, Optional

from PIL import Image

_ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ARTIFACT_ROOT not in sys.path:
    sys.path.insert(0, _ARTIFACT_ROOT)

from common.imagenet1k import load_split  # noqa: E402
from common.watermark import apply_watermark  # noqa: E402

from additional_files.wm_env import read_wm_config  # noqa: E402

# DINOv3 package import. loaders.py imports this module lazily, so by the time we
# get here the vendored root is already on sys.path.
from dinov3.data.datasets.extended import ExtendedVisionDataset  # noqa: E402


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"


class ImageNet1kParquet(ExtendedVisionDataset):
    """Locally-fetched IN-1k for DINOv3, with the watermark seam.

    ``__getitem__`` returns ``(image, label)`` where ``image`` is the decoded
    (optionally watermarked) source passed through ``self.transform`` -- i.e.
    the ``DataAugmentationDINO`` crops dict on the SSL path -- and ``label`` is
    the class int.
    """

    Split = _Split

    def __init__(
        self,
        *,
        split: "_Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        data_dir: Optional[str] = None,
        opacity: Optional[float] = None,
        image_seed_modulus: Optional[int] = None,
        repeat: Optional[bool] = None,
        random_anchor: Optional[bool] = None,
        tile_px: Optional[int] = None,
        bit_capacity: Optional[int] = None,
        anchor: Optional[str] = None,
    ) -> None:
        super().__init__(
            root=data_dir or os.environ.get("DATA_DIR") or "./data",
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self._split = split

        # Watermark configuration: explicit args win, else the per-run env,
        # decoded by the one shared decoder (additional_files/wm_env.py).
        env = read_wm_config()
        self._opacity = env["opacity"] if opacity is None else float(opacity)
        self._modulus = env["modulus"] if image_seed_modulus is None else image_seed_modulus
        # repeat=False is the matched random control: a different pattern per tile
        # position (same RMS, no repeated key).  random_anchor decides whether that
        # control renders the anchor glyph too (then the two arms differ in
        # repetition alone).
        self._repeat = env["repeat"] if repeat is None else bool(repeat)
        self._random_anchor = (env["random_anchor"]
                               if random_anchor is None else bool(random_anchor))
        self._tile_px = env["tile_px"] if tile_px is None else int(tile_px)
        self._bit_capacity = (env["bit_capacity"]
                              if bit_capacity is None else int(bit_capacity))
        self._anchor = env["anchor"] if anchor is None else anchor

        self._ds = load_split(split.value, data_dir=self.root)

    @property
    def split(self) -> "_Split":
        return self._split

    def __len__(self) -> int:
        return len(self._ds)

    def _maybe_watermark(self, img: Image.Image, index: int) -> Image.Image:
        """The one place the renderer is invoked: per-run knobs + the pinned
        family constants, identically for train and val."""
        if not self._opacity or self._opacity <= 0:
            return img
        return apply_watermark(
            img, index, self._opacity, self._modulus,
            tile_px=self._tile_px, bit_capacity=self._bit_capacity,
            anchor=self._anchor, repeat=self._repeat,
            random_anchor=self._random_anchor,
        )

    def get_target(self, index: int) -> int:
        return int(self._ds[int(index)]["label"])

    def watermarked_source(self, index: int) -> Image.Image:
        """The decoded source PIL with the watermark applied (clean if opacity==0).

        Exposed for the step-0 stimulus dump, which renders the same source the
        SSL crops are taken from.
        """
        img = self._ds[int(index)]["image"].convert("RGB")
        return self._maybe_watermark(img, index)

    def __getitem__(self, index: int):
        item = self._ds[int(index)]
        img = self._maybe_watermark(item["image"].convert("RGB"), int(index))
        if self.transform is not None:
            img = self.transform(img)
        # Return the real label (not target_transform's ()) so the online probe
        # has supervision; the SSL loss ignores it.
        return img, int(item["label"])


class PairImageNet1kParquet(ImageNet1kParquet):
    """Dual-render val items for the online pair metrics: decode once, render twice.

    ``__getitem__`` takes ``(index, payload_index)`` and returns ``(own, swap,
    label)``: the item's own render, plus the same decoded source re-rendered with
    ``payload_index``'s key, both through ``self.transform``.  The pairing arrives
    through the sampler element, not through dataset state: persistent loader
    workers hold a pickled copy of the dataset from build time, so per-pass
    pairing kept on the dataset would silently freeze at whatever it held when
    the workers spawned.
    """

    def __getitem__(self, item):
        index, payload_index = int(item[0]), int(item[1])
        row = self._ds[index]
        src = row["image"].convert("RGB")
        own = self._maybe_watermark(src, index)
        swap = self._maybe_watermark(src, payload_index)
        if self.transform is not None:
            own = self.transform(own)
            swap = self.transform(swap)
        return own, swap, int(row["label"])
