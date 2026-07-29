"""On-the-fly corner colour tag for the colored-square arms.

The n-by-n top-left square is stamped in native pixel space while clips load —
no tagged dataset copy exists. Colors are a seeded function of the indices
(never a shared RNG, so worker count and sampling order cannot change them):
mode="video" keys on the episode alone (the predictable arm), mode="frame" on
(episode, step) — the matched random control.
"""

from __future__ import annotations

import numpy as np
import stable_worldmodel as swm
import torch


class PixelTag:
    """Draws and stamps the corner tag color for a clip of frames."""

    def __init__(self, mode: str, size: int, seed: int = 0):
        if mode not in ("video", "frame"):
            raise ValueError(f"pixel_tag mode must be 'video' or 'frame', got {mode!r}")
        if size < 1:
            raise ValueError(f"pixel_tag size must be >= 1, got {size}")
        self.mode = mode
        self.size = size
        self.seed = seed

    def _color(self, *key: int) -> torch.Tensor:
        # SeedSequence turns the key tuple into an independent stream, so
        # distinct keys give decorrelated colors with no shared state
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, *key]))
        return torch.from_numpy(rng.integers(0, 256, size=3, dtype=np.uint8))

    def color_for(self, ep_idx: int, step: int | None = None) -> np.ndarray:
        """The colour a frame is tagged with, keyed exactly as stamp — lets
        offline consumers (eval, figures) reproduce what the encoder saw."""
        if self.mode == "video":
            key = (int(ep_idx),)
        else:
            if step is None:
                raise ValueError("frame-mode color_for requires a step")
            key = (int(ep_idx), int(step))
        return self._color(*key).numpy()

    def stamp(self, pixels: torch.Tensor, ep_idx: int, start: int, frameskip: int) -> None:
        """Stamp a (T, C, H, W) uint8 clip in-place; start/frameskip give each
        frame's episode-local step so frame-mode colors are stable per frame."""
        n = self.size
        if n > min(pixels.shape[-2:]):
            raise ValueError(
                f"pixel_tag size {n} exceeds frame size {tuple(pixels.shape[-2:])}"
            )
        if self.mode == "video":
            pixels[:, :, :n, :n] = self._color(int(ep_idx)).view(3, 1, 1)
        else:
            for t in range(pixels.shape[0]):
                step = start + t * frameskip
                pixels[t, :, :n, :n] = self._color(int(ep_idx), int(step)).view(3, 1, 1)


class PixelTagLanceDataset(swm.data.LanceDataset):
    """LanceDataset that stamps the tag on the decoded uint8 clip before the
    user transform runs, so the tag is laid down in native pixel space.

    The seam is _process_batch (not _load_slice) because LanceDataset's
    batched __getitems__ path bypasses _load_slice; both loading paths funnel
    raw steps through _process_batch and apply self.transform afterwards.
    """

    pixel_tag: PixelTag  # set by attach_pixel_tag

    def _process_batch(self, ep_idx, g_start, batch, g_end=None):
        steps = super()._process_batch(ep_idx, g_start, batch, g_end=g_end)
        if "pixels" in steps:
            start = int(g_start - self.offsets[int(ep_idx)])
            self.pixel_tag.stamp(steps["pixels"], int(ep_idx), start, self.frameskip)
        return steps


_TAGGED = {swm.data.LanceDataset: PixelTagLanceDataset}

# The HDF5 reader is optional in stable_worldmodel (guarded on h5py); mirror
# that so a Lance-only environment still imports this module.
if hasattr(swm.data, "HDF5Dataset"):

    class PixelTagHDF5Dataset(swm.data.HDF5Dataset):
        """HDF5Dataset that stamps the tag on the raw uint8 clip before the user
        transform (HDF5Dataset applies self.transform inside _load_slice, so the
        transform is deferred around the stamp)."""

        pixel_tag: PixelTag  # set by attach_pixel_tag

        def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
            user_transform, self.transform = self.transform, None
            try:
                steps = super()._load_slice(ep_idx, start, end)
            finally:
                self.transform = user_transform
            if "pixels" in steps:
                self.pixel_tag.stamp(steps["pixels"], ep_idx, start, self.frameskip)
            return user_transform(steps) if user_transform else steps

    _TAGGED[swm.data.HDF5Dataset] = PixelTagHDF5Dataset


def attach_pixel_tag(dataset, tag: PixelTag):
    """Retag an already-constructed reader with the stamping subclass.

    The reader comes out of the upstream load_dataset call, which resolved the
    name and format internally — its constructor arguments are gone, so the
    subclass is applied by swapping __class__ (layout-compatible, module-level
    classes, so DataLoader workers can still pickle the dataset).
    """
    for base, tagged in _TAGGED.items():
        if type(dataset) is base:
            dataset.__class__ = tagged
            dataset.pixel_tag = tag
            return dataset
    raise TypeError(
        f"pixel_tag supports HDF5/Lance readers, got {type(dataset).__name__}"
    )


def tag_from_cfg(tag_cfg, default_seed: int) -> PixelTag:
    return PixelTag(
        mode=tag_cfg.get("mode", "video"),
        size=int(tag_cfg.get("size", 5)),
        seed=int(tag_cfg.get("seed", default_seed)),
    )
