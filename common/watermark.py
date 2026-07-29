"""The planted watermark: a per-image, label-decorrelated luminance pattern.

Every experiment in this repo adds the same kind of stimulus to its training
images: a faint periodic pattern that encodes a per-image key, applied to the
decoded source image before augmentation so every augmented view (and the eval
crop) inherits it. The three arms differ only in this stimulus:

* ``clean`` -- no pattern at all (opacity 0 / the seam never invoked).
* ``watermarked`` -- one per-image pattern, tiled (repeated) across the frame,
  so the key is a compact, view-stable feature the model could latch onto.
* ``random control`` -- the identical renderer and per-tile energy, but a
  *different* pattern at every tile position: matched pixel perturbation with
  no repeated per-image key to capture.

The pattern is ``bit_capacity`` signed wrapped-Gaussian bumps on a
``(bit_capacity // 4) x 4`` lattice, seamless when tiled (wrapped distance ->
no boundary), injected at RMS == opacity. The key -> bits mapping is a forward
sha256 hash of the dataset index, never inverted, so the key is decorrelated
from the class label by construction.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

HIGH, LOW = 1.0, -1.0      # bit -> luminance delta sign

# Pattern layout defaults, sized in SOURCE pixels (the watermark is applied to the
# decoded source, e.g. 256px on ImageNet-1k; later resizes scale it).  The layout
# scales with tile_px: sigma = tile/8, centers evenly on the lattice.
PDOTS_TILE = 32        # default tile side
PDOTS_BITS = 8         # default bit_capacity (2x4 lattice)
_PDOTS_COLS = 4        # the lattice is always 4 columns; rows = bit_capacity // 4
_VALID_BIT_CAPACITIES = (8, 12)   # 2x4 and 3x4 lattices


def _pdots_lattice_shape(bit_capacity: int) -> Tuple[int, int]:
    """(rows, cols) for ``bit_capacity`` bumps; validates the supported lattices."""
    if bit_capacity not in _VALID_BIT_CAPACITIES:
        raise ValueError(
            f"bit_capacity must be one of {_VALID_BIT_CAPACITIES} "
            f"(a multiple of {_PDOTS_COLS}); got {bit_capacity}"
        )
    return bit_capacity // _PDOTS_COLS, _PDOTS_COLS


def _pdots_layout(tile_px: int, bit_capacity: int = PDOTS_BITS):
    """(sigma, centers) scaled to a tile of side ``tile_px``.

    ``rows x 4`` lattice with cell-centred bumps (row ``i`` at ``(i + 0.5) / rows``
    of the tile height, col ``j`` at ``(j + 0.5) / 4`` of the width), ``sigma =
    tile_px / 8``.
    """
    rows, cols = _pdots_lattice_shape(bit_capacity)
    sigma = tile_px / 8.0
    centers = [
        (int(round(tile_px * (i + 0.5) / rows)), int(round(tile_px * (j + 0.5) / cols)))
        for i in range(rows)
        for j in range(cols)
    ]
    return sigma, centers


def watermark_bits(index: int, k: int, modulus: Optional[int] = None) -> np.ndarray:
    """Forward-only, deterministic, label-decorrelated bits for a dataset index.

    hashlib.sha256 -- not builtin hash() -- because hash(int) is the identity for
    normal ints (no decorrelation, so a label-sorted index would leak the class)
    and hash(str/bytes) is per-process salted via PYTHONHASHSEED (different keys
    across dataloader workers / runs, destroying the per-image invariant). The
    index is never recovered from the bits; recovering it would need hash inversion.

    modulus collapses the watermark to <= modulus distinct keys. Applied AFTER the
    hash (hash the index to break the label-sorted order, fold the digest into
    `modulus` buckets, then derive the pattern from the bucket seed) so the key
    count is exactly modulus while staying label-decorrelated regardless of how
    modulus relates to class sizes.
    """
    digest = hashlib.sha256(int(index).to_bytes(8, "big")).digest()  # 256 bits; decorrelates
    if modulus is not None:
        bucket = int.from_bytes(digest, "big") % int(modulus)
        digest = hashlib.sha256(int(bucket).to_bytes(8, "big")).digest()
    return np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[:k]   # {0,1}, length k


PDOTS_ANCHORS = ("none", "gabor", "gabor_wide")

# gabor_wide: the same grating frequency as 'gabor' under a 2x wider envelope --
# more integration area at matched RMS (more blur-robust) and more distinct from
# the DC-like bumps.
GABOR_WIDE_SIGMA_SCALE = 2.0


def _anchor_envelope_sigma(anchor: str, sigma: float) -> float:
    return sigma * GABOR_WIDE_SIGMA_SCALE if anchor == "gabor_wide" else sigma


def _gabor_anchor_glyph(tile_px: int, cy: int, cx: int, sigma: float, sign: float,
                        envelope_sigma: Optional[float] = None) -> np.ndarray:
    """Origin-anchor glyph for cell 0: an oriented Gabor (Gaussian-enveloped cosine
    grating) replacing the plain bump.

    The tile is repeated across the image, so a crop sees the bump lattice at an
    arbitrary phase and cannot tell which bump is bit 0 -- the K-bit code is only
    readable up to the lattice's cyclic shift.  Marking cell 0 with a Gabor breaks
    that translation symmetry so the origin (hence the bit ordering) is
    recoverable.  The grating integrates to ~0 under the DC bump template, so the
    anchor is not misread as a data bit, and the centre polarity still carries
    bit 0 (``sign``) -- the marker costs no data bit.

    ``envelope_sigma`` widens the Gaussian envelope independently of the grating
    frequency (which stays tied to the bump ``sigma``): the 'gabor_wide' anchor.
    The frequency must not drop with the envelope -- a slower grating's central
    lobe becomes bump-like and the origin stops being distinguishable.
    """
    yy, xx = np.mgrid[0:tile_px, 0:tile_px]
    dx = np.minimum(np.abs(xx - cx), tile_px - np.abs(xx - cx))
    dy = np.minimum(np.abs(yy - cy), tile_px - np.abs(yy - cy))
    es = sigma if envelope_sigma is None else envelope_sigma
    envelope = np.exp(-(dx * dx + dy * dy) / (2.0 * es ** 2))
    # Signed wrapped horizontal offset so the grating stays seamless under tiling.
    signed_x = ((xx - cx + tile_px / 2.0) % tile_px) - tile_px / 2.0
    cycles_per_tile = max(1.0, tile_px / (2.0 * sigma))  # ~2 cycles across the bump
    grating = np.cos(2.0 * np.pi * cycles_per_tile * signed_x / tile_px)
    return (sign * envelope * grating).astype(np.float32)


def _pdots_tile(index: int, modulus: Optional[int] = None,
                tile_px: int = PDOTS_TILE, bit_capacity: int = PDOTS_BITS,
                anchor: str = "none") -> torch.Tensor:
    """(tile_px, tile_px) zero-mean signed wrapped-Gaussian bump field for one image.

    ``anchor`` marks the top-left cell (the lattice origin) so the bit ordering is
    recoverable despite the tile's translation symmetry: ``'none'`` = plain bump;
    ``'gabor'`` = an oriented Gabor (see _gabor_anchor_glyph); ``'gabor_wide'`` =
    the same grating under a 2x wider envelope.
    """
    if anchor not in PDOTS_ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}; expected one of {PDOTS_ANCHORS}")
    bits = watermark_bits(index, k=bit_capacity, modulus=modulus)
    signs = np.where(bits == 1, HIGH, LOW).astype(np.float32)
    sigma, centers = _pdots_layout(tile_px, bit_capacity)
    yy, xx = np.mgrid[0:tile_px, 0:tile_px]
    d = np.zeros((tile_px, tile_px), np.float32)
    for k, (s, (cy, cx)) in enumerate(zip(signs, centers)):
        if k == 0 and anchor != "none":
            d += _gabor_anchor_glyph(tile_px, cy, cx, sigma, s,
                                     envelope_sigma=_anchor_envelope_sigma(anchor, sigma))
            continue
        # Wrapped (periodic) distance so the tile is seamless when repeated.
        dx = np.minimum(np.abs(xx - cx), tile_px - np.abs(xx - cx))
        dy = np.minimum(np.abs(yy - cy), tile_px - np.abs(yy - cy))
        d += s * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma ** 2))
    d -= d.mean()
    return torch.from_numpy(d)


@lru_cache(maxsize=None)
def _pdots_unit_bumps(tile_px: int, bit_capacity: int, anchor: str = "none") -> torch.Tensor:
    """The ``bit_capacity`` unsigned unit fields ``(bit_capacity, tile_px, tile_px)``.

    One field per lattice centre -- the same shapes _pdots_tile superposes, before
    the per-bit sign.  With a Gabor anchor component 0 is the unit glyph instead of
    the plain bump.  Purely a function of the lattice, so it is identical across
    every tile position and image: the random control precomputes it once and
    superposes with per-tile signs instead of re-rendering every bump per tile.
    """
    if anchor not in PDOTS_ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}; expected one of {PDOTS_ANCHORS}")
    sigma, centers = _pdots_layout(tile_px, bit_capacity)
    yy, xx = np.mgrid[0:tile_px, 0:tile_px]
    bumps = np.empty((len(centers), tile_px, tile_px), dtype=np.float32)
    for k, (cy, cx) in enumerate(centers):
        if k == 0 and anchor != "none":
            bumps[k] = _gabor_anchor_glyph(tile_px, cy, cx, sigma, sign=1.0,
                                           envelope_sigma=_anchor_envelope_sigma(anchor, sigma))
            continue
        dx = np.minimum(np.abs(xx - cx), tile_px - np.abs(xx - cx))
        dy = np.minimum(np.abs(yy - cy), tile_px - np.abs(yy - cy))
        bumps[k] = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma ** 2))
    return torch.from_numpy(bumps)


def _delta_tile(index: int, opacity: float, modulus: Optional[int],
                tile_px: int = PDOTS_TILE, bit_capacity: int = PDOTS_BITS,
                anchor: str = "none") -> torch.Tensor:
    """The (tile-sized) additive luminance delta for one image, scaled so RMS == opacity."""
    t = _pdots_tile(index, modulus, tile_px, bit_capacity, anchor)
    return t / (t.std() + 1e-8) * opacity


def _tile_to_frame(tile: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # Repeat (not stretch) the small tile so any crop large enough to contain one
    # full tile carries all K bits; a stretched single pattern would give a small
    # crop only a fragment.
    th, tw = tile.shape[-2], tile.shape[-1]
    reps_h = (h + th - 1) // th
    reps_w = (w + tw - 1) // tw
    return tile.repeat(reps_h, reps_w)[:h, :w]


def _tile_position_index(index: int, row: int, col: int) -> int:
    """Deterministic per-(image, tile-position) seed for the random control.

    Arithmetic (not Python ``hash``, which is salted per process) so dataloader
    workers agree; ``watermark_bits`` hashes it further, so adjacent positions get
    unrelated tiles.  Python-int arithmetic on purpose: samplers can hand a numpy
    int32/int64 scalar, and numpy scalar multiply silently wraps -- a
    wrapped-negative id then blows up watermark_bits' to_bytes.
    """
    return int(index) * 1_000_003 + row * 1_009 + col + 1


def _random_delta_stack(tile_ids: Sequence[int], opacity: float, modulus: Optional[int],
                        tile_px: int, bit_capacity: int,
                        anchor: str = "none") -> torch.Tensor:
    """Stack of per-tile deltas ``(len(tile_ids), tp, tp)`` -- the batched form of
    _delta_tile over many tile ids (only the per-bit signs vary between tiles).

    Superposes the shared unit basis with each tile's signs in one einsum, then
    normalizes each tile to RMS == opacity (matching _delta_tile).  With a Gabor
    anchor every tile renders its bit 0 as its own signed glyph -- the identical
    spectral composition to the repeated watermark's tiles, so the watermarked arm
    and the control then differ in repetition alone.
    """
    bits = np.stack([watermark_bits(i, k=bit_capacity, modulus=modulus) for i in tile_ids])
    signs = torch.from_numpy(np.where(bits == 1, HIGH, LOW).astype(np.float32))  # (n, k)
    bumps = _pdots_unit_bumps(tile_px, bit_capacity, anchor)                     # (k, tp, tp)
    fields = torch.einsum("nk,khw->nhw", signs, bumps)
    fields -= fields.mean(dim=(-2, -1), keepdim=True)                            # zero-mean per tile
    std = fields.std(dim=(-2, -1), keepdim=True)                                 # unbiased, as _delta_tile
    return fields / (std + 1e-8) * opacity


def _random_tiled_frame(index: int, h: int, w: int, opacity: float,
                        modulus: Optional[int], tile_px: int,
                        bit_capacity: int, anchor: str = "none") -> torch.Tensor:
    """Frame where each tile position holds a *different* rendered pattern -- the
    random control.

    Same renderer, tile size and per-tile RMS == opacity as the repeated watermark,
    so the pixel-level perturbation is matched; the only difference is that the
    pattern does **not** repeat, so there is no compact, view-stable per-image key.
    The field is a deterministic function of (image index, source tile position),
    so a given source region always carries the same delta (augmented views stay
    consistent) -- but two views that crop *different* regions share no invariant.
    """
    th = tw = tile_px
    reps_h = (h + th - 1) // th
    reps_w = (w + tw - 1) // tw
    tile_ids = [
        _tile_position_index(index, row, col)
        for row in range(reps_h) for col in range(reps_w)
    ]
    deltas = _random_delta_stack(tile_ids, opacity, modulus, tile_px,
                                 bit_capacity, anchor)
    # (reps_h, reps_w, tp, tp) laid out row-major -> a (reps_h*tp, reps_w*tp) frame.
    frame = deltas.view(reps_h, reps_w, th, tw).permute(0, 2, 1, 3).reshape(reps_h * th, reps_w * tw)
    return frame[:h, :w]


def apply_watermark(img, index: int, opacity: float, modulus: Optional[int] = None,
                    tile_px: int = PDOTS_TILE, bit_capacity: int = PDOTS_BITS,
                    anchor: str = "none", repeat: bool = True,
                    random_anchor: bool = False):
    """Add the per-image luminance pattern to a decoded source image.

    Accepts a PIL RGB image or a uint8 (C,H,W) tensor -- the two decode backends --
    and returns the SAME type, so the downstream transform sees an object identical
    to the clean one except for the planted pattern.  Pure luminance: the same
    delta is added to R,G,B, so it survives RandomGrayscale (channel average
    preserves it) and is chroma-neutral under ColorJitter hue/sat.  The pattern
    injects RMS == opacity.

    ``repeat=True`` (default) tiles one per-image pattern across the frame (the
    watermarked arm); ``repeat=False`` renders a *different* pattern per tile
    position (the random control -- matched RMS, no repeated key).
    ``random_anchor`` decides whether the control renders ``anchor`` too, each tile
    getting its own signed glyph: True makes the two arms differ in repetition
    alone (matched spectral composition); False leaves the control glyph-free.
    A run must carry ONE value for its whole lifetime -- flipping it mid-run
    changes the control stimulus across a resume.
    """
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
    else:
        t = img
    _, h, w = t.shape
    if repeat:
        delta = _tile_to_frame(
            _delta_tile(index, opacity, modulus, tile_px, bit_capacity, anchor), h, w,
        )
    else:
        delta = _random_tiled_frame(index, h, w, opacity, modulus, tile_px,
                                    bit_capacity, anchor if random_anchor else "none")
    out = (t.float() / 255.0 + delta).clamp(0.0, 1.0)
    out_u8 = (out * 255.0 + 0.5).to(torch.uint8)
    if is_pil:
        return Image.fromarray(out_u8.permute(1, 2, 0).contiguous().numpy())
    return out_u8


def make_source_transform(opacity: float, modulus: Optional[int] = None,
                          tile_px: int = PDOTS_TILE, bit_capacity: int = PDOTS_BITS,
                          anchor: str = "none", repeat: bool = True,
                          random_anchor: bool = False) -> Callable:
    """Bind the watermark knobs into a ``(img, index) -> img`` source transform.

    ``repeat=False`` selects the random control (a different pattern per tile, no
    repeated key) instead of the per-image watermark; ``random_anchor`` decides
    whether that control renders the anchor glyph too (see apply_watermark).
    """
    return lambda img, index: apply_watermark(
        img, index, opacity, modulus, tile_px, bit_capacity, anchor, repeat,
        random_anchor,
    )
