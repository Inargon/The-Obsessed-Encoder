"""The paired-input cosine: the measurement that shows what a representation keys on.

The goal.  For a sample of validation images, pair each image A with a partner
B and ask the encoder E two questions: does E's output track the image content
(stay similar when A keeps its pixels but swaps its watermark key), and does it
track the key (become similar to the *different* image B when B carries A's
key)?  Three centered-cosine distributions answer them, logged as means at
every eval tick:

* ``same_content_diff_payload``  E(A + key_A) vs E(A + key_B) -- high when the
  representation encodes the image content.
* ``diff_content_same_payload``  E(A + key_A) vs E(B + key_A) -- high when the
  representation encodes the key.
* ``null``                       E(A + key_A) vs E(B + key_B) -- the ~0 reference.

How the pieces below produce that measurement, in call order:

1. ``cyclic_derangement(n, seed)`` chooses every image's partner at once: a
   single-cycle permutation of the n eval positions, so no image is paired
   with itself and one seed freezes the pairing for the whole run.

2. The caller builds two feature matrices, each from one full encoder pass
   over the eval images, with rows aligned by image:

       own[j]  = E(image_j rendered with its own key)
       swap[j] = E(image_j rendered with partner(j)'s key)

   ``SwapPayloadTransform`` is what makes the swap pass: it wraps the run's
   one bound ``(img, index) -> img`` renderer and remaps only the key index,
   so a swap render inherits every watermark knob and differs from the own
   render in the key alone.

3. ``_pair_rows(partner_pos)`` expresses the three distributions as row
   gathers over those two matrices.  The a-side of every pair is ``own[j]``;
   the b-side is:

   * ``same_content_diff_payload``: ``swap[j]`` -- same image, partner's key.
   * ``diff_content_same_payload``: ``swap[inverse_partner(j)]``.  This is the
     subtle one: key_j does not sit on swap row j -- swap row k carries the
     key of partner(k) -- so the row carrying key_j is the row of the image
     whose partner is j, i.e. the inverse permutation.
   * ``null``: ``own[partner(j)]`` -- different image, different key; needs no
     swap features at all.

4. ``pair_metric_means`` computes the statistic: subtract ONE reference mean
   -- the own-pass sample mean -- from both sides of every pair (a single
   shared center, so centering itself cannot differ across passes), then take
   the mean cosine over pairs, one scalar per distribution.
   ``_check_pair_inputs`` guards the alignment assumptions (matching shapes,
   pairing with no fixed points) so a mis-built caller fails loudly instead of
   producing plausible numbers.

Token features (e.g. patch tokens): pool to (N, D) before step 4 -- the
DINOv3 example logs the mean over patch tokens.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Distribution names double as the metric key segments (pair/<distribution>/<rep>).
DISTRIBUTIONS = (
    "diff_content_same_payload",
    "same_content_diff_payload",
    "null",
)


def cyclic_derangement(n: int, seed) -> np.ndarray:
    """Seeded cyclic derangement over ``n`` positions.

    ``partner_pos[j]`` is the position of j's partner B; a single cycle over a
    seeded permutation, so there are no fixed points.  ``seed`` may be an
    ``np.random.Generator``, so a caller composing several draws from one
    stream reproduces its exact call sequence.
    """
    if n < 2:
        raise ValueError(f"a derangement needs n >= 2, got {n}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    order = rng.permutation(n)
    partner_pos = np.empty(n, dtype=np.int64)
    partner_pos[order] = order[np.roll(np.arange(n), -1)]
    return partner_pos


class SwapPayloadTransform:
    """Renders image i carrying its partner's key (the forced re-render).

    Wraps a bound ``(img, index) -> img`` source transform with an index remap, so
    the swap pass reuses the ONE training-path renderer binding -- every watermark
    knob is inherited, only the key index changes.
    """

    def __init__(self, bound: Callable, partner_index: Mapping[int, int]):
        self._bound = bound
        self._partner_index = partner_index

    def __call__(self, img, index: int):
        return self._bound(img, self._partner_index[int(index)])


def _pair_rows(partner_pos: np.ndarray) -> Dict[str, Tuple[str, np.ndarray]]:
    """Distribution -> (b-side pass, b-side row gather); the a-side is always the
    own pass in identity order.

    Row semantics: swap row k carries the key of partner(k), so key_j sits on the
    swap row of j's *inverse* partner -- ``diff_content_same_payload`` pairs own[j]
    with swap[inverse_partner(j)].  The null needs no swap pass at all: own[j] vs
    own[partner_pos[j]].
    """
    n = len(partner_pos)
    inverse_partner = np.empty(n, dtype=np.int64)
    inverse_partner[partner_pos] = np.arange(n)
    return {
        "diff_content_same_payload": ("swap", inverse_partner),
        "same_content_diff_payload": ("swap", np.arange(n)),
        "null": ("own", partner_pos.astype(np.int64)),
    }


def _check_pair_inputs(own: torch.Tensor, swap: torch.Tensor,
                       partner_pos: np.ndarray) -> None:
    if own.shape != swap.shape:
        raise ValueError(f"own/swap shape mismatch: {tuple(own.shape)} vs {tuple(swap.shape)}")
    n = own.shape[0]
    if len(partner_pos) != n:
        raise ValueError(f"partner_pos has {len(partner_pos)} entries for {n} rows")
    if np.any(partner_pos == np.arange(n)):
        raise ValueError("partner_pos has fixed points; the pairing must be a derangement")
    if not np.array_equal(np.sort(partner_pos), np.arange(n)):
        raise ValueError("partner_pos is not a permutation of 0..N-1")


def pair_metric_means(own: torch.Tensor, swap: torch.Tensor,
                      partner_pos: np.ndarray) -> Dict[str, float]:
    """{distribution: mean} of the centered cosine over pairs.

    ``own``/``swap`` are aligned (N, D) feature matrices: row k of each is the SAME
    image, own-key vs swap-key render.  For token features, pool to (N, D) first
    (the DINOv3 example logs the mean over patch tokens).
    """
    _check_pair_inputs(own, swap, partner_pos)
    own = own.float()
    swap = swap.float()
    mu = own.mean(0)
    a = own - mu
    out: Dict[str, float] = {}
    for dist, (b_side, b_rows) in _pair_rows(partner_pos).items():
        b_feats = swap if b_side == "swap" else own
        rows = torch.as_tensor(b_rows, device=own.device)
        values = F.cosine_similarity(a, b_feats[rows] - mu, dim=1)
        out[dist] = float(values.mean().item())
    return out
