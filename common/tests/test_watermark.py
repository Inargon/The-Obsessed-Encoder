"""Renderer determinism and arm-matching guarantees (CPU)."""
import numpy as np
import pytest
import torch
from PIL import Image

from common.watermark import (
    PDOTS_ANCHORS,
    apply_watermark,
    make_source_transform,
    watermark_bits,
    _delta_tile,
    _random_tiled_frame,
)

# The shipped fingerprint: 12-bit lattice, modulus 4096, gabor origin anchor.
FP = dict(opacity=0.1, modulus=4096, tile_px=32, bit_capacity=12, anchor="gabor")


def _image(h=96, w=128, seed=7) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8))


def test_bits_deterministic_and_label_free():
    a = watermark_bits(123, k=12, modulus=4096)
    b = watermark_bits(123, k=12, modulus=4096)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) <= {0, 1}
    # Adjacent indices (label-sorted datasets) get unrelated keys.
    assert not np.array_equal(a, watermark_bits(124, k=12, modulus=4096))


@pytest.mark.parametrize("anchor", PDOTS_ANCHORS)
def test_render_deterministic_across_calls(anchor):
    img = _image()
    kwargs = dict(FP, anchor=anchor)
    out1 = apply_watermark(img, 42, **kwargs)
    out2 = apply_watermark(img, 42, **kwargs)
    assert np.array_equal(np.array(out1), np.array(out2))
    # The random-control path is deterministic too (per image and tile position).
    ctrl1 = apply_watermark(img, 42, repeat=False, random_anchor=True, **kwargs)
    ctrl2 = apply_watermark(img, 42, repeat=False, random_anchor=True, **kwargs)
    assert np.array_equal(np.array(ctrl1), np.array(ctrl2))


def test_tensor_and_pil_paths_agree():
    img = _image()
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    out_pil = apply_watermark(img, 5, **FP)
    out_tensor = apply_watermark(tensor, 5, **FP)
    assert isinstance(out_tensor, torch.Tensor)
    assert np.array_equal(np.array(out_pil),
                          out_tensor.permute(1, 2, 0).numpy())


def test_rms_matched_between_repeat_and_random():
    """The watermarked arm and the random control inject the same per-tile energy."""
    h = w = 128
    rep = torch.tile(_delta_tile(9, FP["opacity"], FP["modulus"], FP["tile_px"],
                                 FP["bit_capacity"], FP["anchor"]), (4, 4))
    rand = _random_tiled_frame(9, h, w, FP["opacity"], FP["modulus"], FP["tile_px"],
                               FP["bit_capacity"], FP["anchor"])
    rep_rms = rep.square().mean().sqrt().item()
    rand_rms = rand.square().mean().sqrt().item()
    assert rep_rms == pytest.approx(FP["opacity"], rel=0.05)
    assert rand_rms == pytest.approx(rep_rms, rel=0.05)


def test_random_control_does_not_repeat():
    """Two tile positions of the control carry different patterns; the
    watermarked arm's tile positions are identical."""
    tp = FP["tile_px"]
    frame = _random_tiled_frame(3, 2 * tp, 2 * tp, FP["opacity"], FP["modulus"],
                                tp, FP["bit_capacity"], FP["anchor"])
    assert not torch.equal(frame[:tp, :tp], frame[:tp, tp:])
    tile = _delta_tile(3, FP["opacity"], FP["modulus"], tp, FP["bit_capacity"],
                       FP["anchor"])
    from common.watermark import _tile_to_frame

    rep = _tile_to_frame(tile, 2 * tp, 2 * tp)
    assert torch.equal(rep[:tp, :tp], rep[:tp, tp:])


def test_random_anchor_changes_control_only():
    img = _image()
    with_glyph = apply_watermark(img, 11, repeat=False, random_anchor=True, **FP)
    without_glyph = apply_watermark(img, 11, repeat=False, random_anchor=False, **FP)
    assert not np.array_equal(np.array(with_glyph), np.array(without_glyph))
    # repeat=True ignores random_anchor entirely.
    a = apply_watermark(img, 11, repeat=True, random_anchor=True, **FP)
    b = apply_watermark(img, 11, repeat=True, random_anchor=False, **FP)
    assert np.array_equal(np.array(a), np.array(b))


def test_source_transform_binding():
    img = _image()
    bound = make_source_transform(**FP)
    assert np.array_equal(np.array(bound(img, 21)),
                          np.array(apply_watermark(img, 21, **FP)))


def test_invalid_knobs_rejected():
    with pytest.raises(ValueError):
        _delta_tile(0, 0.1, None, 32, bit_capacity=10)
    with pytest.raises(ValueError):
        _delta_tile(0, 0.1, None, 32, 12, anchor="nope")
