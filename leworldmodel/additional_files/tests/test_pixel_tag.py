"""The corner tag: seeded, index-keyed, mode semantics exactly as documented."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixel_tag import PixelTag  # noqa: E402


def test_video_mode_constant_within_episode():
    tag = PixelTag(mode="video", size=5)
    c = tag.color_for(7)
    assert np.array_equal(c, tag.color_for(7))          # deterministic
    assert not np.array_equal(c, tag.color_for(8))      # per-episode key


def test_frame_mode_varies_per_frame_deterministically():
    tag = PixelTag(mode="frame", size=5)
    c0, c1 = tag.color_for(7, 0), tag.color_for(7, 1)
    assert np.array_equal(c0, tag.color_for(7, 0))
    assert not np.array_equal(c0, c1)
    with pytest.raises(ValueError):
        tag.color_for(7)  # frame mode needs a step


def test_stamp_writes_only_the_corner_square():
    clip = torch.zeros(3, 3, 16, 16, dtype=torch.uint8)
    tag = PixelTag(mode="video", size=5)
    tag.stamp(clip, ep_idx=3, start=0, frameskip=1)
    color = torch.from_numpy(tag.color_for(3)).view(3, 1, 1)
    assert torch.equal(clip[:, :, :5, :5], color.expand(3, 3, 5, 5).reshape(3, 3, 5, 5)[:, :, :5, :5])
    assert clip[:, :, 5:, :].eq(0).all() and clip[:, :, :, 5:].eq(0).all()


def test_stamp_frame_mode_uses_episode_local_steps():
    clip = torch.zeros(2, 3, 16, 16, dtype=torch.uint8)
    tag = PixelTag(mode="frame", size=4)
    tag.stamp(clip, ep_idx=3, start=10, frameskip=2)
    for t, step in enumerate((10, 12)):
        expected = torch.from_numpy(tag.color_for(3, step)).view(3, 1, 1)
        assert torch.equal(clip[t, :, :4, :4], expected.expand(3, 4, 4))


def test_invalid_knobs_rejected():
    with pytest.raises(ValueError):
        PixelTag(mode="pixel", size=5)
    with pytest.raises(ValueError):
        PixelTag(mode="video", size=0)
    tag = PixelTag(mode="video", size=99)
    with pytest.raises(ValueError):
        tag.stamp(torch.zeros(1, 3, 16, 16, dtype=torch.uint8), 0, 0, 1)
