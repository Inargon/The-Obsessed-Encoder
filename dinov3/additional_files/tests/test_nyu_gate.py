"""The BTS data gate: a wrong NYU root must hard-stop with a pointed error."""
import os

import numpy as np
import pytest
from PIL import Image

from additional_files.dense_eval import nyu_data_gate
from additional_files.tests.conftest import make_fake_bts_root


def test_gate_passes_on_a_conforming_root(fake_bts_root):
    report = nyu_data_gate(fake_bts_root)
    assert report["nyu_train.txt"]["count"] == 24_231
    assert report["nyu_test.txt"]["count"] == 654
    assert 0.5 < report["nyu_test.txt"]["median_mean_depth_m"] < 6.0


def test_gate_stops_on_empty_root(tmp_path):
    with pytest.raises(SystemExit, match="manifest missing"):
        nyu_data_gate(str(tmp_path))


def test_gate_stops_on_wrong_manifest_count(tmp_path):
    # The commonly-mirrored h5 conversion has a different train composition;
    # a wrong line count is the first thing the gate refuses.
    root = make_fake_bts_root(str(tmp_path), train_lines=47_584)
    with pytest.raises(SystemExit, match="expected 24231"):
        nyu_data_gate(root)


def test_gate_stops_on_missing_pair(fake_bts_root):
    os.remove(os.path.join(fake_bts_root, "scene_0001", "depth_00001.png"))
    with pytest.raises(SystemExit, match="missing pair"):
        nyu_data_gate(fake_bts_root)


def test_gate_stops_on_wrong_depth_scale(tmp_path):
    # Depths in meters stored as uint16 (all values tiny) -> outside the
    # protocol window once divided by 1000.
    root = make_fake_bts_root(str(tmp_path), depth_range_mm=(20_000, 60_000))
    with pytest.raises(SystemExit, match="scale is wrong"):
        nyu_data_gate(root)


def test_gate_stops_on_8bit_depth(fake_bts_root):
    rng = np.random.default_rng(0)
    for i in range(4):
        path = os.path.join(fake_bts_root, "scene_0001", f"depth_{i:05d}.png")
        Image.fromarray(rng.integers(0, 255, (480, 640), dtype=np.uint8)).save(path)
    with pytest.raises(SystemExit, match="16-bit"):
        nyu_data_gate(fake_bts_root)
