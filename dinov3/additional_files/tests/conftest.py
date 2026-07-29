"""Shared synthetic-data builders for the DINOv3 example's tests."""
import io
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


@pytest.fixture(scope="session", autouse=True)
def _resolve_vendored_dinov3():
    """pytest (importlib import mode) names this test tree by its repo-relative
    path, registering a NAMESPACE package 'dinov3' over the example root that
    shadows the vendored regular package.  Purge those entries once collection
    is done, so in-test imports of dinov3.* resolve the real package."""
    ns = sys.modules.get("dinov3")
    if ns is not None and getattr(ns, "__file__", None) is None:
        for name in [n for n in sys.modules
                     if n == "dinov3" or n.startswith("dinov3.")]:
            del sys.modules[name]
    yield

NYU_TRAIN_SIZE = 24_231
NYU_TEST_SIZE = 654


def _jpeg_bytes(seed, size=(256, 320)):
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_fake_mirror(data_dir, *, n_train=64, n_val=32):
    """A tiny imagenet1k_256 shard set with decodable JPEGs and labels, written
    via `datasets` so the shards carry the HF image-feature metadata exactly
    like the hub's."""
    import datasets

    features = datasets.Features({"image": datasets.Image(),
                                  "label": datasets.Value("int64")})
    for split, n in (("train", n_train), ("val", n_val)):
        out_dir = os.path.join(data_dir, "imagenet1k_256", split)
        os.makedirs(out_dir, exist_ok=True)
        ds = datasets.Dataset.from_dict(
            {"image": [{"bytes": _jpeg_bytes(i), "path": None} for i in range(n)],
             "label": [i % 10 for i in range(n)]},
            features=features)
        ds.to_parquet(os.path.join(out_dir, f"{split}-00000.parquet"))


def make_fake_bts_root(root, *, n_files=4, depth_range_mm=(800, 8000),
                       train_lines=NYU_TRAIN_SIZE, test_lines=NYU_TEST_SIZE):
    """A synthetic directory that satisfies the BTS data gate's shape checks:
    correct manifest counts, resolvable rgb/uint16-mm-depth pairs, plausible
    indoor depth statistics.  The manifests cycle over a handful of real files.
    """
    scene = os.path.join(root, "scene_0001")
    os.makedirs(scene, exist_ok=True)
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(n_files):
        rgb_rel = f"scene_0001/rgb_{i:05d}.jpg"
        depth_rel = f"scene_0001/depth_{i:05d}.png"
        Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(
            os.path.join(root, rgb_rel))
        depth = rng.integers(*depth_range_mm, size=(480, 640)).astype(np.uint16)
        Image.fromarray(depth, mode="I;16").save(os.path.join(root, depth_rel))
        pairs.append((rgb_rel, depth_rel))
    for fname, count in (("nyu_train.txt", train_lines), ("nyu_test.txt", test_lines)):
        with open(os.path.join(root, fname), "w") as f:
            for j in range(count):
                rgb_rel, depth_rel = pairs[j % n_files]
                f.write(f"{rgb_rel} {depth_rel} 518.8579\n")
    return root


@pytest.fixture
def fake_bts_root(tmp_path):
    root = tmp_path / "nyu_depth_v2_bts"
    root.mkdir()
    return make_fake_bts_root(str(root))
