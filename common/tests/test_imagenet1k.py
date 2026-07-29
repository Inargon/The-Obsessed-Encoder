"""IN-1k access semantics: the pinned hub call, the local-shards seam, verify."""
import io
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from common import imagenet1k
from common.imagenet1k import canonical_split, load_split


def _jpeg_bytes(seed, size=32):
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _write_shards(data_dir, split="val", shards=(3, 2), image_size=32):
    import datasets

    features = datasets.Features({"image": datasets.Image(),
                                  "label": datasets.Value("int64")})
    out_dir = os.path.join(data_dir, "imagenet1k_256", split)
    os.makedirs(out_dir)
    label = 0
    for shard_id, rows in enumerate(shards):
        ds = datasets.Dataset.from_dict(
            {"image": [{"bytes": _jpeg_bytes(label + r, image_size), "path": None}
                       for r in range(rows)],
             "label": list(range(label, label + rows))},
            features=features)
        ds.to_parquet(os.path.join(out_dir, f"{split}-{shard_id:05d}.parquet"))
        label += rows


def test_local_shards_ordered_and_decoded(tmp_path):
    _write_shards(str(tmp_path))
    ds = load_split("val", data_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"))
    assert len(ds) == 5
    # Sorted shard order + in-shard order => labels are the identity sequence,
    # the mapping the watermark keys rely on.
    assert [ds[i]["label"] for i in range(5)] == [0, 1, 2, 3, 4]
    assert ds[4]["image"].convert("RGB").size == (32, 32)


def test_hub_branch_uses_the_pinned_revision(tmp_path, monkeypatch):
    """With no local shards, the load must be the vanilla hub call, pinned."""
    import datasets

    calls = {}

    def fake_load_dataset(path, *args, **kwargs):
        calls["path"], calls["kwargs"] = path, kwargs
        return "sentinel"

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    out = load_split("validation", data_dir=str(tmp_path))
    assert out == "sentinel"
    assert calls["path"] == imagenet1k.HF_DATASET
    assert calls["kwargs"]["revision"] == imagenet1k.HF_REVISION
    assert calls["kwargs"]["split"] == "val"


def test_split_aliases():
    assert canonical_split("validation") == "val"
    with pytest.raises(ValueError):
        canonical_split("test")


def test_foreign_parquet_without_image_feature_is_refused(tmp_path):
    """Local shards not from the hub (no HF feature metadata) must hard-stop,
    not yield dicts where the callers expect decoded PILs."""
    out_dir = tmp_path / "imagenet1k_256" / "val"
    out_dir.mkdir(parents=True)
    table = pa.table({
        "image": [{"bytes": _jpeg_bytes(0), "path": None}],
        "label": [0],
    })
    pq.write_table(table, str(out_dir / "val-00000.parquet"))
    with pytest.raises(SystemExit, match="no HF image feature"):
        load_split("val", data_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"))


def test_verify_counts_and_decodes(tmp_path, monkeypatch):
    # verify() checks real-data plausibility (short side >= 200), so this
    # fixture writes full-size images.
    _write_shards(str(tmp_path), split="val", shards=(3, 2), image_size=256)
    monkeypatch.setattr(imagenet1k, "EXPECTED_ROWS", {"val": 5})
    cache = str(tmp_path / "cache")
    counts = imagenet1k.verify(str(tmp_path), splits=("val",), cache_dir=cache)
    assert counts == {"val": 5}
    monkeypatch.setattr(imagenet1k, "EXPECTED_ROWS", {"val": 6})
    with pytest.raises(SystemExit, match="expected 6"):
        imagenet1k.verify(str(tmp_path), splits=("val",), cache_dir=cache)


def test_revision_is_pinned():
    assert len(imagenet1k.HF_REVISION) == 40  # a commit hash, not a branch name