"""ImageNet-1k access: one pinned ``load_dataset`` call, shared by both examples.

Everything is vanilla HF ``datasets`` -- download, caching, Arrow conversion,
memory-mapped random access.  This module only holds what the library cannot
know for us:

* the dataset identity and the **pinned revision** (one home, so the examples
  cannot drift apart),
* a **local-shards branch**: if ``<data_dir>/imagenet1k_256/<split>/`` holds
  parquet shards, those are loaded instead of the hub -- the seam the test
  fixtures and air-gapped copies use,
* the **verify gate** (exact split sizes, sample decodes) run by prepare_data.

The Arrow cache follows ``HF_HOME``, which the runners point inside
``DATA_DIR``, so everything lands on the mounted data volume.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

HF_DATASET = "evanarlian/imagenet_1k_resized_256"
# The dataset repo's commit this artifact is built against (unchanged upstream
# since 2023-08); every load pins it so the bytes cannot drift under us.
HF_REVISION = "8de107e08d1f61884d5a4ec2dea8740667264a60"

EXPECTED_ROWS = {"train": 1_281_167, "val": 50_000}

_SPLIT_ALIASES = {"train": "train", "val": "val", "validation": "val"}


def canonical_split(split: str) -> str:
    try:
        return _SPLIT_ALIASES[str(split).lower()]
    except KeyError:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(_SPLIT_ALIASES)}")


def _local_shards(data_dir: Optional[str], split: str) -> list:
    root = data_dir or os.environ.get("DATA_DIR") or "./data"
    return sorted(glob.glob(os.path.join(root, "imagenet1k_256", split, "*.parquet")))


def locally_available(split: str = "val", data_dir: Optional[str] = None) -> bool:
    """True when load_split can serve the split without touching the network:
    local parquet shards under data_dir, or the pinned hub snapshot already in
    the HF cache. Figure panels probe this instead of letting a cache miss
    silently start the ~45 GB mirror download."""
    if _local_shards(data_dir, canonical_split(split)):
        return True
    hub = os.environ.get("HF_HUB_CACHE")
    if not hub:
        home = os.environ.get("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface")
        hub = os.path.join(home, "hub")
    snapshot = os.path.join(hub, "datasets--" + HF_DATASET.replace("/", "--"),
                            "snapshots", HF_REVISION)
    return os.path.isdir(snapshot)


def load_split(split: str, data_dir: Optional[str] = None, *,
               cache_dir: Optional[str] = None):
    """The split as a ``datasets.Dataset`` (decoded PIL under ``item["image"]``,
    int label under ``item["label"]``); the pinned hub dataset, unless local
    shards exist under ``data_dir``.

    Local shards are passed in sorted order, so dataset index -> image is a
    fixed mapping either way (the watermark uses the index as the per-image key).
    """
    import datasets

    split = canonical_split(split)
    paths = _local_shards(data_dir, split)
    if paths:
        ds = datasets.load_dataset("parquet", data_files=paths, split="train",
                                   cache_dir=cache_dir)
        if not isinstance(ds.features.get("image"), datasets.Image):
            raise SystemExit(
                f"local shards for {split!r} carry no HF image feature metadata "
                "-- they are not hub shards; remove them to load from the hub")
        return ds
    return datasets.load_dataset(HF_DATASET, revision=HF_REVISION, split=split,
                                 cache_dir=cache_dir)


def verify(data_dir: Optional[str] = None,
           splits: Tuple[str, ...] = ("train", "val"), *,
           cache_dir: Optional[str] = None) -> Dict[str, int]:
    """Hard-stop check that the data is complete and decodable.

    Counts must match the published split sizes exactly; a handful of sampled
    rows must decode to RGB images with in-range labels.  Constructing the
    datasets here also downloads (first use) and pre-builds the Arrow cache, so
    the first training run pays neither.
    """
    import numpy as np

    counts: Dict[str, int] = {}
    for split in splits:
        split = canonical_split(split)
        ds = load_split(split, data_dir=data_dir, cache_dir=cache_dir)
        if len(ds) != EXPECTED_ROWS[split]:
            raise SystemExit(
                f"imagenet1k {split}: {len(ds)} rows, expected "
                f"{EXPECTED_ROWS[split]} -- re-run prepare_data.py")
        rng = np.random.default_rng(0)
        for i in rng.integers(0, len(ds), size=8):
            item = ds[int(i)]
            img = item["image"].convert("RGB")
            if min(img.size) < 200 or not 0 <= item["label"] < 1000:
                raise SystemExit(
                    f"imagenet1k {split} row {i} failed the decode check "
                    f"(size {img.size}, label {item['label']})")
        counts[split] = len(ds)
    return counts
