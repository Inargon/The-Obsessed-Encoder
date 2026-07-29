"""Dataset preparation for the LeJEPA example.

* ``imagenet1k`` -- pre-downloads the revision-pinned
  ``evanarlian/imagenet_1k_resized_256`` dataset into the HF cache under
  DATA_DIR, builds the Arrow cache, and verifies counts and sample decodes.
  ~45 GB on disk; needs a Hugging Face connection.
* ``imagenette`` -- pre-downloads the frgfm/imagenette 160px parquet revision
  into the HF cache under DATA_DIR (the quickstart dataset; the training
  process would download it on first use anyway).

    uv run python lejepa/additional_files/prepare_data.py --dataset imagenet1k
"""
from __future__ import annotations

import argparse
import os
import sys

ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ARTIFACT_ROOT not in sys.path:
    sys.path.insert(0, ARTIFACT_ROOT)


def prepare_imagenette(data_dir: str) -> None:
    os.environ.setdefault("HF_HOME", os.path.join(data_dir, "hf_cache"))
    from datasets import load_dataset

    for split, expected in (("train", 9469), ("validation", 3925)):
        ds = load_dataset(
            "frgfm/imagenette",
            revision="refs/convert/parquet",
            data_files={
                "train": "160px/train/*.parquet",
                "validation": "160px/validation/*.parquet",
            },
            split=split,
        )
        if len(ds) != expected:
            raise SystemExit(f"imagenette {split}: {len(ds)} rows, expected {expected}")
        print(f"imagenette {split}: {len(ds)} rows ok")


def prepare_imagenet1k(data_dir: str) -> None:
    # Everything (hub download of the pinned revision + Arrow cache) lands under
    # DATA_DIR via HF_HOME; verify() triggers both, so the first training run
    # pays neither.
    os.environ.setdefault("HF_HOME", os.path.join(data_dir, "hf_cache"))
    from common.imagenet1k import verify

    counts = verify(data_dir)
    print(f"imagenet1k ok: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("imagenet1k", "imagenette", "all"),
                        default="imagenet1k")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    if args.dataset in ("imagenette", "all"):
        prepare_imagenette(data_dir)
    if args.dataset in ("imagenet1k", "all"):
        prepare_imagenet1k(data_dir)


if __name__ == "__main__":
    main()
