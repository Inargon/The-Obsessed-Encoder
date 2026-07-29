"""Dataset preparation for the DINOv3 example -- everything the runs need.

* ImageNet-1k: the same revision-pinned hub dataset as the LeJEPA example
  (downloads on first use, ~45 GB incl. Arrow cache; verifies thereafter).
* NYU-Depth-v2 (BTS protocol data, needed by the dense probe): fetched when
  absent (a ~6.7 GB Google Drive pull; ImageNet is finished first, so a Drive
  hiccup costs nothing -- re-run to resume, or use fetch_nyu.py directly,
  which also takes a browser-downloaded archive via --archive).

    uv run python dinov3/additional_files/prepare_data.py
    uv run python dinov3/additional_files/prepare_data.py --skip-nyu
"""
from __future__ import annotations

import argparse
import os
import sys

_VENDORED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACT_ROOT = os.path.dirname(_VENDORED_ROOT)
for _p in (_ARTIFACT_ROOT, _VENDORED_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    parser.add_argument("--skip-nyu", action="store_true",
                        help="prepare ImageNet-1k only (the dense probe needs NYU)")
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    os.environ.setdefault("HF_HOME", os.path.join(data_dir, "hf_cache"))
    from common.imagenet1k import verify

    # Despite the name, this is the preparation step too: constructing the
    # datasets downloads on first use (~45 GB) and pre-builds the Arrow cache;
    # with the data already present it is a pure verification pass.
    counts = verify(data_dir)
    print(f"imagenet1k ready (downloaded on first use, verified): {counts}")

    if args.skip_nyu:
        print("NYU skipped (--skip-nyu); the dense probe needs it")
        return
    from additional_files.fetch_nyu import fetch

    fetch(os.path.join(data_dir, "nyu_depth_v2_bts"))


if __name__ == "__main__":
    main()
