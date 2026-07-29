"""Fetch the BTS NYU-Depth-v2 protocol data (not rehosted here).

The dense probe follows Meta's DINOv3 depth protocol exactly, and that protocol
is defined on the BTS distribution of NYU: the 24,231-pair train manifest and
the 654-image Eigen test split with RAW ground-truth depth.  The widely-mirrored
h5 conversion of NYU is a different dataset (different train composition,
filled instead of raw test depth), so its numbers are not comparable to the
published benchmark -- exactly the kind of silent confounder this repo exists
to exclude -- and the data gate in dense_eval.py refuses it.

This script automates the re-fetch recipe from DINOv3's DATASETS.md (the single
Google Drive archive, ~6.7 GB) into ``DATA_DIR/nyu_depth_v2_bts/`` and runs the
gate on the result:

    uv run python dinov3/additional_files/fetch_nyu.py

The download requires ``gdown`` (in the lockfile).  If Google Drive rate-limits
the fetch, download the archive in a browser instead
(https://drive.google.com/file/d/{GDRIVE_ID}/view), then:

    uv run python dinov3/additional_files/fetch_nyu.py --archive /path/to/nyu.zip
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile

_VENDORED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACT_ROOT = os.path.dirname(_VENDORED_ROOT)
for _p in (_ARTIFACT_ROOT, _VENDORED_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from additional_files.dense_eval import default_data_root, nyu_data_gate  # noqa: E402

# The DATASETS.md "Option 2" archive (rgb + uint16-mm depth PNGs + manifests).
GDRIVE_ID = "1xI9ksHzCC_kUz6Z4FL_b1ttgj3RVHGwW"


def _download(archive_path: str) -> None:
    import gdown

    print(f"downloading the BTS NYU archive (~6.7 GB) to {archive_path} ...")
    out = gdown.download(id=GDRIVE_ID, output=archive_path, resume=True)
    if out is None:
        raise SystemExit(
            "gdown could not fetch the archive (Google Drive quota?). Download "
            f"https://drive.google.com/file/d/{GDRIVE_ID}/view manually and re-run "
            "with --archive <path>.")


def _extract(archive_path: str, data_root: str) -> None:
    staging = data_root + ".extract"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    print(f"extracting into {staging} ...")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(staging)
    # The archive unpacks to a single `nyu/` directory holding the manifests and
    # scene folders; that directory becomes the data root.
    inner = os.path.join(staging, "nyu")
    source = inner if os.path.isdir(inner) else staging
    if not os.path.exists(os.path.join(source, "nyu_train.txt")):
        raise SystemExit(f"unexpected archive layout under {staging}; "
                         "expected nyu_train.txt at the top of the extracted tree")
    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    os.replace(source, data_root)
    if os.path.exists(staging):
        shutil.rmtree(staging)


def fetch(data_root: str = None, archive: str = None,
          keep_archive: bool = False) -> str:
    """Idempotent fetch-and-gate; also the seam prepare_data.py calls."""
    data_root = os.path.abspath(data_root or default_data_root())
    if os.path.exists(os.path.join(data_root, "nyu_train.txt")):
        print(f"{data_root} already populated; running the gate only")
        print(f"BTS data gate passed: {nyu_data_gate(data_root)}")
        return data_root

    os.makedirs(os.path.dirname(data_root), exist_ok=True)
    downloaded = False
    if archive is None:
        archive = data_root + ".zip"
        _download(archive)
        downloaded = True
    _extract(archive, data_root)
    if downloaded and not keep_archive:
        os.remove(archive)
    print(f"BTS data gate passed: {nyu_data_gate(data_root)}")
    print(f"NYU protocol data ready at {data_root}")
    return data_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None,
                        help="target directory; default DATA_DIR/nyu_depth_v2_bts")
    parser.add_argument("--archive", default=None,
                        help="already-downloaded nyu.zip (skips the Drive fetch)")
    parser.add_argument("--keep-archive", action="store_true",
                        help="keep the downloaded zip after extraction")
    args = parser.parse_args()
    fetch(args.data_root, archive=args.archive, keep_archive=args.keep_archive)


if __name__ == "__main__":
    main()
