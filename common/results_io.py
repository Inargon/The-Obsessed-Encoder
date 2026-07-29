"""Local run files -- the source of truth every figure renders from.

Each run directory (``RESULTS_DIR/<config>_seed<k>/``) holds:

* ``metrics.jsonl`` -- one JSON object per logged payload, each carrying ``step``.
  Every metric a run reports is mirrored here; wandb is optional monitoring.
* ``summary.json`` -- written once, at successful completion only.  Its presence
  is the done-marker the runner's skip logic keys on, so a failed or interrupted
  run must never leave one behind.  It carries the provenance stamp (software
  versions, GPU, git commit, lockfile hash) alongside the run's final numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional

METRICS_NAME = "metrics.jsonl"
SUMMARY_NAME = "summary.json"


def metrics_path(run_dir: str) -> str:
    return os.path.join(run_dir, METRICS_NAME)


def summary_path(run_dir: str) -> str:
    return os.path.join(run_dir, SUMMARY_NAME)


def append_metrics(run_dir: str, payload: Dict[str, Any]) -> None:
    """Append one payload as a JSON line; creates the run dir on first write.

    A run killed mid-write leaves a newline-less partial last line; a resumed
    run appending straight after it would fuse the two into one unparseable
    interior line, so the write repairs the boundary first.
    """
    os.makedirs(run_dir, exist_ok=True)
    with open(metrics_path(run_dir), "ab+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                # Drop the partial line back to the previous newline: the
                # payload was never fully recorded, and read_metrics treats a
                # newline-terminated corrupt line as fatal.
                chunk = min(size, 1 << 20)
                f.seek(size - chunk)
                cut = f.read(chunk).rfind(b"\n")
                f.truncate(size - chunk + cut + 1 if cut != -1 else 0)
                f.seek(0, os.SEEK_END)
        # default=float: framework loggers hand numpy / tensor scalars through.
        f.write((json.dumps(payload, default=float) + "\n").encode())


def read_metrics(run_dir: str) -> List[Dict[str, Any]]:
    """All payloads of a run, in log order.

    A run killed mid-write can leave one truncated final line -- recognised by
    the missing trailing newline -- and only that line is dropped (the payload
    was never fully recorded).  A malformed line anywhere else, or a corrupt
    newline-terminated final line, is an error: that file was not written by
    this module.
    """
    path = metrics_path(run_dir)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        raw = f.read()
    lines = raw.splitlines()
    truncated_tail = bool(raw) and not raw.endswith("\n")
    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 and truncated_tail:
                break
            raise
    return rows


def iter_run_dirs(results_dir: str) -> Iterator[str]:
    """Run directories under a results root (any dir holding a metrics.jsonl)."""
    if not os.path.isdir(results_dir):
        return
    for name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, name)
        if os.path.isdir(run_dir) and os.path.exists(metrics_path(run_dir)):
            yield run_dir


def has_summary(run_dir: str) -> bool:
    return os.path.exists(summary_path(run_dir))


def read_summary(run_dir: str) -> Optional[Dict[str, Any]]:
    if not has_summary(run_dir):
        return None
    with open(summary_path(run_dir)) as f:
        return json.load(f)


def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _lockfile_hash() -> Optional[str]:
    lock = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "uv.lock")
    if not os.path.exists(lock):
        return None
    with open(lock, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def provenance() -> Dict[str, Any]:
    """The environment stamp for summary.json: enough for a reader to check the
    published numbers came from the pinned environment (every field is best-effort
    -- a missing GPU or git checkout yields nulls, never a failure)."""
    import platform

    import torch

    commit = _run(["git", "rev-parse", "HEAD"])
    stamp: Dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": None,
        "driver": _run(["nvidia-smi", "--query-gpu=driver_version",
                        "--format=csv,noheader"]),
        "git_commit": commit,
        # A commit hash from a dirty tree is not provenance; record the taint
        # (null when there is no git checkout to ask).
        "git_dirty": (bool(_run(["git", "status", "--porcelain"]))
                      if commit else None),
        "lockfile_sha256": _lockfile_hash(),
    }
    if torch.cuda.is_available():
        stamp["gpu"] = torch.cuda.get_device_name(0)
    return stamp


def write_summary(run_dir: str, payload: Dict[str, Any]) -> None:
    """Write the done-marker atomically (tmp + rename), stamping provenance and a
    completion timestamp.  Call this ONLY after the run finished successfully."""
    os.makedirs(run_dir, exist_ok=True)
    doc = dict(payload)
    doc.setdefault("provenance", provenance())
    doc.setdefault("finished_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    tmp = summary_path(run_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    os.replace(tmp, summary_path(run_dir))


def metric_series(rows: List[Dict[str, Any]], key: str) -> tuple:
    """(steps, values) for one metric key, in logged order (the per-case graph
    scripts' reading primitive)."""
    steps, vals = [], []
    for row in rows:
        if key in row:
            steps.append(int(row.get("step", len(steps))))
            vals.append(float(row[key]))
    return steps, vals


def results_dir():
    """The results root the environment selects (the Docker volume convention)."""
    from pathlib import Path

    return Path(os.environ.get("RESULTS_DIR", "./results")).resolve()
