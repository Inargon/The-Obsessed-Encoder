"""Shared grid-runner semantics for the per-example ``additional_files/run.py``.

One run = one printed, human-runnable subprocess; results live under
``RESULTS_DIR/<config>_seed<k>/`` with ``summary.json`` as the done-marker.  A
run directory that already holds a summary is skipped (``--force`` reruns
everything, ``--rerun <run> ...`` reruns the named ones); the ``--gpus`` pool
dispatches queued runs onto local GPUs via ``CUDA_VISIBLE_DEVICES``.  A failed
subprocess never writes a summary and makes the whole invocation exit nonzero.
"""
from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from common import results_io


@dataclass
class RunSpec:
    name: str                       # <config>_seed<k>
    config: str
    seed: int
    command: List[str]
    env: Dict[str, str] = field(default_factory=dict)  # extra env, printed with the command
    cwd: Optional[str] = None


def add_common_arguments(parser: argparse.ArgumentParser, *, default_tag: str) -> None:
    parser.add_argument("--configs", default="clean,watermarked,random_control",
                        help="comma-separated config names to run")
    parser.add_argument("--seeds", default="3",
                        help="seed count (e.g. 3 -> seeds 0,1,2) or an explicit "
                             "comma-separated list (e.g. 0,2)")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "./results"),
                        help="root for run directories and figures")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"),
                        help="dataset root (see prepare_data.py)")
    parser.add_argument("--gpus", default="0",
                        help="comma-separated local GPU ids; runs are dispatched "
                             "onto them via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--tag", default=default_tag,
                        help="invocation tag: wandb group and run-name suffix context")
    parser.add_argument("--force", action="store_true",
                        help="rerun every run even if its summary.json exists")
    parser.add_argument("--rerun", nargs="*", default=[],
                        help="run names (<config>_seed<k>) to rerun even if complete")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip all runs; render the figures from local results")
    parser.add_argument("--dry-run", action="store_true",
                        help="print every run's command without executing anything")


def parse_seeds(spec: str) -> List[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) == 1 and "," not in spec:
        count = int(parts[0])
        if count < 1:
            raise SystemExit(
                "--seeds takes a count >= 1 (e.g. 3 -> seeds 0,1,2) or an "
                "explicit comma list (a single seed: '2,')")
        return list(range(count))
    return [int(p) for p in parts]


def run_dir_for(results_dir: str, name: str) -> str:
    return os.path.join(os.path.abspath(results_dir), name)


def clear_run_dir(results_dir: str, name: str) -> None:
    """Reset a run directory for a from-scratch launch.

    A restarted run must not inherit the previous attempt's files: appended
    metrics.jsonl rows would blend two trajectories into the figures, and a
    stale checkpoint would turn a "rerun" into a silent resume.
    """
    import shutil

    run_dir = run_dir_for(results_dir, name)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)


def should_run(results_dir: str, spec: RunSpec, *, force: bool,
               rerun: Sequence[str]) -> bool:
    if force or spec.name in rerun:
        return True
    return not results_io.has_summary(run_dir_for(results_dir, spec.name))


def format_command(spec: RunSpec, gpu: Optional[str] = None) -> str:
    """The human-runnable form of a run: env assignments + argv, one line."""
    env = dict(spec.env)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
    argv = " ".join(shlex.quote(a) for a in spec.command)
    return f"{assignments} {argv}".strip()


def launch(spec: RunSpec, gpu: Optional[str]) -> int:
    env = dict(os.environ)
    env.update(spec.env)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"[run] {spec.name} (gpu {gpu}):\n  {format_command(spec, gpu)}", flush=True)
    proc = subprocess.run(spec.command, env=env, cwd=spec.cwd)
    return proc.returncode


def execute_pool(specs: Sequence[RunSpec], gpus: Sequence[str], *,
                 launch_fn: Callable[[RunSpec, Optional[str]], int] = launch,
                 on_success: Optional[Callable[[RunSpec], None]] = None) -> Dict[str, int]:
    """Dispatch runs onto the GPU pool; one run per GPU at a time.

    Returns {run name: exit code}.  ``on_success`` fires after a zero exit --
    the per-example runner writes the summary (and runs any per-run tail) there.
    """
    if specs and not gpus:
        # An empty pool with work pending would return {} and read as success.
        raise SystemExit("--gpus resolved to an empty list but runs are pending")
    todo: "queue.Queue[RunSpec]" = queue.Queue()
    for spec in specs:
        todo.put(spec)
    results: Dict[str, int] = {}
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                spec = todo.get_nowait()
            except queue.Empty:
                return
            try:
                code = launch_fn(spec, gpu)
            except Exception:  # a raising launcher must surface as a failed
                import traceback  # run, never as a silently missing result

                traceback.print_exc()
                code = 1
            if code == 0 and on_success is not None:
                try:
                    on_success(spec)
                except Exception as e:  # a failed tail must not read as success
                    print(f"[run] {spec.name}: post-run step failed: {e}", flush=True)
                    code = 1
            with lock:
                results[spec.name] = code
            if code != 0:
                print(f"[run] {spec.name} FAILED (exit {code}); no summary written",
                      flush=True)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def print_dry_run(specs: Sequence[RunSpec], gpus: Sequence[str]) -> None:
    """--dry-run: the exact commands the pool would launch, nothing executed."""
    for spec in specs:
        gpu = gpus[0] if gpus else None
        print(f"[dry-run] {spec.name}:\n  {format_command(spec, gpu)}")


def summarize_results(results: Dict[str, int]) -> int:
    """Print the outcome table; return the process exit code (1 if anything failed)."""
    failed = sorted(name for name, code in results.items() if code != 0)
    done = sorted(name for name, code in results.items() if code == 0)
    for name in done:
        print(f"[run] {name}: complete")
    for name in failed:
        print(f"[run] {name}: FAILED")
    return 1 if failed else 0
