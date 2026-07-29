"""Lightning logger appending every payload to metrics.jsonl — the file the
figures render from, so wandb stays optional monitoring."""

from __future__ import annotations

import sys
from pathlib import Path

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only

try:
    from common.results_io import METRICS_NAME, append_metrics
except ImportError:  # by-hand run without PYTHONPATH
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from common.results_io import METRICS_NAME, append_metrics


class JsonlLogger(Logger):
    def __init__(self, path: str):
        super().__init__()
        path = Path(path)
        if path.name != METRICS_NAME:
            raise ValueError(f"metrics_jsonl_path must end in {METRICS_NAME}; got {path}")
        self.run_dir = str(path.parent)

    @property
    def name(self):
        return "jsonl"

    @property
    def version(self):
        return ""

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        append_metrics(self.run_dir, {"step": int(step) if step is not None else None, **metrics})

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        pass
