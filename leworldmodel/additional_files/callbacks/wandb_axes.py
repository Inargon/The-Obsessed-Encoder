"""Pin off-cadence metric panels (eval/*, pair/*) to the true training step:
Lightning's WandbLogger lets wandb count log calls as its x-axis, which would
plot a metric logged every N steps against 0..num_ticks. Display only."""

from __future__ import annotations

from lightning.pytorch.loggers import WandbLogger


def pin_step_axis(trainer, *prefixes: str) -> None:
    for lg in trainer.loggers:
        if isinstance(lg, WandbLogger):
            for prefix in prefixes:
                lg.experiment.define_metric(f"{prefix}/*", step_metric="trainer/global_step")
