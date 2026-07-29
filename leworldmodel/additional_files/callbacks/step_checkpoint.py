"""weights_step_<N>.pt saves through the upstream save_pretrained, so
intra-epoch checkpoints load exactly like the per-epoch ones."""

from __future__ import annotations

import lightning as pl
from stable_worldmodel.wm.utils import save_pretrained


class StepCheckpointCallback(pl.Callback):
    def __init__(self, every_n_steps: int, run_name: str, model_cfg):
        self.every = every_n_steps
        self.run_name = run_name
        self.model_cfg = model_cfg
        self._last_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        # skip step 0, off-cadence steps, and grad-accumulation repeat fires
        if step == 0 or step % self.every or step == self._last_step:
            return
        self._last_step = step
        if trainer.is_global_zero:
            save_pretrained(
                pl_module.model,
                run_name=self.run_name,
                config=self.model_cfg,
                filename=f"weights_step_{step}.pt",
            )
