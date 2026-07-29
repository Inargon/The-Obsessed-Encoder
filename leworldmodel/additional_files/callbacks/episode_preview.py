"""One-shot wandb videos of the first N training episodes — the exact frames
the encoder trains on, corner tag included when the run uses one."""

from __future__ import annotations

import lightning as pl
import wandb
from lightning.pytorch.loggers import WandbLogger


class EpisodePreviewCallback(pl.Callback):
    def __init__(self, dataset, num_episodes: int, fps: int = 7):
        self.dataset = dataset
        self.num_episodes = num_episodes
        self.fps = fps

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        wandb_loggers = [lg for lg in trainer.loggers if isinstance(lg, WandbLogger)]
        if not wandb_loggers:
            return
        # Lift the preprocessing transform for the reads: the tag is stamped
        # below the transform seam and stays in. Runs before the dataloaders
        # start iterating, so worker copies are unaffected.
        user_transform, self.dataset.transform = self.dataset.transform, None
        try:
            videos = {}
            for ep in range(min(self.num_episodes, len(self.dataset.lengths))):
                clip = self.dataset.load_episode(ep)["pixels"]  # (T, C, H, W) uint8
                videos[f"samples/train_episode_{ep}"] = wandb.Video(
                    clip.numpy(), fps=self.fps, format="gif"
                )
        finally:
            self.dataset.transform = user_transform
        for lg in wandb_loggers:
            lg.experiment.log(videos, step=0)
