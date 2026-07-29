"""In-training goal-reaching eval (the eval.py recipe), logging
eval/success_rate in [0, 1]. World, dataset and policy build once on the
first tick; the solver holds a live reference to the training model.

Each arm evaluates on the kind of data it trained on: randgoal points
eval.dataset_name at the scripted dataset, and the square arms set
eval.tag_mode so the planner sees tagged frames — planning on untagged
frames would evaluate those encoders off-distribution.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as tvt

from .wandb_axes import pin_step_axis
from ..pixel_tag import PixelTag


class EvalTagStamp:
    """Stamp the training corner tag onto a 3-channel uint8 eval frame before
    Normalize/Resize, in either layout: (H, W, 3) as read from disk, or
    (3, H, W) as WorldModelPolicy._prepare_info delivers frames to the
    transform Compose (it permutes channel-first BEFORE the transforms run).
    video holds one fixed colour across the eval (constant within a rollout,
    as in training); frame draws fresh per frame. Anything that is not a
    3-channel uint8 image raises -- a pass-through here would silently
    evaluate tag-trained encoders on untagged frames.
    """

    _VIDEO_KEY = 0  # reserved eval episode key, distinct from real episodes

    def __init__(self, tag: PixelTag):
        self.tag = tag
        self._step = 0

    def _color(self):
        if self.tag.mode == "video":
            return self.tag.color_for(self._VIDEO_KEY)
        color = self.tag.color_for(self._VIDEO_KEY, self._step)
        self._step += 1
        return color

    def __call__(self, img):
        m = self.tag.size
        # Type must be preserved: the downstream ToImage permutes numpy
        # (assumed HWC) but not tensors (assumed CHW) -- returning the wrong
        # kind double-permutes the frame.
        if torch.is_tensor(img):
            if img.dtype != torch.uint8 or img.ndim != 3 or img.shape[0] != 3:
                raise TypeError(
                    f"EvalTagStamp expects a (3, H, W) uint8 tensor, got "
                    f"dtype={img.dtype} shape={tuple(img.shape)}")
            out = img.clone()
            out[:, :m, :m] = torch.as_tensor(
                self._color(), dtype=torch.uint8).view(3, 1, 1)
            return out
        arr = np.asarray(img)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise TypeError(
                f"EvalTagStamp expects an (H, W, 3) uint8 frame, got "
                f"dtype={arr.dtype} shape={arr.shape}")
        arr = arr.copy()
        arr[:m, :m, :] = self._color()
        return arr


def eval_transforms(img_size: int, tag: PixelTag | None) -> dict:
    """Per-stream eval transforms, tag stamp prepended when tagging is on
    (a fresh stamp per stream, so frame-mode step counters don't interleave)."""

    def one():
        stamp = [EvalTagStamp(tag)] if tag is not None else []
        base = [
            tvt.ToImage(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.Normalize(**spt.data.dataset_stats.ImageNet),
            tvt.Resize(size=img_size),
        ]
        return tvt.Compose(stamp + base)

    return {"pixels": one(), "goal": one()}


class GoalEvalCallback(pl.Callback):
    """Config block: every_n_steps (0/absent = off), dataset_name, config,
    solver, num_eval, tag_mode/tag_size/tag_seed (see module docstring)."""

    def __init__(self, eval_cfg, seed: int, default_dataset: str | None = None):
        self.cfg_block = eval_cfg
        self.seed = seed
        self.default_dataset = default_dataset
        self.every = int(eval_cfg.get("every_n_steps", 0))
        self._world = None
        self._last_step = -1

    def _tag(self) -> PixelTag | None:
        mode = self.cfg_block.get("tag_mode")
        if not mode:
            return None
        return PixelTag(
            mode=mode,
            size=int(self.cfg_block.get("tag_size", 5)),
            seed=int(self.cfg_block.get("tag_seed", self.seed)),
        )

    def _setup(self, model):
        cfg_dir = Path(__file__).resolve().parents[2] / "config" / "eval"
        cfg = OmegaConf.load(cfg_dir / (self.cfg_block.get("config", "pusht") + ".yaml"))
        cfg.solver = OmegaConf.load(
            cfg_dir / "solver" / (self.cfg_block.get("solver", "cem") + ".yaml")
        )
        cfg.pop("defaults", None)
        # Precedence: the arm's explicit override, else the training dataset
        # (each arm evaluates on its own distribution). The upstream eval
        # yaml's dataset_name is a legacy extensionless value; never used.
        cfg.eval.dataset_name = (self.cfg_block.get("dataset_name")
                                 or self.default_dataset
                                 or cfg.eval.dataset_name)
        if self.cfg_block.get("num_eval"):
            cfg.eval.num_eval = int(self.cfg_block.get("num_eval"))
        cfg.world.num_envs = cfg.eval.num_eval
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
        self.cfg = cfg

        cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
        dataset = swm.data.load_dataset(
            cfg.eval.dataset_name,
            cache_dir=cache_dir,
            keys_to_cache=cfg.dataset.keys_to_cache,
        )

        # goal columns share their observation counterpart's scaler
        process = {}
        for col in cfg.dataset.keys_to_cache:
            if col == "pixels":
                continue
            data = dataset.get_col_data(col)
            process[col] = preprocessing.StandardScaler().fit(data[~np.isnan(data).any(axis=1)])
            if col != "action":
                process[f"goal_{col}"] = process[col]

        # fixed eval group: (episode, start) pairs with goal_offset_steps of
        # runway. The episode column name differs across dataset generations,
        # and column_names lists only the data keys on some formats -- resolve
        # by asking the table itself.
        for ep_col in ("episode_idx", "ep_idx"):
            try:
                episode_idx = dataset.get_col_data(ep_col)
                break
            except Exception:
                episode_idx = None
        if episode_idx is None:
            raise ValueError("dataset has neither an 'episode_idx' nor an "
                             "'ep_idx' column")
        step_idx = dataset.get_col_data("step_idx")
        last = {ep: step_idx[episode_idx == ep].max() for ep in np.unique(episode_idx)}
        max_start = np.array([last[ep] - cfg.eval.goal_offset_steps for ep in episode_idx])
        valid = np.nonzero(step_idx <= max_start)[0]
        rng = np.random.default_rng(cfg.seed)
        # dataset row indexing requires monotonic indices
        rows = np.sort(valid[rng.choice(len(valid), size=cfg.eval.num_eval, replace=False)])
        # Index the full column arrays directly: get_row_data exposes only the
        # data keys on some formats (the index columns are not in its payload).
        self._episodes = np.asarray(episode_idx[rows]).tolist()
        self._starts = np.asarray(step_idx[rows]).tolist()

        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(**cfg.plan_config),
            process=process,
            transform=eval_transforms(int(cfg.eval.img_size), self._tag()),
        )
        self._world = swm.World(**cfg.world, image_shape=(224, 224))
        self._world.set_policy(policy)
        self._align_destinations(dataset, rows)
        self._dataset = dataset

    def _align_destinations(self, dataset, rows) -> None:
        """Draw each env's destination T where its dataset episode had it.

        Reset never samples the goal variation, so every env would render the
        default centre destination while the goal frame it plans against shows
        the episode's own. On a randomized-destination dataset that contradiction
        is off-distribution for the planner, and no reachable state can produce
        the goal frame. Datasets with a fixed destination have no goal_pose
        column and already agree with the env default.
        """
        try:
            poses = np.asarray(dataset.get_col_data("goal_pose"))[rows]
        except Exception:
            return
        for env, pose in zip(self._world.envs.envs, poses):
            goal = env.unwrapped.variation_space["goal"]
            goal["position"].set_init_value(np.asarray(pose[:2], dtype=np.float64))
            goal["angle"].set_init_value(float(pose[2]))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        # skip step 0, off-cadence steps, and grad-accumulation repeat fires
        if step == 0 or step % self.every or step == self._last_step:
            return
        self._last_step = step
        if self._world is None:
            pin_step_axis(trainer, "eval")
            self._setup(pl_module.model)
        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                metrics = self._world.evaluate(
                    dataset=self._dataset,
                    episodes_idx=self._episodes,
                    start_steps=self._starts,
                    goal_offset=self.cfg.eval.goal_offset_steps,
                    eval_budget=self.cfg.eval.eval_budget,
                    callables=OmegaConf.to_container(self.cfg.eval.get("callables"), resolve=True),
                )
        finally:
            if was_training:
                pl_module.train()
        # World.evaluate reports percent; the logged metric is a rate in [0, 1].
        rate = float(metrics["success_rate"]) / 100.0
        print(f"[eval@{step}] success_rate={rate:.3f}")
        for lg in trainer.loggers:
            lg.log_metrics({"eval/success_rate": rate}, step=step)
