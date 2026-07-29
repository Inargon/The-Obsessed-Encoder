"""The opt-in training callbacks, one factory per feature.

Each factory reads its own config block and returns [] when the block is
absent — so train.py's marked block is a plain list, one row per callback,
and the default (no config) run attaches nothing and stays exactly upstream.
"""

from __future__ import annotations

# Relative imports on purpose: two examples ship a package named
# additional_files, and absolute imports resolve to whichever the interpreter
# cached first (root-level pytest caches dinov3's, breaking collection here).
from .episode_preview import EpisodePreviewCallback
from .goal_eval import GoalEvalCallback
from .jsonl_logger import JsonlLogger
from .pair_analysis import PairAnalysisCallback
from .step_checkpoint import StepCheckpointCallback


def goal_eval(cfg):
    """eval.every_n_steps → in-training planning eval, eval/success_rate in [0, 1]."""
    block = cfg.get("eval")
    if not (block and block.get("every_n_steps")):
        return []
    # The eval dataset defaults to the TRAINING dataset (each arm evaluates on
    # its own distribution); upstream's eval yaml carries an extensionless
    # legacy name the resolver refuses.
    return [GoalEvalCallback(block, cfg.seed,
                             default_dataset=cfg.data.dataset.name)]


def pair_analysis(cfg):
    """pair.every_n_steps → the pair-cosine suites (colour / t_position)."""
    block = cfg.get("pair")
    if not (block and block.get("every_n_steps")):
        return []
    tag_geom = cfg.get("pixel_tag") or {}
    return [
        PairAnalysisCallback(
            int(block.every_n_steps),
            num_pairs=int(block.get("num_pairs", 256)),
            img_size=cfg.img_size,
            tag_size=int(tag_geom.get("size", 5)),
            tag_seed=int(tag_geom.get("seed", cfg.seed)),
            suites=[str(s) for s in block.get("suites", ["colour"])],
        )
    ]


def episode_preview(cfg, dataset):
    """preview.num_episodes → one-shot wandb videos of example training episodes."""
    block = cfg.get("preview")
    if not (block and block.get("num_episodes")):
        return []
    return [EpisodePreviewCallback(dataset, int(block.num_episodes))]


def step_checkpoint(cfg):
    """checkpoint.every_n_steps → weights_step_<N>.pt in the upstream save format."""
    block = cfg.get("checkpoint")
    if not (block and block.get("every_n_steps")):
        return []
    return [StepCheckpointCallback(int(block.every_n_steps), cfg.output_model_name, cfg.model)]


def jsonl_mirror(cfg, logger):
    """metrics_jsonl_path → wandb (when enabled) plus a local metrics.jsonl mirror;
    absent, upstream's logger value passes through untouched."""
    path = cfg.get("metrics_jsonl_path")
    if not path:
        return logger
    return ([logger] if logger is not None else []) + [JsonlLogger(str(path))]


def unify_wandb_keys(logger):
    """Rebind the wandb logger's log_metrics so the wandb stream uses the
    repo-wide unified section names (common.wandb_keys). Instance-level rebind
    rather than a wrapper class: stable-pretraining locates the logger with
    isinstance(WandbLogger) checks, which a wrapper would defeat. The jsonl
    mirror is a separate logger and keeps the upstream keys."""
    if logger is None:
        return
    from common import wandb_keys

    original = logger.log_metrics

    def renamed(metrics, step=None):
        original(wandb_keys.remap(metrics, wandb_keys.LEWM), step)

    logger.log_metrics = renamed
