import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback

# >>> obsessed-encoder: everything we add lives in additional_files; the
# config blocks that activate each piece are documented in additional_files/README.md
from additional_files import callbacks
from additional_files.allocation_regularizers import AllocationRegularizer
from additional_files.control_objectives import ControlObjective
from additional_files.pixel_tag import attach_pixel_tag, tag_from_cfg
# <<< obsessed-encoder


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    # Opt-in relation-core ablation. A value below one tests whether the
    # absolute next-latent objective is what keeps paying episode-constant
    # shortcuts even when a reachability objective is present. Absent from
    # upstream configs, this remains exactly 1.0.
    pred_weight = float(cfg.loss.get("pred_weight", 1.0))
    output["loss"] = pred_weight * output["pred_loss"] + lambd * output["sigreg_loss"]

    # Opt-in capacity-allocation experiment. The published arms have no
    # ``loss.allocation`` block and therefore retain the exact upstream loss.
    if hasattr(self, "allocation_reg"):
        allocation = self.allocation_reg(emb)
        output.update(allocation)
        output["loss"] = output["loss"] + output["allocation_loss"]

    # Control-sufficiency experiments attach their trainable heads inside the
    # world model so the existing ``model_opt`` optimizer owns their parameters.
    if hasattr(self.model, "control_objective"):
        control = self.model.control_objective(emb, batch["action"], pred_emb)
        output.update(control)
        output["loss"] = output["loss"] + output["control_loss"]

    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k in {
            "local_effective_rank",
            "reachability_accuracy",
            "reachability_shuffled_accuracy",
            "reachability_action_margin",
        }
    }
    self.log_dict(metrics_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # >>> obsessed-encoder: opt-in full seeding for reproducible runs; absent,
    # upstream's behavior stands (only the train/val split generator is seeded)
    if cfg.get("seed_everything"):
        pl.seed_everything(cfg.seed, workers=True)
    # <<< obsessed-encoder

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    # >>> obsessed-encoder: opt-in on-the-fly corner colour tag; with no
    # pixel_tag config the reader stays exactly upstream's (docs: additional_files/README.md)
    tag_cfg = cfg.get("pixel_tag")
    if tag_cfg and tag_cfg.get("enabled", True):
        dataset = attach_pixel_tag(dataset, tag_from_cfg(tag_cfg, cfg.seed))
    # <<< obsessed-encoder
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    control_cfg = cfg.loss.get("control")
    if control_cfg and control_cfg.get("enabled", True):
        control_kwargs = OmegaConf.to_container(control_cfg, resolve=True)
        control_kwargs.pop("enabled", None)
        world_model.control_objective = ControlObjective(
            embed_dim=cfg.embed_dim,
            action_dim=cfg.model.action_encoder.input_dim,
            **control_kwargs,
        )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module_kwargs = {}
    allocation_cfg = cfg.loss.get("allocation")
    if allocation_cfg and allocation_cfg.get("enabled", True):
        allocation_kwargs = OmegaConf.to_container(allocation_cfg, resolve=True)
        allocation_kwargs.pop("enabled", None)
        module_kwargs["allocation_reg"] = AllocationRegularizer(**allocation_kwargs)

    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
        **module_kwargs,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    # >>> obsessed-encoder: local metrics mirror — every logged payload is also
    # appended to a metrics.jsonl, the file the figures render from; with
    # metrics_jsonl_path absent, upstream's logger value passes through untouched.
    # The wandb stream itself is renamed to the repo-wide unified sections.
    callbacks.unify_wandb_keys(logger)
    trainer_logger = callbacks.jsonl_mirror(cfg, logger)
    # <<< obsessed-encoder

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    # >>> obsessed-encoder: opt-in callbacks, one per row; each factory returns
    # an empty list unless its config block is set (docs: additional_files/README.md)
    extra_callbacks = [
        *callbacks.goal_eval(cfg),
        *callbacks.pair_analysis(cfg),
        *callbacks.episode_preview(cfg, dataset),
        *callbacks.step_checkpoint(cfg),
    ]
    # <<< obsessed-encoder

    trainer = pl.Trainer(
        **cfg.trainer,
        # >>> obsessed-encoder: attach the opt-in callbacks/loggers built above
        # (with their configs absent this is exactly the upstream call)
        callbacks=[object_dump_callback] + extra_callbacks,
        num_sanity_val_steps=1,
        logger=trainer_logger,
        # <<< obsessed-encoder
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
