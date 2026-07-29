"""RunObserver -- the measurement layer bolted onto DINOv3's do_train.

The open-source DINOv3 release stripped the benchmark harness and has no
wandb, so every signal this example reports is net-new and lives here, off to
the side of the training loop.  do_train constructs one RunObserver
(env-gated) and calls a few thin hooks; everything else -- the local
metrics.jsonl mirror, optional wandb, the passive online probe, the rotating
val pass with the pair metrics, the step-0 stimulus dump -- is contained in
this module.

Single-GPU by design.  Anything that could fail (the val featurization
touching the compiled/FSDP teacher) is wrapped so it logs a warning and
self-disables rather than killing a multi-day run.

Env seam (set per run by additional_files/run.py; the printed command shows it):

| Env | Default | Meaning |
|-----|---------|---------|
| ``DINOV3_PROBE`` | unset | ``1`` enables the label collate + this observer |
| ``DINOV3_PROBE_LR`` | ``1e-3`` | online probe AdamW lr |
| ``DINOV3_VAL_PERIOD`` | ``500`` | iters between rotating val passes (0 = off) |
| ``DINOV3_VAL_CHUNK`` / ``DINOV3_VAL_BS`` | ``4096`` / ``256`` | val chunk / batch |
| ``DINOV3_LOG_PERIOD`` | ``10`` | iters between SSL-loss logs |
| ``DINOV3_PAIR_METRICS`` | ``1`` | pair metrics on watermarked runs (auto-off clean) |
| ``DINOV3_RUN_DIR`` | unset | run directory for the metrics.jsonl mirror |
| ``DINOV3_WANDB_PROJECT/_GROUP/_NAME/_ID`` | see below | optional wandb naming |
| ``DINOV3_DUMP_INDICES`` | unset | val indices for the step-0 dump |
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from torchvision.transforms import v2

import dinov3.distributed as distributed
from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

_ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ARTIFACT_ROOT not in sys.path:
    sys.path.insert(0, _ARTIFACT_ROOT)

from common import results_io, wandb_keys  # noqa: E402
from common.pair_metrics import pair_metric_means  # noqa: E402

from additional_files.imagenet1k_parquet import (  # noqa: E402
    ImageNet1kParquet,
    PairImageNet1kParquet,
)
from additional_files.online_probe import OnlineLinearProbe  # noqa: E402
from additional_files.wm_env import env_bool, env_float, env_int  # noqa: E402

logger = logging.getLogger("dinov3")

OBSERVER_SIDECAR_NAME = "observer_probe.pt"

_PARAM_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# Persistent-val-loader lifecycle: ONE loader with persistent workers instead of
# a per-pass rebuild (worker/shm/fd churn under do_train's gc.disable()).  The
# workers re-enable the cyclic garbage collector on init -- they fork from
# do_train's gc-disabled interpreter and would otherwise accumulate per-item
# reference cycles for their whole (persistent) life.
_VAL_NUM_WORKERS = 4


def _enable_gc_in_workers(_worker_id):
    import gc as _gc

    _gc.enable()


def observer_enabled() -> bool:
    return os.environ.get("DINOV3_PROBE") == "1"


class _MutableIndicesSampler(torch.utils.data.Sampler):
    """Fixed loader, per-pass indices: the observer replaces ``items`` before each
    val pass.  Samplers are re-iterated in the MAIN process each epoch, so this is
    the one channel that reaches persistent workers with fresh per-pass state; for
    the pair-metric dataset the items are (index, payload_index) tuples carrying
    the pairing itself."""

    def __init__(self):
        self.items: List = []

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


def _val_eval_transform(size: int = 224) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize(size + 32),
            v2.CenterCrop(size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


class RunObserver:
    def __init__(self, cfg, model):
        self.is_main = distributed.is_main_process()
        self.param_dtype = _PARAM_DTYPE[cfg.compute_precision.param_dtype]
        self.embed_dim = model.embed_dim

        self.opacity = env_float(os.environ, "DINOV3_WM_OPACITY", 0.0)
        self.val_period = env_int(os.environ, "DINOV3_VAL_PERIOD", 500)
        self.val_chunk = env_int(os.environ, "DINOV3_VAL_CHUNK", 4096)
        self.val_bs = env_int(os.environ, "DINOV3_VAL_BS", 256)
        self.log_period = env_int(os.environ, "DINOV3_LOG_PERIOD", 10)
        self.run_dir = os.environ.get("DINOV3_RUN_DIR") or None
        self.dump_indices = [int(x) for x in os.environ.get("DINOV3_DUMP_INDICES", "").split(",") if x.strip()]

        self.probe = OnlineLinearProbe(self.embed_dim,
                                       lr=env_float(os.environ, "DINOV3_PROBE_LR", 1e-3))
        # Pair metrics: default on for watermarked runs (clean runs skip silently
        # -- keys absent, never NaN/0).
        self.pair_metrics = env_bool(os.environ, "DINOV3_PAIR_METRICS", True) and self.opacity > 0
        self._pair_disabled = False
        self._wandb = None
        self._val_ds = None
        self._val_loader = None
        self._val_sampler: Optional[_MutableIndicesSampler] = None
        self._val_perm: Optional[torch.Tensor] = None
        self._val_ptr = 0
        self._val_disabled = False
        if self.is_main:
            self._init_wandb(cfg)

    # -- logging ---------------------------------------------------------------
    def _init_wandb(self, cfg):
        try:
            import wandb

            self._wandb = wandb.init(
                project=os.environ.get("DINOV3_WANDB_PROJECT", "obsessed-encoder-dinov3"),
                group=os.environ.get("DINOV3_WANDB_GROUP") or None,
                id=os.environ.get("DINOV3_WANDB_ID") or None,
                resume="allow",
                name=os.environ.get("DINOV3_WANDB_NAME") or None,
                config={
                    "opacity": self.opacity,
                    "seed": cfg.train.seed,
                    "batch_size_per_gpu": cfg.train.batch_size_per_gpu,
                    "arch": cfg.student.arch,
                },
            )
        except Exception as e:  # wandb must never take training down
            logger.warning("RunObserver: wandb.init failed (%s); continuing without wandb", e)
            self._wandb = None

    def log(self, payload: Dict[str, float], step: int):
        if not self.is_main or not payload:
            return
        # Local files are the source of truth; wandb is optional monitoring.
        if self.run_dir is not None:
            try:
                results_io.append_metrics(self.run_dir, {"step": step, **payload})
            except Exception as e:
                logger.warning("RunObserver: metrics mirror failed (%s)", e)
        if self._wandb is not None:
            # wandb gets the repo-wide unified section names; the jsonl mirror
            # above keeps the upstream keys.
            self._wandb.log(
                wandb_keys.remap(payload, wandb_keys.DINOV3,
                                 prefixes=wandb_keys.DINOV3_PREFIXES),
                step=step)

    # -- step-0 stimulus dump ---------------------------------------------------
    def step0_dump(self):
        if not self.is_main or self.opacity <= 0:
            return
        try:
            from additional_files.dump_views import dump_val_composites

            out_dir = os.path.join(self.run_dir, "step0_views") if self.run_dir else None
            dump_val_composites(
                opacity=self.opacity, indices=self.dump_indices or None,
                out_dir=out_dir, wandb_run=self._wandb,
            )
            if self._wandb is not None:
                self._wandb.log({}, step=0)  # commit the buffered images
        except Exception as e:
            logger.warning("RunObserver: step-0 dump failed (%s); continuing", e)

    # -- per-step hook -----------------------------------------------------------
    def on_step(self, iteration: int, model, data, metrics_dict, scalars: Optional[Dict[str, float]] = None):
        if not self.is_main:
            return
        payload: Dict[str, float] = {}

        labels = data.get("collated_labels")
        cls = getattr(model, "last_teacher_cls", None)
        if labels is not None and cls is not None:
            payload.update(self.probe.observe(cls, labels))

        if self.val_period > 0 and (iteration + 1) % self.val_period == 0:
            payload.update(self._val_pass(model))

        if iteration % self.log_period == 0:
            for k, v in metrics_dict.items():
                try:
                    payload[f"ssl/{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
            if scalars:
                payload.update(scalars)

        self.log(payload, step=iteration)

    # -- rotating val pass --------------------------------------------------------
    def _pair_active(self) -> bool:
        return self.pair_metrics and not self._pair_disabled

    def _build_val(self):
        # The dual-render dataset is constructed only when the pair metrics will
        # consume it: clean runs keep the single-render item and pay nothing for
        # the second render.
        dataset_cls = PairImageNet1kParquet if self._pair_active() else ImageNet1kParquet
        self._val_ds = dataset_cls(
            split=ImageNet1kParquet.Split.VAL,
            transform=_val_eval_transform(),
        )
        # Pre-shuffle once (fixed seed): the mirror's val split is label-sorted,
        # so a contiguous chunk would be a single class block.
        g = torch.Generator().manual_seed(0)
        self._val_perm = torch.randperm(len(self._val_ds), generator=g)

    def _build_val_loader(self):
        # One persistent multi-worker loader instead of a per-pass rebuild.  The
        # explicit generator keeps the loader's base-seed draw off the global
        # torch stream: the val pass must be RNG-neutral so it stays invisible to
        # the training trajectory.
        self._val_sampler = _MutableIndicesSampler()
        self._val_loader = torch.utils.data.DataLoader(
            self._val_ds,
            batch_size=self.val_bs,
            sampler=self._val_sampler,
            num_workers=_VAL_NUM_WORKERS,
            persistent_workers=_VAL_NUM_WORKERS > 0,
            drop_last=False,
            generator=torch.Generator().manual_seed(0),
            worker_init_fn=_enable_gc_in_workers,
        )

    def _teardown_val_loader(self):
        if self._val_loader is None:
            return
        loader = self._val_loader
        self._val_loader = None
        self._val_sampler = None
        del loader
        gc.collect()  # do_train disabled the cyclic collector; reclaim workers now

    def _next_val_indices(self) -> List[int]:
        n = len(self._val_perm)
        idx = [int(self._val_perm[(self._val_ptr + i) % n]) for i in range(min(self.val_chunk, n))]
        self._val_ptr = (self._val_ptr + self.val_chunk) % n
        return idx

    @torch.no_grad()
    def _val_pass(self, model) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        if self._val_disabled:
            return payload
        try:
            if self._val_ds is None:
                self._build_val()
            if self._val_loader is None:
                self._build_val_loader()

            idx = self._next_val_indices()
            n = len(idx)
            # ``pair`` freezes the ITEM SHAPE for this pass (the loader's dataset
            # was built for it); ``pair_ok`` tracks mid-pass failures, which skip
            # the remaining swap work but never the probe.
            pair = pair_ok = isinstance(self._val_ds, PairImageNet1kParquet)
            if pair:
                # Cyclic-shift pairing within the current chunk order: item k
                # carries key k+1, deterministic given val_ptr.  The pairing
                # travels through the sampler items -- persistent workers never
                # see observer state.
                self._val_sampler.items = [(idx[k], idx[(k + 1) % n]) for k in range(n)]
            else:
                self._val_sampler.items = list(idx)

            own_tokens: List[torch.Tensor] = []
            swap_tokens: List[torch.Tensor] = []
            correct = 0.0
            total = 0
            backbone = model.teacher.backbone
            for batch in self._val_loader:
                own_imgs, swap_imgs, labels = batch if pair else (batch[0], None, batch[1])
                own_imgs = own_imgs.to("cuda", dtype=self.param_dtype, non_blocking=True)
                out = backbone(own_imgs, is_training=True)
                cls = out["x_norm_clstoken"]
                correct += float(self.probe.accuracy(cls, labels))
                total += labels.shape[0]
                if pair_ok:
                    try:
                        # Pool to the (B, D) token mean per batch, in fp32 -- the
                        # only representation the pair metric reads.  Retaining the
                        # full (N, T, D) populations instead would be a multi-GB
                        # GPU transient every val pass on the ViT-L run.
                        own_tokens.append(out["x_norm_patchtokens"].float().mean(1))
                        swap_imgs = swap_imgs.to("cuda", dtype=self.param_dtype, non_blocking=True)
                        swap_out = backbone(swap_imgs, is_training=True)
                        swap_tokens.append(swap_out["x_norm_patchtokens"].float().mean(1))
                    except Exception as e:
                        logger.warning(
                            "RunObserver: pair-metric featurization failed (%s); "
                            "disabling pair metrics", e)
                        pair_ok = False
            payload["probe/val_top1"] = correct / max(total, 1)

            if pair_ok:
                try:
                    payload.update(self._pair_metric_payload(own_tokens, swap_tokens))
                except Exception as e:
                    logger.warning(
                        "RunObserver: pair-metric computation failed (%s); "
                        "disabling pair metrics", e)
                    pair_ok = False
            own_tokens.clear()
            swap_tokens.clear()
            gc.collect()

            if pair and not pair_ok:
                # Self-disable (the never-kill rule) and fall back to the
                # single-render dataset from the next pass on.
                self._pair_disabled = True
                self._teardown_val_loader()
                self._val_ds = None
            return payload
        except Exception as e:
            logger.warning("RunObserver: val pass failed (%s); disabling further val passes", e)
            self._val_disabled = True
            return payload

    def _pair_metric_payload(self, own_tokens: List[torch.Tensor],
                             swap_tokens: List[torch.Tensor]) -> Dict[str, float]:
        """pair/<distribution>/tokens_mean means for one chunk: the centered
        cosine of the pooled patch-token mean, the representation the example's
        pair figure reads.  Inputs are the already-pooled (B, D) per-batch means."""
        own = torch.cat(own_tokens)
        swap = torch.cat(swap_tokens)
        partner_pos = np.roll(np.arange(own.shape[0]), -1)  # item k carries key k+1
        means = pair_metric_means(own, swap, partner_pos)
        return {f"pair/{dist}/tokens_mean": value for dist, value in means.items()}

    # -- checkpoint round-trip for the probe (resume) ------------------------------
    def state_dict(self) -> dict:
        return {"probe": self.probe.state_dict(), "val_ptr": self._val_ptr}

    def load_state_dict(self, sd: dict) -> None:
        if "probe" in sd:
            self.probe.load_state_dict(sd["probe"])
        self._val_ptr = int(sd.get("val_ptr", 0))

    def save_sidecar(self, output_dir: str) -> None:
        """Persist the probe state next to the training checkpoint (main process
        only).  Atomic rename so an interrupted write never leaves a truncated
        sidecar for the next session to trip on."""
        if not self.is_main:
            return
        try:
            path = os.path.join(output_dir, OBSERVER_SIDECAR_NAME)
            tmp = path + ".tmp"
            torch.save(self.state_dict(), tmp)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning("RunObserver: sidecar save failed (%s); continuing", e)

    def load_sidecar(self, output_dir: str) -> None:
        """Restore the probe state on resume (before the first on_step).

        A missing sidecar on a resumed session is loudly warned, not fatal: the
        probe would restart from scratch and its re-training dip can mimic a real
        probe drop, so treat early post-resume probe metrics as suspect.
        """
        if not self.is_main:
            return
        path = os.path.join(output_dir, OBSERVER_SIDECAR_NAME)
        try:
            if not os.path.exists(path):
                logger.warning(
                    "RunObserver: no probe sidecar at %s on resume -- the probe "
                    "restarts from scratch; treat early post-resume probe metrics "
                    "as suspect", path,
                )
                return
            self.load_state_dict(torch.load(path, map_location=self.probe.device, weights_only=False))
            logger.info("RunObserver: restored probe sidecar from %s", path)
        except Exception as e:
            logger.warning("RunObserver: sidecar load failed (%s); continuing with a fresh probe", e)

    def finish(self):
        if self.is_main and self._wandb is not None:
            self._wandb.finish()
