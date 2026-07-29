"""Pair analysis: the in-training paired-input cosine readout.

Two suites of freshly-rendered PushT stimuli — never training frames — probe
what the encoder keys on:

* colour suite:     tag = the corner colour, content = the rendered scene.
* t_position suite: tag = the destination outline pose (T-dest),
                    content = the block pose (T-orig); colour held fixed.

For the run's suite, three pairings of mean-centered cosine similarity:
  same_tag     — different content, same tag
  same_content — same content, different tag
  baseline     — neither shared (the ≈ 0 null reference)

An honest encoder reads high on same_content and ≈ 0 on same_tag; a captured
encoder is the mirror image. Runs under model.eval() + no_grad with dedicated
RNGs only, so it never mutates BN stats, dropout, or the global RNG.
"""

from __future__ import annotations

import lightning as pl
import numpy as np
import torch

from common.pair_metrics import cyclic_derangement
from .wandb_axes import pin_step_axis
from ..pixel_tag import PixelTag
from ..stimuli import StimulusRenderer, sample_dest, sample_orig, stamp_corner
from utils import get_img_preprocessor

SAMPLE_SEED = 0  # fixed so the same held-out stimulus set is used every tick
REPS = ("backbone", "projection")
SUITES = ("colour", "t_position")


class PairAnalysisCallback(pl.Callback):
    """Logs pair/{rep}/{pairing}/cos_mean every every_n_steps optimizer
    steps, to every attached logger."""

    def __init__(self, every_n_steps, num_pairs, img_size, tag_size, tag_seed, suites):
        # suites selects which signal this run probes: "colour" (corner tag)
        # is meaningful when the tag is stamped; "t_position" (destination
        # outline) when training varied the destination. Metric keys carry no
        # suite name, so a run must probe exactly the one matching its dataset.
        self.every = every_n_steps
        self.num_pairs = num_pairs
        unknown = set(suites) - set(SUITES)
        if unknown:
            raise ValueError(f"unknown pair suite(s): {sorted(unknown)}")
        if len(suites) != 1:
            raise ValueError(f"exactly one pair suite per run, got: {list(suites)}")
        self.suite = suites[0]
        self.img_size = img_size
        self.tag = PixelTag(mode="video", size=tag_size, seed=tag_seed)
        self.preprocess = get_img_preprocessor("pixels", "pixels", img_size=img_size)
        self.renderer = StimulusRenderer()
        self._last_step = -1
        self._axes_pinned = False

    @torch.no_grad()
    def _encode(self, model, frames, device):
        px = self.preprocess({"pixels": np.stack(frames)})["pixels"].to(device)
        out = model.encoder(px, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0].float()
        return {"backbone": cls, "projection": model.projector(cls).float()}

    @staticmethod
    def _pairings(k, partner):
        # name -> (a_pass, a_rows, b_pass, b_rows). The name is what the couple
        # holds identical; their ordering is the double dissociation.
        identity = np.arange(k)
        inv = np.empty(k, dtype=np.int64)
        inv[partner] = identity
        return {
            "same_tag": ("own", identity, "swap", inv),           # content differs
            "same_content": ("own", identity, "swap", identity),  # tag differs
            "baseline": ("own", identity, "own", partner),        # neither shared
        }

    @staticmethod
    def _pair_metrics(passes, partner):
        # Mean cosine of mean-centered embeddings per (rep, pairing). One
        # reference mean per representation.
        out = {}
        for pairing, (ap, ar, bp, br) in PairAnalysisCallback._pairings(len(partner), partner).items():
            for rep in REPS:
                mu = passes["own"][rep].mean(0)
                a, b = passes[ap][rep][ar], passes[bp][rep][br]
                cc = torch.nn.functional.cosine_similarity(a - mu, b - mu, dim=1)
                out[f"pair/{rep}/{pairing}/cos_mean"] = cc.mean().item()
        return out

    def _colour_passes(self, model, device, rng):
        k = self.num_pairs
        orig, dest = sample_orig(rng, k), sample_dest(rng, k)
        partner = cyclic_derangement(k, rng)
        base = [self.renderer.render(orig[j], dest[j]) for j in range(k)]
        col = [self.tag.color_for(j) for j in range(k)]  # per-scene distinct colours
        n = self.tag.size
        own = self._encode(model, [stamp_corner(base[j], col[j], n) for j in range(k)], device)
        swap = self._encode(model, [stamp_corner(base[j], col[partner[j]], n) for j in range(k)], device)
        return {"own": own, "swap": swap}, partner

    def _tpos_passes(self, model, device, rng):
        k = self.num_pairs
        orig, dest = sample_orig(rng, k), sample_dest(rng, k)
        partner = cyclic_derangement(k, rng)
        c0 = self.tag.color_for(0)  # one fixed colour held across the suite
        n = self.tag.size
        own = self._encode(
            model, [stamp_corner(self.renderer.render(orig[j], dest[j]), c0, n) for j in range(k)], device
        )
        swap = self._encode(
            model,
            [stamp_corner(self.renderer.render(orig[j], dest[partner[j]]), c0, n) for j in range(k)],
            device,
        )
        return {"own": own, "swap": swap}, partner

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        # skip step 0, off-cadence steps, and grad-accumulation repeat fires
        if step == 0 or step % self.every or step == self._last_step:
            return
        self._last_step = step
        if not trainer.is_global_zero:
            return
        if not self._axes_pinned:
            pin_step_axis(trainer, "pair")
            self._axes_pinned = True
        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()
        try:
            builders = {"colour": self._colour_passes, "t_position": self._tpos_passes}
            with torch.no_grad():
                rng = np.random.default_rng(SAMPLE_SEED)
                passes, partner = builders[self.suite](pl_module.model, device, rng)
                metrics = self._pair_metrics(passes, partner)
        finally:
            if was_training:
                pl_module.train()
        for lg in trainer.loggers:
            lg.log_metrics(metrics, step=step)
