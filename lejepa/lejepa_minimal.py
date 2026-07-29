"""Minimal LeJEPA reproduction (ViT/s-8 + Imagenette).

Verbatim port of the upstream minimal example:
https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md

Two deviations from upstream, both non-substantive:
  1. The DataLoader worker count is read from the Hydra config
     (`+num_workers=...`, default 8) and pin_memory / persistent_workers are
     enabled. Throughput-only knobs; they do not change the optimisation math.
  2. The Imagenette 160px data is loaded from the auto-converted parquet
     revision of frgfm/imagenette rather than its (now unsupported) loading
     script. Byte-identical images and labels — see HFDataset below.
Everything that affects results (lamb, V, proj_dim, lr, bs, epochs, augments,
architecture, losses, schedule) is left exactly as upstream, so the published
accuracy curves are the reference.

Every change made for this repo's experiments sits inside a
`# >>> obsessed-encoder` ... `# <<< obsessed-encoder` block and is inert
at its default (unset config keys reproduce the upstream behaviour bit for
bit). The per-change inventory is in additional_files/README.md.
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm, wandb, hydra, tqdm
from omegaconf import DictConfig
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

# >>> obsessed-encoder: imports for the marked blocks below -- the only
# non-upstream imports in this file (the watermark renderer, the pair-metric
# primitives, and the local metrics mirror)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import results_io, wandb_keys
from common.pair_metrics import SwapPayloadTransform, cyclic_derangement, pair_metric_means
from common.watermark import make_source_transform
# <<< obsessed-encoder


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class HFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        V=1,
        # >>> obsessed-encoder: watermark seam + dataset seam, both default-inert
        # (source_transform=None and dataset='imagenette' reproduce upstream exactly)
        source_transform=None,
        dataset="imagenette",
        data_dir=None,
        # <<< obsessed-encoder
    ):
        self.V = V
        # >>> obsessed-encoder: dataset seam -- 'imagenet1k' reads the locally
        # fetched, revision-pinned ImageNet-1k parquet shards through the SAME
        # `datasets` library as the default Imagenette path below; the item
        # interface (item["image"] / item["label"]) is identical.
        self.source_transform = source_transform
        if dataset == "imagenet1k":
            from common.imagenet1k import load_split

            self.ds = load_split(split, data_dir=data_dir)
        elif dataset != "imagenette":
            raise ValueError(f"unknown dataset {dataset!r}; expected 'imagenette' or 'imagenet1k'")
        else:
            # Upstream uses load_dataset("frgfm/imagenette", "160px", split). That
            # repo is script-based and `datasets` >=4 dropped script support, so we
            # read the byte-identical 160px data from its auto-converted parquet
            # revision instead. Same images, same labels.
            self.ds = load_dataset(
                "frgfm/imagenette",
                revision="refs/convert/parquet",
                data_files={
                    "train": "160px/train/*.parquet",
                    "validation": "160px/validation/*.parquet",
                },
                split=split,
            )
        # (the load above is upstream's, only moved under the else branch)
        # <<< obsessed-encoder
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        # >>> obsessed-encoder: watermark seam -- the pattern is applied to the
        # decoded source image BEFORE augmentation, with the dataset index as the
        # per-image key, so every augmented view inherits the same pattern. With no
        # source_transform bound this is the upstream item path exactly.
        if self.source_transform is not None:
            img = self.source_transform(img, i)
        # <<< obsessed-encoder
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), item["label"]

    def __len__(self):
        return len(self.ds)


# >>> obsessed-encoder: startup sample grid -- for a few training images, the
# deterministic eval crop plus V augmented training views OF THE SAME image
# (same content, same watermark key), exactly as the model sees them,
# denormalized back to pixels. Written to <run_dir>/samples.png and logged to
# wandb; called under a forked RNG so the dump never touches the training
# trajectory.
def dump_sample_grid(train_ds, indices, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def to_pixels(t):
        x = (t.float().cpu() * std + mean).clamp(0.0, 1.0)
        return x.permute(1, 2, 0).numpy()

    cols = 1 + train_ds.V
    fig, axes = plt.subplots(len(indices), cols, figsize=(2.0 * cols, 2.1 * len(indices)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(indices):
        # One decode + one watermark render, then both transform stacks of the
        # same source image -- the legibility check needs the identical stimulus
        # through the eval and training paths.
        img = train_ds.ds[i]["image"].convert("RGB")
        if train_ds.source_transform is not None:
            img = train_ds.source_transform(img, i)
        panes = [(train_ds.test(img), "eval crop")] + [
            (train_ds.aug(img), f"view {v + 1}") for v in range(train_ds.V)
        ]
        for c, (view, title) in enumerate(panes):
            ax = axes[r, c]
            ax.imshow(to_pixels(view))
            if r == 0:
                ax.set_title(title, fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
# <<< obsessed-encoder


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    # >>> obsessed-encoder: run identity -- seed as config (default 0, upstream's
    # hard-coded value), wandb naming as config, and the local run directory for the
    # metrics mirror (unset => no mirror, wandb-only, exactly upstream)
    seed = int(cfg.get("seed", 0))
    run_dir = cfg.get("run_dir") or None
    wandb.init(
        project=cfg.get("wandb_project", "obsessed-encoder-lejepa"),
        name=cfg.get("wandb_name") or None,
        group=cfg.get("wandb_group") or None,
        config=dict(cfg),
    )
    torch.manual_seed(seed)

    def log_metrics(payload, step):
        # wandb gets the repo-wide unified section names; the jsonl mirror keeps
        # the upstream keys (the figures' source of truth).
        wandb.log(wandb_keys.remap(payload, wandb_keys.LEJEPA), step=step)
        if run_dir is not None:
            results_io.append_metrics(run_dir, {"step": step, **payload})
    # <<< obsessed-encoder

    # Docstring deviation (1), NOT a sentinel block: the loader knobs here and in
    # the two DataLoader calls below apply unconditionally (throughput only).
    num_workers = cfg.get("num_workers", 8)

    # >>> obsessed-encoder: watermark seam -- bind the render knobs into ONE
    # (img, index) -> img source transform shared by the train, eval, and swap
    # datasets. Unset / zero opacity leaves it None => the exact upstream data path.
    opacity = float(cfg.get("watermark_opacity", 0.0) or 0.0)
    watermark = None
    if opacity > 0:
        watermark = make_source_transform(
            opacity=opacity,
            modulus=cfg.get("watermark_modulus"),
            tile_px=int(cfg.get("watermark_tile", 32)),
            bit_capacity=int(cfg.get("watermark_bits", 12)),
            anchor=cfg.get("origin_anchor", "none"),
            repeat=bool(cfg.get("watermark_repeat", True)),
            random_anchor=bool(cfg.get("random_anchor", False)),
        )
    dataset_name = cfg.get("dataset", "imagenette")
    data_dir = cfg.get("data_dir") or None
    num_classes = {"imagenette": 10, "imagenet1k": 1000}[dataset_name]

    train_ds = HFDataset("train", V=cfg.V, source_transform=watermark,
                         dataset=dataset_name, data_dir=data_dir)
    test_ds = HFDataset("validation", V=1, source_transform=watermark,
                        dataset=dataset_name, data_dir=data_dir)
    # <<< obsessed-encoder
    train = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    # >>> obsessed-encoder: step-budget knobs (both None => upstream: epoch loop,
    # evaluation at epoch end). Evaluation always runs on the full validation
    # split, exactly as upstream.
    max_steps = int(cfg.get("max_steps") or 0) or None
    eval_every_steps = int(cfg.get("eval_every_steps") or 0) or None
    # <<< obsessed-encoder
    test = DataLoader(
        test_ds,
        batch_size=256,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # >>> obsessed-encoder: startup sample grid (RNG-isolated; run_dir-gated so
    # the default invocation is untouched)
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            grid_indices = np.linspace(0, len(train_ds) - 1, num=3, dtype=np.int64).tolist()
            grid_path = dump_sample_grid(train_ds, grid_indices,
                                         os.path.join(run_dir, "samples.png"))
        wandb.log({"samples": wandb.Image(grid_path)}, step=0)
    # <<< obsessed-encoder

    # modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim).to("cuda")
    # >>> obsessed-encoder: probe width follows the dataset seam (10 classes ==
    # upstream's Imagenette probe; 1000 for the imagenet1k mirror)
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, num_classes)).to("cuda")
    # <<< obsessed-encoder
    sigreg = SIGReg().to("cuda")
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    # >>> obsessed-encoder: LR horizon -- the cosine anneals over lr_horizon
    # steps instead of the training length, so a step-budgeted run stops
    # mid-schedule. The schedule RULE (warmup -> cosine, eta_min) is upstream's;
    # default None keeps horizon == training length, i.e. the upstream T_max exactly.
    horizon_steps = int(cfg.get("lr_horizon") or max_steps or total_steps)
    s2 = CosineAnnealingLR(opt, T_max=horizon_steps - warmup_steps, eta_min=1e-3)
    # <<< obsessed-encoder
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    # >>> obsessed-encoder: pair metrics -- at every eval tick, re-render each
    # eval image carrying its partner's key (a seeded cyclic derangement) and log
    # the three centered-cosine means at the backbone representation. Built only
    # when a watermark is configured: the clean run has no keys to swap, logs no
    # pair/ metrics, and exercises the unmodified path.
    pair = None
    if watermark is not None:
        partner_pos = cyclic_derangement(len(test_ds), seed)
        partner_index = {j: int(partner_pos[j]) for j in range(len(test_ds))}
        swap_ds = HFDataset("validation", V=1,
                            source_transform=SwapPayloadTransform(watermark, partner_index),
                            dataset=dataset_name, data_dir=data_dir)
        # Explicit generators keep these loaders' seed draws off the global torch
        # stream, so the pair passes are invisible to the training trajectory.
        pair = dict(
            partner_pos=partner_pos,
            own=DataLoader(test_ds, batch_size=256, num_workers=num_workers,
                           pin_memory=True,
                           generator=torch.Generator().manual_seed(seed)),
            swap=DataLoader(swap_ds, batch_size=256, num_workers=num_workers,
                            pin_memory=True,
                            generator=torch.Generator().manual_seed(seed)),
        )

    @torch.inference_mode()
    def _backbone_features(loader):
        feats = []
        for vs, _ in loader:
            with autocast("cuda", dtype=torch.bfloat16):
                feats.append(net(vs.to("cuda", non_blocking=True))[0].float())
        return torch.cat(feats)

    def pair_payload():
        if pair is None:
            return {}
        own = _backbone_features(pair["own"])
        swap = _backbone_features(pair["swap"])
        means = pair_metric_means(own, swap, pair["partner_pos"])
        # Release the feature matrices and return their cached blocks before
        # training resumes: at the recipe's batch size the training peak is close
        # to a 48 GB card's capacity, and eval-tick allocations left in the cache
        # fragment the next backward. Memory hygiene only.
        del own, swap
        torch.cuda.empty_cache()
        return {f"pair/{dist}/backbone": value for dist, value in means.items()}
    # <<< obsessed-encoder

    scaler = GradScaler(enabled="cuda" == "cuda")

    # >>> obsessed-encoder: the upstream epoch-end evaluation, extracted verbatim
    # into a function so the step-budget path can evaluate mid-epoch; the pair-metric
    # tick rides the same call. global_step drives the metrics mirror and the budget.
    def evaluate():
        net.eval(), probe.eval()
        correct = 0
        with torch.inference_mode():
            for vs, y in test:
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    correct += (probe(net(vs)[0]).argmax(1) == y).sum().item()
        payload = {"test/acc": correct / len(test_ds), **pair_payload()}
        net.train(), probe.train()
        return payload

    global_step = 0
    last_eval_step = None
    stop = False
    # <<< obsessed-encoder
    # Training
    for epoch in range(cfg.epochs):
        net.train(), probe.train()
        for vs, y in tqdm.tqdm(train, total=len(train)):
            with autocast("cuda", dtype=torch.bfloat16):
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)
                loss = lejepa_loss + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            # >>> obsessed-encoder: metrics mirror + step budget + step-cadence
            # evaluation. With max_steps / eval_every_steps unset this logs exactly
            # the upstream per-step payload and never breaks early.
            global_step += 1
            log_metrics(
                {
                    "train/probe": probe_loss.item(),
                    "train/lejepa": lejepa_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                },
                step=global_step,
            )
            if eval_every_steps and global_step % eval_every_steps == 0:
                log_metrics(evaluate(), step=global_step)
                last_eval_step = global_step
            if max_steps and global_step >= max_steps:
                stop = True
                break
            # <<< obsessed-encoder

        # >>> obsessed-encoder: the upstream epoch-end evaluation, via the
        # extracted function (skipped when the step-cadence path owns evaluation)
        if not eval_every_steps:
            log_metrics({**evaluate(), "test/epoch": epoch}, step=global_step)
        if stop:
            break
        # <<< obsessed-encoder
    # >>> obsessed-encoder: final evaluation for step-budgeted runs (upstream
    # evaluated at every epoch end; a budgeted run can stop between ticks)
    if eval_every_steps and last_eval_step != global_step:
        log_metrics(evaluate(), step=global_step)
    # <<< obsessed-encoder
    wandb.finish()


if __name__ == "__main__":
    main()
