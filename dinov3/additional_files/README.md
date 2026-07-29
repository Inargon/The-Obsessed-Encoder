# DINOv3 example

Reproduces the blog's DINOv3 case: Meta's published
[DINOv3](https://github.com/facebookresearch/dinov3) reference implementation
(vendored at the parent directory under its [license](../LICENSE.md)), training
the `vitl_im1k_lin834` "Fast setup" recipe (ViT-L/16, ImageNet-1k) from scratch
with a faint per-image watermark at **opacity 0.1** — plus the dense NYU-depth
track that shows local features failing alongside the global probe.

## Layout

* `../dinov3/` — the vendored upstream package, byte-identical to
  [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) at
  commit `346f38f` except for marked `# >>> obsessed-encoder` blocks in
  three files (inventory below) and the omission of upstream's `notebooks/`
  directory (demo notebooks; nothing in the training path references them).
  Verify yourself with the commands in the repository root's `VERIFYING.md`.
* `configs/vitl_im1k.yaml` — the shipped recipe: the published Fast-setup
  config adapted to a single GPU (batch 128 with upstream's native
  `sqrt_wrt_1024` LR scaling; 500K-iteration schedule horizon; teacher
  milestone dumped every 5K iterations; upstream's activation checkpointing
  on, since without it iBOT's heavy-tailed activation peaks OOM even 80 GB
  cards).  Watermark-free — identical across arms.
* `configs/arms.yaml` — the three arms as `DINOV3_WM_*` environment sets
  (upstream's dataset string can't carry configuration, so the watermark knobs
  ride the environment; the printed command shows the assignments).
  `random_control` differs from `watermarked` in `DINOV3_WM_REPEAT=0` and
  nothing else; `clean` sets no `DINOV3_WM_*` vars at all
  (`tests/test_configs.py` enforces it).
* the additional_files — the dataset bridge (`imagenet1k_parquet.py`), the passive
  observer (`train_hooks.py`, `online_probe.py`, `probe_collate.py`), the
  stimulus dump (`dump_views.py`), the dense probe (`dense_eval.py`,
  `fetch_nyu.py`), and the runner (`run.py`).

## Run

```bash
uv run python dinov3/additional_files/prepare_data.py       # IN-1k (~50 GB) + BTS NYU (~6.7 GB)
uv run python dinov3/additional_files/run.py --seeds 3 --gpus 0,1,2
uv run python dinov3/additional_files/run.py --plot-only
```

Each run is one printed `torchrun` command with its environment, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 DATA_DIR=$PWD/data DINOV3_MAX_ITER=50000 DINOV3_PROBE=1 \
DINOV3_RUN_DIR=$PWD/results/watermarked_seed0 DINOV3_WM_OPACITY=0.1 \
DINOV3_WM_MODULUS=4096 DINOV3_WM_TILE=32 DINOV3_WM_BITS=12 DINOV3_WM_ANCHOR=gabor \
DINOV3_WM_RANDOM_ANCHOR=1 DINOV3_WM_REPEAT=1 \
PYTHONPATH=$PWD/dinov3:$PWD \
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  dinov3/dinov3/train/train.py \
  --config-file dinov3/additional_files/configs/vitl_im1k.yaml \
  --output-dir $PWD/results/watermarked_seed0/train --seed 0 train.seed=0
```

**Resume is upstream's own**: the trainer natively resumes from the latest
checkpoint in its output dir, so re-invoking a killed run (same command, or
just re-running `run.py`) continues it — on a persistent volume an interruption
costs at most `checkpointing.period` (1500) iterations.  The online probe rides every
checkpoint as a sidecar so its curve stays continuous across resumes.

## The dense NYU-depth track

Training dumps a teacher checkpoint every 5K iterations
(`train/eval/training_<it>/teacher_checkpoint.pth`).  After training, the
runner evaluates each milestone with the **truncated dense probe**
(`dense_eval.py`): the vendored NYU-depth linear-probe protocol, byte-identical
config, with only the head-training loop capped at 4800 iterations (the full
38.4K-iteration protocol is `--head-iters 38400 --tta`).  Watermarked arms are
measured on watermarked NYU images — their own operating point.

Results append to the run's `metrics.jsonl` (`dense/nyu_rmse` at
`step = teacher iteration`) and to `dense_eval_results.csv`.  The tail is
resumable: re-invoking evaluates only milestones without a CSV row.  A failed
milestone gets a `failed` row and is *not* retried automatically (a corrupt
checkpoint must not loop), and the run reports failure without its done-marker
until every milestone is `ok` — fix the cause, then `--retry-failed`
re-evaluates the failed rows.

**The data gate**: every `dense_eval.py` invocation first hard-stops unless the
NYU root is the BTS protocol data (manifest counts 24,231/654, uint16-mm
depths, raw-GT test split).  If you see `dense_eval: HARD STOP -- manifest
missing` or a count mismatch, your root is the wrong (usually the h5-mirror)
distribution — it trains fine but is not the dataset DINOv3 was benchmarked
on (different train composition, filled instead of raw test depth), so its
numbers are not comparable and it is refused.  `fetch_nyu.py` fetches the
right one.

## Figures (`--plot-only`)

1. `f1_dino_watermark.png` — the stimulus quadriptych at opacity 0.1 (blog Fig. 1).
2. `f2_dino_crossover.png` — SSL total loss + IN-1k probe top-1 + NYU-depth RMSE on the
   shared log-iteration axis (blog Fig. 2).
3. `f3_dino_pair.png` — the paired-input cosine time series at the pooled
   patch-token representation (blog Fig. 3).

### Reproduce blog Figure 2 from a fresh machine

1. `uv sync`
2. `uv run python dinov3/additional_files/prepare_data.py`
3. `uv run python dinov3/additional_files/run.py --seeds 3 --gpus 0,1,2`
   (one 80 GB-class GPU per concurrent run; `--gpus 0` runs the 9 runs
   sequentially)
4. `uv run python dinov3/additional_files/run.py --plot-only`
5. open `results/figures/f2_dino_crossover.png`

## Marked-change inventory (the vendored tree)

Three files carry `# >>> obsessed-encoder` blocks; nothing else in the
vendored tree is modified:

1. `dinov3/data/loaders.py` — one `elif` registering `ImageNet1kParquet`
   (lazy import of the additional_files).
2. `dinov3/train/ssl_meta_arch.py` — stash the detached teacher CLS (mean over
   the 2 global crops) for the passive probe.  Never enters the SSL loss or the
   backward graph.
3. `dinov3/train/train.py` — the label pass-through collate (env-gated), the
   `DINOV3_MAX_ITER` step cap (decoupled from the schedule horizon), a
   `worker_init_fn` that re-enables the cyclic garbage collector inside
   dataloader workers (they fork from `do_train`'s gc-disabled interpreter and
   would otherwise accumulate per-item reference cycles for the whole run —
   measured at ~6 GiB of worker RSS per 1000 batches, flat with the collector
   on), and the observer construction + per-step hook + probe sidecar on
   checkpoint writes.

Every block is inert without the experiment's environment: with `DINOV3_PROBE`
unset and no `DINOV3_*` vars, the training loop is upstream's. The one block
that is *active* in every shipped arm is the dataset registration in
`loaders.py` — the shipped config's `dataset_path: ImageNet1kParquet:...`
selects the pinned local ImageNet-1k shards (that is the block's whole job);
its watermark seam still keys off the `DINOV3_WM_*` env vars.

## Troubleshooting

* **CUDA out of memory** — with the shipped `train.checkpointing: true` the
  peak is ~24 GB (a 40 GB-class GPU suffices); without it, iBOT's variable
  masking peaks past 80 GB.  Batch size is not a tunable here: it changes the
  training statistics this experiment measures (upstream's LR rule would also
  rescale).
* **`no teacher milestones found`** — the run trained past the milestone period
  but `train/eval/` is empty: the training subprocess was launched without the
  shipped config (milestones come from `evaluation.eval_period_iterations`).
* **Dense probe row `failed`** — the error column carries the child's last
  lines; the most common cause is the NYU gate (see above).  Fix the cause,
  then `--retry-failed`.
