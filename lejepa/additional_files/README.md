# LeJEPA example

Reproduces the blog's LeJEPA case: the [upstream minimal recipe](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md)
(ViT-S/8, SIGReg + invariance loss, online linear probe), trained on
ImageNet-1k with a faint per-image watermark at **opacity 0.05** — and the two
matched controls.

## The configurations

`configs.yaml` maps each arm to the hydra overrides of `../lejepa_minimal.py`:

| Config | Meaning |
|---|---|
| `clean` | unmodified data (the blog's baseline) |
| `watermarked` | one per-image pattern, repeated across the frame (the test arm) |
| `random_control` | identical renderer and energy, never repeats (the control) |

Shared operating point: ImageNet-1k, batch 256, `lr 2e-3`, `lamb 0.02`,
`proj_dim 16`, `V 4` (the upstream-recommended hyperparameters), **30K steps**
with the LR cosine laid over a **200K-step horizon** (the run deliberately
stops mid-schedule), evaluation + pair metrics every 2000 steps over the full
validation split.  Watermark: 12-bit periodic
pattern, 32px tile, key modulus 4096, gabor origin anchor, anchor rendered on
the control too (`random_anchor: true`) so test and control differ in repetition alone.

`random_control` differs from `watermarked` in `watermark_repeat` and nothing
else; `clean` carries no watermark keys.  `tests/test_configs.py` enforces it.

## Run

```bash
uv run python lejepa/additional_files/prepare_data.py            # once, ~50 GB
uv run python lejepa/additional_files/run.py --seeds 3 --gpus 0  # 9 runs
uv run python lejepa/additional_files/run.py --plot-only         # figures only
```

Every run is printed as one self-contained command, e.g. (single run by hand):

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=./data/hf_cache \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lejepa/lejepa_minimal.py \
  +dataset=imagenet1k +bs=256 +lr=2e-3 +lamb=0.02 +proj_dim=16 +V=4 +epochs=1000 \
  +num_workers=16 +max_steps=30000 +eval_every_steps=2000 \
  +lr_horizon=200000 +watermark_opacity=0.05 +watermark_modulus=4096 \
  +watermark_bits=12 +watermark_tile=32 +origin_anchor=gabor +random_anchor=true \
  +watermark_repeat=true +seed=0 +run_dir=$PWD/results/watermarked_seed0 \
  +data_dir=$PWD/data hydra.run.dir=$PWD/results/watermarked_seed0/hydra
```

The default invocation with none of these overrides (`python
lejepa/lejepa_minimal.py +lamb=0.02 +V=4 +proj_dim=16 +lr=2e-3 +bs=256
+epochs=800`) is the untouched upstream Imagenette recipe and reproduces its
published ~90% online-probe accuracy — the environment's fidelity anchor.

There is **no resume**: the upstream minimal file has no checkpointing and we
did not add any, so a killed run restarts from scratch (delete its partial run
directory, or just re-invoke — a run directory without `summary.json` is
re-run from step 0 but appends to the old `metrics.jsonl`; delete it first for
a clean mirror).

## Figures (`--plot-only`)

Rendered into `RESULTS_DIR/figures/` from local `metrics.jsonl` files only:

1. `f4_lejepa_watermark.png` — the stimulus pair (blog Fig. 4): watermarked ·
   random control. Renderer-only; needs the dataset but no runs.
2. `f5_lejepa_crossover.png` — LeJEPA loss + online probe top-1 (blog Fig. 5), clean
   grey/blue, watermarked solid red, control dashed; mean ± std over seeds.
3. `f6_lejepa_pair.png` — the paired-input cosine time series (blog Fig. 6) at the
   backbone representation, control and watermarked panels.

### Reproduce blog Figure 5 from a fresh machine

1. `uv sync`
2. `uv run python lejepa/additional_files/prepare_data.py`
3. `uv run python lejepa/additional_files/run.py --seeds 3 --gpus 0`
4. `uv run python lejepa/additional_files/run.py --plot-only`
5. open `results/figures/f5_lejepa_crossover.png`

(Figure 4 needs only steps 1–2 + `--plot-only`; Figure 6 is written by step 4.)

## Marked-change inventory

All changes to `../lejepa_minimal.py` sit inside
`# >>> obsessed-encoder` … `# <<< obsessed-encoder` blocks; each is inert
at its default (unset keys reproduce upstream bit for bit):

1. **Watermark seam** — `watermark_opacity/modulus/bits/tile/repeat` +
   `origin_anchor` + `random_anchor` bind one `(img, index) -> img` source
   transform, applied to the decoded source before augmentation.  Unset/zero
   opacity ⇒ the exact upstream data path.
2. **Seed as config** — `seed` (default 0, upstream's hard-coded value).
3. **Local metrics mirror** — every logged payload also appended to
   `<run_dir>/metrics.jsonl`; wandb naming (`wandb_project/_name/_group`)
   exposed as config.
4. **Startup sample grid** — eval crop + V augmented views for 3 images to
   `<run_dir>/samples.png`, under a forked RNG (training trajectory untouched);
   skipped when no `run_dir` is set.
5. **Dataset seam** — `dataset: imagenette` (upstream default) | `imagenet1k`
   (the locally fetched, revision-pinned IN-1k shards, read through the same
   `datasets` library); the probe head width follows (10 → 1000).
6. **Step budget & eval cadence** — `max_steps` + `eval_every_steps`; both
   unset ⇒ upstream epoch loop with epoch-end evaluation.  Evaluation always
   runs on the full validation split, as upstream.
7. **LR horizon** — `lr_horizon`: the cosine anneals over this many steps
   instead of the training length.  The schedule *rule* (one-epoch linear
   warmup → cosine, upstream's eta_min) is untouched; default ⇒ horizon =
   training length, exactly upstream's T_max.
8. **Pair metrics** — at each eval tick, the three centered-cosine
   distributions (`pair/<distribution>/backbone`) between own-key and
   swapped-key renders of the eval images.  Built only when a watermark is
   configured: the clean run has no keys to swap, logs no `pair/` metrics, and
   exercises the unmodified path.

Forbidden inside the file (and absent): any change to the optimization math or
augmentations; any import from our directories outside marked blocks.

## Troubleshooting

* **`ImageNet-1k shards not found`** — run `prepare_data.py`; the runner
  refuses to start without the data.
* **CUDA out of memory** — the shipped batch (256, ViT-S/8 at 128px, V=4)
  peaks around 42 GB; a 48 GB card fits it only with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the runner sets it).
  There is no sanctioned smaller-batch config: batch size changes the training
  statistics this experiment measures.
* **Throughput** — `num_workers: 16` in `configs.yaml` assumes a large-core
  box; lower it if dataloader workers thrash.
