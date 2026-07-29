# The Obsessed Encoder — reproductions

Companion code for the blog post
[**"The Obsessed Encoder"**](https://www.enigma.inc/posts/obsessed-encoder).
Each directory reproduces one of the post's cases end to end:
plant a faint, predictable feature in the training images of a published
self-supervised recipe, and watch the objective improve while the
representation empties out.

**Their code, our pixels.**  Each example's training code is the published
upstream, modified only inside marked, default-inert blocks:

```
# >>> obsessed-encoder: <what and why>
...
# <<< obsessed-encoder
```

Every marked block is inert at its defaults — with no watermark configured the
code paths, math, and augmentations are exactly upstream's.  Each example's
README carries its complete marked-change inventory.  Grep for
`obsessed-encoder` to see every line we touched.

| Example | Upstream | What is planted |
|---|---|---|
| [`lejepa/`](lejepa/additional_files/README.md) | [LeJEPA](https://arxiv.org/abs/2511.08544) minimal recipe (ViT-S/8) | per-image luminance watermark, opacity 0.05 |
| [`dinov3/`](dinov3/additional_files/README.md) | [DINOv3](https://arxiv.org/abs/2508.10104) `vitl_im1k_lin834` recipe (ViT-L/16) | the same watermark, opacity 0.1 |
| [`leworldmodel/`](leworldmodel/additional_files/README.md) | [LeWorldModel](https://arxiv.org/abs/2603.19312) on PushT ([lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) @ `8edfeb3`) | a 5×5 corner square, coloured once per episode |

## The three groups

Every experiment runs the same three configurations (the blog's
baseline / test / control):

* **baseline — `clean`**: unmodified data.
* **test — `watermarked`**: a faint per-image pattern, repeated across the frame —
  a *predictable* feature every augmented view shares.
* **control — `random_control`**: the identical renderer and per-tile energy, but a
  different pattern at every tile position — matched pixels, nothing
  predictable to capture.

The LeWM case instantiates the same template with its own group names
(`baseline` / `colored_square_episode` / `colored_square_frame`, plus the
task-semantic `randgoal` exhibit) — see its
[README](leworldmodel/additional_files/README.md) for the mapping.

`random_control` differs from `watermarked` in exactly one knob (the repeat
flag); `clean` carries no watermark configuration at all.  Tests enforce this
discipline (`test_configs.py` in each example).

## Quickstart (uv)

Needs [uv](https://docs.astral.sh/uv/), an NVIDIA GPU, and a C/C++ compiler on
`PATH` (`apt-get install gcc g++`) — the DINOv3 recipe trains under
`torch.compile`, whose Triton backend builds a small C extension at runtime.

```bash
uv sync                                              # exact locked environment
uv run python lejepa/additional_files/prepare_data.py      # IN-1k data (~50 GB incl. Arrow cache)
uv run python lejepa/additional_files/run.py --seeds 3 --gpus 0
uv run python lejepa/additional_files/run.py --plot-only   # figures from local runs
```

The DINOv3 example's preparation also fetches the NYU-depth protocol data for
its dense probe (a ~6.7 GB Google Drive pull; `fetch_nyu.py` remains the
standalone/retry path and accepts a browser-downloaded archive):

```bash
uv run python dinov3/additional_files/prepare_data.py
uv run python dinov3/additional_files/run.py --seeds 3 --gpus 0,1,2
```

Every run the runner launches is printed as a single human-runnable command —
copy it to reproduce any run by hand.  Results land under `RESULTS_DIR`
(default `./results`), one directory per run, with `metrics.jsonl` as the
source of truth for every figure and `summary.json` as the completion marker
(re-invocations skip completed runs; `--force` reruns).  Datasets live under
`DATA_DIR` (default `./data`).  wandb is optional monitoring: `WANDB_MODE=offline`
or `disabled` is fully functional.

## Reproducing on a fresh GPU box

The published results were produced with this exact lockfile on the public
PyTorch runtime image, `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`.  From
a bare container (or any fresh CUDA box), four setup steps recreate that
environment:

```bash
# 1. system packages: a C/C++ compiler (torch.compile's Triton backend builds a
#    small extension at runtime) and headless-Chrome's shared libraries
apt-get update && apt-get install -y --no-install-recommends gcc g++ git \
  libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libdbus-1-3 \
  libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
  libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 libxshmfence1 fonts-liberation curl

# 2. the locked Python environment

curl -LsSf https://astral.sh/uv/0.10.7/install.sh | sh
uv sync --frozen

# 3. Chrome for figure export (plotly renders through kaleido, which drives a
#    real browser)
uv run plotly_get_chrome -y

# 4. wandb: offline unless you have credentials (the runners default to live
#    logging; without a key they crash at run start)
export WANDB_MODE=offline    # or: wandb login

# 5. data + runs, exactly as in the uv quickstart above
uv run python lejepa/additional_files/prepare_data.py
uv run python lejepa/additional_files/run.py --seeds 3 --gpus 0
```

**wandb**: runs work fully offline (`WANDB_MODE=offline` or `disabled`); the
offline records land inside each run directory and can be uploaded later from
any machine with credentials (`wandb sync results/<run>/wandb/offline-run-*`).

**Interrupted boxes lose nothing**: completed runs are skipped via their
`summary.json`, an interrupted DINOv3 run resumes natively from its
checkpoint, and an interrupted LeJEPA run restarts (the upstream minimal
recipe has no checkpointing — a deliberate non-addition).

## Hardware and runtimes

Measured on dedicated H100-80GB pods (release campaign, one GPU per run);
collapse onset is visible within ~1.5 h in every system.

| Example | Minimum GPU | Approx. runtime per run |
|---|---|---|
| LeJEPA (30K steps, bs 256, V=4) | 48 GB-class GPU (peak ~42 GB) | ~2.5 h |
| DINOv3 (50K iterations, ViT-L bs 128) | 40 GB-class GPU (peak ~24 GB — the shipped config carries upstream's activation checkpointing; without it, iBOT's variable masking peaks past 80 GB) | ~10–11 h (incl. the dense NYU tail) |
| LeWM (10 epochs, PushT) | any recent CUDA GPU | ~2–7 h (randgoal fastest) |

## Expected results

Filled in from the release runs (3 seeds per group); see each example's README
for the qualitative reading in the meantime.

## Version fidelity

The obvious objection to "we ran their code and it degraded" is "you ran it on
different or buggy dependency versions."  Three answers, in increasing order of
strength:

1. **Pin-to-upstream mapping.**  Science-relevant packages are pinned exactly
   (see `pyproject.toml` / `uv.lock`), chosen to satisfy each upstream's own
   declared environment:

   | Upstream declares | We pin | Why |
   |---|---|---|
   | LeJEPA: torch / torchvision / timm (unpinned) | torch 2.9.1, torchvision 0.24.1, timm 1.0.27 | current stable at implementation time; the fidelity anchor below reproduces the published curve on it |
   | DINOv3 `conda.yaml`: python 3.11, torch (unpinned) | python 3.12, torch 2.9.1 | one shared environment for all examples; the published NYU-depth number reproduces exactly on it (anchor below) |
   | DINOv3: numpy (unpinned; release predates numpy 2 scalar-promotion) | numpy 2.2.6 + a marked float32 restore in the depth-eval wrapper | see `dinov3/additional_files/dense_eval.py` — values untouched, dtype restored to what numpy 1.x produced |
   | figure rendering (not science-bearing) | plotly 6.9.0, kaleido 1.3.0, Chrome-for-Testing via `plotly_get_chrome` | kaleido ≥1 renders through a real browser; the Chrome build is fetched pinned at setup (step 3 above) |

2. **Fidelity anchors.**  Upstream-published numbers reproduced in this exact
   environment: the LeJEPA minimal recipe's Imagenette accuracy (**~90.4%**
   top-1, matching the published ~90.7% curve), and DINOv3's published
   NYU-depth RMSE (**0.349** — exact reproduction via the vendored protocol on
   the BTS data).  A broken environment does not reproduce published numbers.

3. **The control argument** (the strong one).  All three groups share one
   environment, one lockfile, one container.  A version bug would have to
   affect the `watermarked` group but not the identically-run `random_control` —
   the test-vs-control comparison is version-confounder-immune by construction.

## Tests

```bash
uv run pytest                 # CPU suite
uv run pytest -m gpu          # end-to-end smokes (needs a CUDA device)
```

