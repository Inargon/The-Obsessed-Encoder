# LeWM reproduction — the world-model case of the obsessed encoder

This directory holds everything *we* added around the upstream LeWM code. The
example's root (`leworldmodel/`) is a copy of the published
[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) repository at commit
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, edited **only** inside marked blocks:

```
# >>> obsessed-encoder: <what/why>
...
# <<< obsessed-encoder
```

`../changes.diff` is the complete diff against upstream — the only touched
file is `train.py`, and every marked change is individually inert: with its
config keys absent, the code path is exactly upstream's. `eval.py`, `jepa.py`,
`module.py`, `utils.py` and `config/` are byte-identical to upstream.

## Layout

```
additional_files/
├── run.py                  # the reproduction CLI (train the arms, render figures)
├── configs.yaml            # the four arms + shipped cadences
├── pixel_tag.py            # the on-the-fly corner colour tag (dataset seam)
├── stimuli.py              # controlled sim rendering (pair analysis + encoder readout)
├── callbacks/              # everything train.py's marked blocks attach
├── graphs/                 # all figure logic (reads local results only, no GPU)
└── pusht_random_dest/      # the scripted random-destination dataset generator
```

## Marked-change inventory (`../train.py`, in file order)

| # | Block | Activated by | Default (absent) |
|---|-------|--------------|------------------|
| 1 | Import of our additions (`additional_files`) | — | — |
| 2 | Full seeding: `pl.seed_everything(cfg.seed, workers=True)`; shipped configs run `seed=0` | `+seed_everything=true` | upstream: only the train/val split is seeded |
| 3 | Corner colour tag: the dataset reader swapped for a subclass stamping a seeded n×n square on the raw clip before preprocessing (`pixel_tag.py`) | `+pixel_tag.mode=video\|frame`, `+pixel_tag.size=5` | exact upstream reader |
| 4 | Metrics mirror + wandb naming: every logged payload appended to a `metrics.jsonl`, the figures' only data source (`callbacks/jsonl_logger.py`); when wandb is enabled, its `log_metrics` is rebound so the wandb stream uses the repo-wide unified sections (`callbacks.unify_wandb_keys`; the jsonl keeps upstream keys) | `+metrics_jsonl_path=<path>` (the wandb rename rides `wandb.enabled`) | upstream logger wiring |
| 5 | Opt-in callbacks, one factory per row (`callbacks/`): goal-reaching eval logging `eval/success_rate` in [0, 1]; pair analysis (one cosine suite per run); wandb preview of training episodes; step-interval checkpoints in the upstream save format | `+eval.every_n_steps`, `+pair.every_n_steps`, `+preview.num_episodes`, `+checkpoint.every_n_steps` | empty list |
| 6 | Trainer wiring: hands the callbacks/loggers built above to `pl.Trainer` | — | exact upstream call |

Forbidden inside the file: any change to the optimization math, and any use of
our code outside marked blocks.

## The four arms

| Public name | What varies | Role |
|---|---|---|
| `baseline` | original PushT expert data, fixed goal | baseline |
| `colored_square_episode` | 5×5 corner square, one random colour **per episode** | test — the predictable feature |
| `colored_square_frame` | same square, fresh colour **every frame** | control — matched, never repeats |
| `randgoal` | destination-T pose sampled per episode (scripted expert demos) | the natural, task-semantic key |

`colored_square_episode` and `colored_square_frame` differ **only** in the tag
mode — train and eval — and `baseline` carries no tag keys at all. Every arm
evaluates on the same kind of data it trained on: the square arms plan on
frames carrying their tag mode, `randgoal` evaluates on the scripted dataset.

Shipped defaults (`configs.yaml`): 10 epochs (the paper's operating point),
eval every 2000 steps, pair analysis every 100 on the three perturbed arms
(baseline logs no pair metrics), checkpoints every 500 steps, seeds from
`--seeds` (default one run, seed 0);
everything else is the upstream config.

## Quickstart

uv path (one environment for the whole repository; from `leworldmodel/`):

```bash
(cd .. && uv sync)
PYTHONPATH=.. uv run --project .. python additional_files/run.py
```

Docker path (the repository's single image; from the repo root):

```bash
docker build -t obsessed-encoder .
docker run --gpus all \
  -v $PWD/results:/artifact/results -v $PWD/data:/artifact/data \
  obsessed-encoder bash -c \
  'cd leworldmodel && PYTHONPATH=/artifact python additional_files/run.py'
```

Results land under `RESULTS_DIR/<config>_seed<k>/` (`metrics.jsonl` +
`summary.json`); a run whose `summary.json` exists is skipped (`--force` /
`--rerun <name>` restart from scratch). wandb is monitoring only (silence it
with `WANDB_MODE=offline|disabled`; the Docker image defaults to offline) —
figures read the local files exclusively. Note `--data-dir` is
inert for this example: datasets resolve through the stable-worldmodel cache
(`STABLEWM_HOME`), not through a data directory.

## Datasets

* **Expert set** (`pusht_expert_train.h5`, 18,685 episodes -- the name the shipped configs load):

```bash
uv run hf download quentinll/lewm-pusht pusht_expert_train.h5.zst \
    --repo-type dataset --local-dir /tmp/lewm_expert
uv run python -c "import zstandard, pathlib, os
home = pathlib.Path(os.environ.get('STABLEWM_HOME', pathlib.Path.home() / '.stable_worldmodel'))
(home / 'datasets').mkdir(parents=True, exist_ok=True)
with open('/tmp/lewm_expert/pusht_expert_train.h5.zst', 'rb') as src, \
     open(home / 'datasets' / 'pusht_expert_train.h5', 'wb') as dst:
    zstandard.ZstdDecompressor().copy_stream(src, dst)"
```

  (The resolver reads `$STABLEWM_HOME/datasets/<name with extension>`; the
  upstream README's root placement and extension-less names predate it.)
* **Random-destination set**: download the published `pusht_scripted_goal_train`
  dataset, or regenerate it —

```bash
python additional_files/pusht_random_dest/collect.py \
    --name pusht_scripted_goal_train --episodes 18685 --num-workers 8
```

  `policy.py` is the scripted geometric push expert, `collect.py` the
  simulation/rollout side, `dataset_writer.py` the Lance schema writer. Only
  successful episodes are written; the default count matches the expert set.

## Reproducing the blog figures

1. Everything at once: `python additional_files/run.py` trains the arms and
   renders all figures into `RESULTS_DIR/figures/` (`--plot-only` to render
   from existing results).
2. Individual figures:
   - Datasets montage (fig 7): `python additional_files/graphs/dataset_montage.py`
     — writes the static grid plus an animated GIF that plays whole episodes
     (two fixed-goal, two random-goal, one per tag mode).
   - Crossover (fig 8) + the pair cosine (fig 9, every perturbed arm on one
     row; the backbone read is the blog's, the projection read is written
     alongside it):
     `python additional_files/graphs/training_curves.py --results-dir <RESULTS_DIR>`
     — built on the shared `common.plotting` factory, so the bands are mean ±
     std over whatever seeds are present.
   - Encoder readout (fig 10): `python additional_files/graphs/encoder_readout.py --run-name randgoal_seed0`
     — loads a run checkpoint and reads the encoder directly (no fitted
     probe). Two panels sharing one wander box and one seed, so the moving tee
     traces the same trajectory in both and they differ only in which tee
     moves: the goal-T, then the block-T. 800 points per path (`--points` to
     change). Writes an animated GIF that replays the stimulus above the path
     as it draws.

## Single run by hand

`run.py` prints every training command it executes; e.g. the predictable arm:

```bash
cd leworldmodel && PYTHONPATH=.. python train.py \
  output_model_name=colored_square_episode subdir=colored_square_episode \
  trainer.max_epochs=10 seed=0 +seed_everything=true +preview.num_episodes=2 \
  +metrics_jsonl_path=results/colored_square_episode/metrics.jsonl \
  +eval.every_n_steps=2000 +pair.every_n_steps=100 "+pair.suites=[colour]" \
  +checkpoint.every_n_steps=500 \
  +pixel_tag.mode=video +pixel_tag.size=5 +eval.tag_mode=video +eval.tag_size=5
```

## SIGReg and tag folding

From the repository root, the CPU-only synthetic audit compares two encoders
with identical empirical output distributions and different sensitivity to
the same tag intervention.

```bash
python -m leworldmodel.additional_files.sigreg_folding_audit synthetic \
  --out results/sigreg-folding-synthetic.json
```

The report includes the actual LeWorldModel SIGReg statistic, an i.i.d. Gaussian
reference with matching sample count, dimension and directions, and fixed
candidate rankings. The candidate bank shares a tag while the goal tag stays
fixed. Covariance participation ratio and known intrinsic dimension describe
the harmonic curve. The norm spread and largest axis score give complementary
readouts of this particular construction.

For an existing tag-folding export, run the same statistic on raw embeddings
without loading a model or regenerating pixels.

```bash
python -m leworldmodel.additional_files.sigreg_folding_audit embeddings \
  path/to/tag-folding-probe-embeddings.npz --out results/sigreg-folding-frozen.json
```

Pass `--sigreg-audit` to `tag_folding_probe.py` to include the audit in a new
probe report. These scores describe RGB sweeps conditional on a fixed physical
scene. Training SIGReg sees batches sampled across scenes, so those scores
answer a separate question.

For the existing counterfactual planning training arm, append
`+loss.counterfactual_planning.ranking_metric=centered_mse` to its Hydra
overrides to compare centered candidate costs directly. The original
`js` metric remains the default. Use
`loss.counterfactual_planning.representation_weight=0` on both comparison
arms to isolate the relative-cost signal from full-embedding matching.

On the cluster, the tagged audit runs a small frozen-checkpoint smoke before
the full sweep. The archive includes the raw embeddings for offline reanalysis.

```bash
python leworldmodel/additional_files/tag_folding_probe_campaign.py \
  --submit --confirm-run --sigreg-audit --partition gpu_shared --nodelist ''
```

The counterfactual launcher can submit a matched pair of two-step training
checks. Each short run writes its resolved configuration and job manifest.

```bash
for metric in js centered_mse; do
  python leworldmodel/additional_files/counterfactual_planning_campaign.py \
    --submit --smoke-only --ranking-metric "$metric" --relative-cost-only
done
```

A warm-start pilot can compare centered cost matching with `margin_guard`.
The guard applies to a retagged action pair once the observed action beats
the shuffled action by the supervised margin in the reference view. It
penalizes a retagged gap below 25% of that detached reference gap. Both pilot
arms use action-margin supervision on the reference view, a fresh optimizer
at a constant learning rate of `1e-5`, and the same initialization.

```bash
for metric in centered_mse margin_guard; do
  python leworldmodel/additional_files/counterfactual_planning_campaign.py \
    --submit --pilot-steps 96 --init-checkpoint "$JEPA_INIT_CHECKPOINT" \
    --ranking-metric "$metric" --relative-cost-only
done
```

The tag intervention is configurable independently of the ranking metric.
Use `--tag-intervention context_only` to retag the predictor's context while
using the original future latent, or `goal_only` to keep the original
predictions and retag the future latent. `balanced` cycles through both-tag,
context-only, and goal-only costs over the shuffled batch rows. The default
is `both`. Matched loss comparisons use `--relative-cost-only` to hold the
full-embedding matching weight at zero.
