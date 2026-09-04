# Capacity-allocation experiment

This first experiment tests three new regularizers on the **same per-episode
coloured-square failure case**. It uses the published PushT expert dataset and
adds the predictable 5x5 tag online. It intentionally does not rerun the
published clean, random-control, or unregularized test arms.

## Arms

- `conditional`: variance and decorrelation after subtracting each episode's
  temporal mean. Episode-constant features cannot satisfy this loss.
- `local_rank`: a differentiable hinge on neighborhood effective rank. This
  targets the low-dimensional folded manifold shown in the blog post.
- `hybrid`: both terms together.

The original SIGReg and prediction losses remain enabled in all arms. The only
difference among arms is `loss.allocation.*`.

## Run

From `leworldmodel/`, with the published environment and RandGoal dataset
already prepared:

```bash
python additional_files/run_allocations.py --seeds 3 --gpus 0,1,2
```

This launches exactly nine runs: three methods times seeds 0, 1, and 2. To
shake out all three methods cheaply before the full campaign:

```bash
python additional_files/run_allocations.py \
  --seeds 1 --gpus 0 \
  --extra-opts '+trainer.max_steps=20 +trainer.log_every_n_steps=1 ++eval.every_n_steps=100000 ++pair.every_n_steps=100000'
```

Inspect the exact commands without running them:

```bash
python additional_files/run_allocations.py --seeds 1 --gpus 0 --dry-run
```

Results default to `leworldmodel/results/allocation/`. Override with the
`RESULTS_DIR` environment variable or `--results-dir`.

The original reproduction figure factory is skipped because it assumes the
published four-arm grid and tries to load RandGoal. Allocation runs retain the
complete `metrics.jsonl` streams; a dedicated comparison plot is produced only
after the three methods pass the shakeout.

## Logged diagnostics

Alongside prediction loss, pair similarity, and planning success, the new arms
log their component losses and `fit/local_effective_rank`. A successful method
should recover content similarity and planning under the predictable tag
without merely making the training objective look healthier. The best method
will then be promoted to the harder task-semantic RandGoal experiment.
