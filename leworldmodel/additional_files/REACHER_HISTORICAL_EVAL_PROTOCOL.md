# Clean Reacher historical evaluation contract

This file records the only protocol under which clean Reacher success rates in
the article are comparable. It was recovered from Slurm job `91315` and Hydra
run `le-wm-repro-20260514/outputs/2026-09-23/23-10-51` on 2026-10-01.

## Runtime

- Repository: `/grp01/ids_compcog/song/code/le-wm-repro-20260514`
- Python: `/grp01/ids_compcog/song/envs/lewm-repro-py310/bin/python`
- `LOCAL_DATASET_DIR=/grp01/ids_compcog/song/swm/datasets`
- Python 3.10.21
- stable-worldmodel 0.0.6
- stable-pretraining 0.1.6
- torch 2.7.1+cu118
- numpy 2.2.6
- mujoco 3.8.0

The modern Python 3.12 stack is not an interchangeable Reacher evaluator.

## Checkpoint conversion

Modern ViT layer keys are renamed to the historical HuggingFace layout. The
12-layer ViT-tiny conversion renames exactly 192 tensors. Training-only
`control_objective.*` tensors are removed. Tensor values are not modified.
Use `convert_reacher_historical_checkpoint.py`; it refuses overwrites and
emits SHA-256 provenance.

## Hydra contract

- config name: `reacher`
- `policy=<absolute historical checkpoint>`
- `+cache_dir=/grp01/ids_compcog/song/swm/datasets`
- `eval.dataset_name=dmc/reacher_random`
- `dataset.keys_to_cache=[action]`
- `seed=42`
- `solver.n_steps=30`

The PyPI `stable-worldmodel==0.0.6` wheel does not contain the LeWM model-loader
module. The runner therefore prepends the pinned companion source checkout
`stable-worldmodel-repro-20260514` to `PYTHONPATH`, records its Git commit, and
explicitly imports `stable_worldmodel.wm.utils` before executing `eval.py`.

Resolved planning settings are 300 CEM samples, top-30 elites, horizon 5,
receding horizon 5, action block 5, goal offset 25, evaluation budget 50, and
50 episodes.

The recovered Hydra metadata recorded the cache root as `.../swm`, but the
historical `eval.py` passes `cfg.cache_dir` directly to `HDF5Dataset`. The old
parent-level path no longer resolves. The campaign therefore pins the extant
file-equivalent dataset directory `.../swm/datasets` and opens it through the
historical loader before submitting any GPU job.

## Regression gate

Before accepting a new method, one matrix run must reproduce the frozen
checkpoint baselines: JEPA 90%, Full 86%, and Inverse+Reach 76%. The
`reacher_historical_eval_campaign.py` campaign enforces these values before a
new arm is considered comparable. A predictor-only Cycle score of 62% obtained
with the modern Python 3.12 evaluator is explicitly cross-protocol and invalid.
