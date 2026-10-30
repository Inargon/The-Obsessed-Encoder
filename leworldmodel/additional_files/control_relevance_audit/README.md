# Frozen control relevance audit

Evaluation-only diagnostics. No training code, old results or checkpoints are changed.

## Cluster smoke

From the decision-aligned repository root:

    sbatch leworldmodel/additional_files/control_relevance_audit/slurm_smoke.sbatch

Uses the existing obsessed-encoder-py312 environment, gpu_shared, one GPU, two CPU
cores, 16 GB RAM, 30 minutes. The two clean matched epoch-10 runs are:

- pusht_clean_jepa_matched_full_seed0
- pusht_clean_ours_matched_full_seed0

Cache root: /grp01/ids_compcog/song/swm. Dataset is passed by absolute H5 path.
No dependency installation or network model downloads (HF_HUB_OFFLINE=1).
Results use a new Slurm-job-specific directory. Existing results are not overwritten.

## Measurements and limitations

Each checkpoint encodes the same 128 windows (two history images, frameskip=1)
and records their raw state history and five subsequent native action labels.
Native state is seven-dimensional according to preflight 111913. Every raw column
is preserved and probed separately. No unverified coordinate, angle, velocity or
contact meaning is assigned. Raw future-minus-current targets do not wrap angles;
interpretation of those columns must wait for schema verification.

Ridge probes test current-state recoverability and recorded state changes using
(a) action only, (b) features+actions, (c) recorded-state history+actions. Baseline
(c) is not asserted to contain the entire physical simulator state.

Split is 60/20/20 by complete episode ID. Standardization fits training episodes
only; alpha is selected on validation episodes only. Final fit stays train-only.
The runner verifies identical sampled rows, states, actions, and episode splits
across both checkpoints. Reports include checkpoint SHA256 and export provenance.
Probe-heldout episodes are not necessarily held out during encoder pretraining.
The 128-window run is a pipeline smoke; it is not a well-powered comparison.

These are observational linear probes, NOT same-start action interventions, the
model's original forward-predictor performance, or closed-loop success rates.
Poor linear R2 alone does not establish that information is absent. Follow up with
matched nonlinear probes, simulator-controlled action branches and actual planning.

## Dataset adapter

Uses the existing LeWM image preprocessor and a strict Hydra checkpoint loader.
Only training-only control_objective.* checkpoint keys are discarded. Other weight
incompatibilities fail. Requires native episode_idx/ep_idx and step_idx metadata;
never invents clip-based episode IDs. Unexpected temporal layouts fail explicitly.
The real cluster run is needed to validate this adapter against its installed SWM.

Standalone commands from leworldmodel:

    python additional_files/control_relevance_audit/export_features.py --run RUN --checkpoint weights_epoch_10.pt --dataset /absolute/pusht_expert_train.h5 --cache-dir /actual/cache/root --out /new/features.npz
    python additional_files/control_relevance_audit/audit.py --input /new/features.npz --out /new/audit.json

Input NPZ requires features [N,D], state/previous_state/next_state [N,S],
action [N,H*action_dim], episode_id [N]. Exported bundles mark state_schema=raw.
Legacy five-dimensional manually prepared bundles without state_schema use the
old pusher_xy/block_xy/angle-radians schema; this is not applied to raw exports.
Do not mix horizons, units or state schemas within a bundle.

## Tests

    python -m unittest discover -s leworldmodel/additional_files/control_relevance_audit -p test_audit.py -v

Tests use synthetic arrays. Passing tests is not empirical research evidence.

## Index correction (2026-10-31)

Smoke job 111915 is superseded: the old exporter assigned episode labels using
clip IDs as raw row offsets, so its split is not validated. New exports use
clip_indices to obtain episode and local start, then add offsets[episode].
Each selected window is checked against raw state/action and episode/step metadata
before preprocessing. Bundles retain both clip_index and raw row_index.
Rerun schema inspection, then rerun smoke into a new job-specific output directory.
