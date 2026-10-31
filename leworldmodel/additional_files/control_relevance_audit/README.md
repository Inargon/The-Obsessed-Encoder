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

## Multiple split readouts

`slurm_readout.sbatch` reuses the corrected exports from job 111924 (override with
AUDIT_INPUT_DIR). It evaluates split seeds 73/74/75 with linear ridge and a
nonlinear random-Fourier feature ridge probe with a linear skip. The feature
map seed is fixed; width is 256. Scaling, centering and target normalization
use only training episodes; bandwidth and regularization use validation only.
Outputs include all shared baselines, per-column metrics, selected hyperparameters,
source hashes, and episode split lists. No encoder is trained and no new GPU
feature export is performed. The current submission requests gpu_shared for
compatibility with the known cluster queue, although computation is CPU/NumPy.
Repeated splits do not measure encoder training-seed uncertainty. This limited
nonlinear family cannot establish information absence if it fails. These are
probe-held-out episodes, not verified encoder-training-held-out episodes.
Raw angular columns remain descriptive and require a later periodic target audit.

## Delayed pair single-frame readout

`slurm_pair_readout.sbatch` reads fixture job 112313 (override PAIR_INPUT_DIR).
One initial RGB image per scene is encoded with each frozen checkpoint and its
projector. No repeated-image history is fabricated. Probe targets are the three
terminal candidate costs centered per scene. Linear and RFF probes use the same
pair-group splits (73/74/75) and validation search. Both scenes stay in the same
split. Recorded initial state and train-mean candidate costs are baselines.
Reports retain test pair IDs, chosen and true candidate actions, candidate regret,
and separate metrics for the fixed delayed/choice-difference mask. All pairs train
the probe; that mask is used only for descriptive reporting. Thirty-two pairs are
only a pipeline check. This does not test closed-loop control, a new method, or
match the original two-frame probe protocol. Runtime checkpoint loading and image
preprocessing require verification in the cluster environment.

## Larger varied-goal diagnostic

`slurm_goal_pairs.sbatch` collects 256 independent pairs with seed 902, then runs
single-frame readouts automatically. Goal poses are sampled before rollout and
shared by both scenes and all branches in each pair. Stored goals determine the
terminal candidate costs; angle differences are wrapped. Goal x/y/sin/cos are
explicit inputs for both encoders and recorded-state probes. Goal-only probes and
a train-mean action baseline expose goal-conditioned or unconditional preferences.
No label balancing or favorable-subset training is performed; best-action counts
are recorded. Split 73 is the primary fixed pair-group split; 74/75 are sensitivity
checks on overlapping data. More pairs and goal variation do not guarantee that
action-label imbalance is removed or that the recorded-state probe succeeds.
Retain all results even if baseline dominance persists. New output is job-specific.

## Recorded-state terminal prediction positive-control attempt

`slurm_state_terminal.sbatch` reuses fixture 112328 without rendering or encoder
training. Each fixed candidate sequence has a terminal block x/y/sin/cos target.
Initial recorded state (periodic angle encoding) is the only dynamics input;
goals are used solely in analytic terminal scoring. True terminal poses must
reproduce the saved costs before any fitting. Linear ridge, RFF ridge and a small
MLP use training-only normalization and validation-only hyperparameter/epoch
selection. Split 73 remains primary, 74/75 sensitivity checks. Baselines include
training-mean candidate costs and a stationary block. The MLP uses seed 901,
widths 64/128, weight decay 0/.001, 600 epochs with validation every 10 epochs.
Local tests cover analytic scoring and data grouping; torch fitting requires the
cluster runtime. No success is assumed. If the state-input control is inadequate,
representation deficiency cannot be established by this protocol.

## Matched visual terminal readout

`slurm_visual_terminal.sbatch` reuses fixture and cached projected features from
112328. Fixture SHA256, row IDs, initial states and goal suffixes must match.
The four appended goal coordinates are removed before terminal prediction;
rendered images still contain the target outline, so this is not a goal-free
visual representation. Recorded state, JEPA and ours use identical targets,
pair splits and linear/RFF/MLP search settings. Input dimensionality and hence
parameter counts differ. The three output heads correspond to the three fixed
candidate sequences; there is no generalization claim to unseen action sequences.
No new render, feature export or encoder training is performed. Source artifacts
are retained; results use a new job-specific directory.
