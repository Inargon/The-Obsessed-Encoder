# Control-grounded allocation experiment

The earlier one-step IDM established that action-relevant supervision resists
the episode-colour shortcut. This follow-up compares two stronger objectives:

- `factorized_reachability`: reserve a 32-dimensional context block and apply
  direct reachability, IDM, conditional variance/decorrelation, and a
  scale-invariant static-leak penalty to the remaining dynamic block.

- `direct_reachability`: combine the proven one-step IDM with same-episode
  endpoint contrast directly in the planner-facing embedding. There is no
  learned reachability projection head in which to quarantine the signal.

- `multi_horizon_idm`: recover the mean control over lags of 1--3 sampled
  frames (5--15 environment steps with the PushT frameskip). Exact long action
  sequences are deliberately not regressed because endpoints do not uniquely
  determine the path between them.
- `masked_reachability`: add shared random-subspace masking, an action cycle
  through the JEPA predictor, and endpoint classification against negatives
  drawn from the same episode.

The completed seed-0 run of `masked_reachability` reached 0.74 peak planning
success and 0.66 over its final five evaluations. Historical experiments later
provided by the project show that this does not exceed the 0.78 one-step IDM or
0.82 seed-0 MSID result. They also show that removing tag geometry is neither
necessary nor sufficient for planning success.

The next pilot therefore tests a narrower hypothesis. All three arms use the
historical MSID target (recover the complete action sequence at horizons 1--3)
and differ only in the coefficient on absolute JEPA prediction:

- `masked_sequence_pred1`: prediction weight 1.0;
- `masked_sequence_pred03`: prediction weight 0.3;
- `masked_sequence_pred0`: prediction weight 0.0.

Each arm also logs `reachability_shuffled_accuracy`: the reachability query is
given another sample's actions while its start, candidate endpoints, horizon,
and random feature mask remain fixed. A useful action-conditioned objective
must lose accuracy under this intervention. `reachability_action_margin` is
the correct endpoint's logit under the real action minus its logit under the
shuffled action.

Same-episode negatives are essential: every candidate has the same constant
colour tag, so that tag cannot identify the true action-conditioned endpoint.

Run unit tests, then the 500-step pilot:

```bash
python -m pytest \
  leworldmodel/additional_files/tests/test_control_objectives.py \
  leworldmodel/additional_files/tests/test_control_configs.py -q
sbatch leworldmodel/additional_files/slurm_control_pilot.sbatch
```

The original control pilot and two selected full seed-0 runs are complete;
their outcomes are recorded in `EXPERIMENT_LOG_2026-09-05.md`.

Run the 10000-step, three-arm prediction-weight sweep with:

```bash
sbatch leworldmodel/additional_files/slurm_masked_pred_sweep.sbatch
```

This is three methods at seed 0, not three seeds. Do not promote an arm merely
because its auxiliary accuracy rises: it must beat shuffled actions, preserve
content-facing backbone geometry, and improve planning success.
