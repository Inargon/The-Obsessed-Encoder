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

Same-episode negatives are essential: every candidate has the same constant
colour tag, so that tag cannot identify the true action-conditioned endpoint.

Run unit tests, then the 500-step pilot:

```bash
python -m pytest \
  leworldmodel/additional_files/tests/test_control_objectives.py \
  leworldmodel/additional_files/tests/test_control_configs.py -q
sbatch leworldmodel/additional_files/slurm_control_pilot.sbatch
```

Only after both pilot arms complete with finite control metrics should the two
full seed-0 arms be launched with `slurm_control_full.sbatch`.
