# Experiment log — 2026-09-05

## Outcome

The first allocation/control candidates produced mixed results. Lower JEPA
prediction loss is not counted as success because the Obsessed Encoder failure
itself produces deceptively low prediction loss. Planner success and the
same-content versus same-tag signature are recorded separately.

### Conditional allocation, seed 0 (job 85383)

Final checkpoint, step 139329:

- prediction loss: 0.00652
- backbone same-content cosine: 0.32684
- backbone same-tag cosine: 0.63883
- latest planner success: 0.54
- recent-five-checkpoint mean planner success: 0.56
- best planner success: 0.64 (step 64000)

The method restored substantial planning ability without an auxiliary action
loss, but did not repair the shortcut signature (`same_tag > same_content`).

### Masked reachability, seed 0 (final checkpoint)

Early measurements looked collapsed, but by step 97500 the method showed a
late representation recovery. At the final checkpoint, step 139329:

- prediction loss: 0.01015
- inverse dynamics loss: 0.11734
- reachability loss: 0.24900
- reachability accuracy: 0.92839 (chance is approximately 0.25)
- backbone same-content cosine: 0.47017
- backbone same-tag cosine: 0.51750
- latest planner success: 0.66
- recent-five-checkpoint mean planner success: 0.66
- best planner success: 0.74 (step 110000)

The early conclusion that the auxiliary task was quarantined was premature.
The final backbone gap `same_tag - same_content` fell from 0.312 for the
conditional arm to 0.047, while recent planner success improved by 0.10 and
best success improved by 0.10. This is the leading method. Because both arms
use only seed 0, the result is a strong candidate signal rather than a
statistically confirmed improvement.

### Multi-lag IDM pilot, seed 0 (commit 60971e2)

At step 500:

- h=1 inverse loss: 0.37324 -> 0.39824
- h=2 inverse loss: 0.28027 -> 0.29279
- h=3 inverse loss: 0.20392 -> 0.22043
- backbone same-content cosine: 0.27186
- backbone same-tag cosine: 0.66418

None of the inverse losses improved and tag similarity dominated. Do not run
this configuration to full horizon.

### Context/dynamics factorization pilot, seed 0 (job 85449)

At step 500:

- prediction loss: 0.22347
- inverse dynamics loss: 0.37321 -> 0.39635
- reachability loss: 2.78311 -> 2.28385
- reachability accuracy: 0.23958 -> 0.32292
- context consistency loss: 0.14029 -> 0.09680
- dynamic static-leak ratio: 0.64453 -> 0.82813
- backbone same-content cosine: 0.44647
- backbone same-tag cosine: 0.51090

The whole-embedding shortcut gap was already small at step 500, which is an
encouraging representation signal. However, the intended dynamics subspace
became more dominated by episode-static variance, inverse dynamics worsened,
and prediction learning was slow. The current hard split plus trajectory
centering should not be promoted to a full run. Its useful idea should be
retained, but absolute controllable state must not be treated as nuisance.

## Engineering checks that passed

- Allocation smoke: three methods completed 20 steps with finite gradients.
- Scale-invariant local-rank implementation: six unit tests passed.
- Control pilot: both arms completed 500 steps without NaNs or exceptions.
- Stable-pretraining caches were redirected from the shared account quota to
  `/grp01/ids_compcog/song/cache/stable-pretraining`.

## Decision

Keep masked reachability as the only full-run winner. Do not extend multi-lag
IDM or the current factorized configuration. A revised factorized method should
learn a nuisance projector from appearance/tag interventions while preserving
absolute controllable state for IDM and reachability, rather than subtracting
each trajectory mean. Before another long run, log same-content and same-tag
geometry separately for the learned control and nuisance subspaces.
