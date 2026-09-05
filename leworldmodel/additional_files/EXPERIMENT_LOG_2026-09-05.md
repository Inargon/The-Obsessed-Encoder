# Experiment log — 2026-09-05

## Outcome

The first allocation/control candidates produced mixed results. Lower JEPA
prediction loss is not counted as success because the Obsessed Encoder failure
itself produces deceptively low prediction loss. Planner success and the
same-content versus same-tag signature are recorded separately.

### Conditional allocation, seed 0 (job 85383)

At step 119400:

- prediction loss: 0.00630
- backbone same-content cosine: 0.32855
- backbone same-tag cosine: 0.63743
- latest planner success: 0.56
- best planner success: 0.64

The method restored substantial planning ability without an auxiliary action
loss, but did not repair the shortcut signature (`same_tag > same_content`).

### Masked reachability, seed 0

Early measurements looked collapsed, but by step 97500 the method showed a
late representation recovery:

- prediction loss: 0.00905
- inverse dynamics loss: 0.11718
- action-cycle loss: 0.03047
- reachability loss: 0.23472
- reachability accuracy: 0.93359 (chance is approximately 0.25)
- backbone same-content cosine: 0.47638
- backbone same-tag cosine: 0.51188
- latest planner success: 0.64
- best planner success: 0.70

The early conclusion that the auxiliary task was quarantined was premature.
The late backbone geometry nearly crossed to `same_content > same_tag`, while
planner success exceeded the conditional arm. Treat this as the leading
candidate pending its final checkpoint and repeated seeds.

### Multi-lag IDM pilot, seed 0 (commit 60971e2)

At step 500:

- h=1 inverse loss: 0.37324 -> 0.39824
- h=2 inverse loss: 0.28027 -> 0.29279
- h=3 inverse loss: 0.20392 -> 0.22043
- backbone same-content cosine: 0.27186
- backbone same-tag cosine: 0.66418

None of the inverse losses improved and tag similarity dominated. Do not run
this configuration to full horizon.

## Engineering checks that passed

- Allocation smoke: three methods completed 20 steps with finite gradients.
- Scale-invariant local-rank implementation: six unit tests passed.
- Control pilot: both arms completed 500 steps without NaNs or exceptions.
- Stable-pretraining caches were redirected from the shared account quota to
  `/grp01/ids_compcog/song/cache/stable-pretraining`.

## Next hypothesis (not yet tested)

Direct action-conditioned reachability contrast in the unprojected embedding:
use the JEPA prediction as the query and all endpoints from the same episode as
negatives. This removes the learned projection head that allowed control
information to live in a small auxiliary subspace. The prototype is commit
`f51ce4a`; it must pass unit tests and a 500-step similarity pilot before any
full run.
