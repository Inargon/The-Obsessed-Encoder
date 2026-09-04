# Experiment log — 2026-09-05

## Outcome

The first allocation/control candidates did **not** solve the episode-colour
shortcut. Lower JEPA prediction loss is not counted as success because the
Obsessed Encoder failure itself produces deceptively low prediction loss.

### Conditional allocation, seed 0 (job 85383)

At step 7100:

- prediction loss: 0.03657
- backbone same-content cosine: 0.30104
- backbone same-tag cosine: 0.66988

The reversed similarity ordering (`same_tag > same_content`) is a failure.

### Masked reachability, seed 0

At step 1600--1649:

- prediction loss: 0.04515
- inverse dynamics loss: 0.39174
- action-cycle loss: 0.12110
- reachability loss: 1.24709
- reachability accuracy: 0.40365 (chance is approximately 0.25)
- backbone same-content cosine: 0.10517
- backbone same-tag cosine: 0.85586

The auxiliary task learned, but its information was quarantined in a small
subspace and the planner-facing embedding collapsed even more strongly.

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
