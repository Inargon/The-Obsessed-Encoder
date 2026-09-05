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

## Decision-causal audit note

A downstream action head can improve imitation performance without making the
world prediction a causal mediator of the decision. Three claims must remain
separate:

1. actions are decodable from the latent;
2. the policy or planner changes when its predicted-future input changes;
3. that change follows a physically valid counterfactual and causes the
   corresponding change in the environment.

An eventual zero-training `Decision-Causal Scramble Audit` should therefore
intervene on future latents while holding the current state fixed. Random
off-manifold noise is not decisive. The preferred intervention uses valid
minimal pairs with matched pusher endpoints and divergent T-block outcomes,
then measures policy change, candidate-ranking change, and the realized
T-block effect. This is also the appropriate audit for Delta-JEPA-style
`latent -> action` imitation heads: action decodability alone does not establish
that a learned world prediction is used for control.

For the current LeWM experiments, the auxiliary IDM/reachability heads are
representation regularizers and are absent at deployment; CEM consumes the
world-model prediction. The newly logged `reachability_shuffled_accuracy` and
`reachability_action_margin` test whether the auxiliary reachability head uses
its action input, but they do not by themselves establish end-to-end decision
causality through CEM.

## Near-term priority

The next several days prioritize planning success on tagged PushT. The current
thresholds are the historical one-step IDM result (0.78) and seed-0 MSID peak
(0.82); clean performance is approximately 0.88--0.94 depending on the
evaluation run. Mechanistic audits remain important, but should not delay
short, controlled pilots that can raise success. The completed boundary pilot
tested full-sequence masked MSID at absolute-prediction weights 1.0, 0.3, and
0.0. The immediate follow-up tests the interior weights 0.3, 0.5, and 0.7 for
10000 steps, one seed each. Promote only an arm that both uses actions under
the shuffle intervention and improves planner success; pair geometry alone is
insufficient.

### Absolute-prediction boundary pilot, seed 0 (job 85558)

All three full-sequence masked-MSID arms completed 2000 steps. Planning at this
point was still near its known uninformative early baseline: weight 1.0 scored
0.00, weight 0.3 scored 0.04, and weight 0.0 scored 0.02. The representation
and optimization metrics nevertheless gave a clear boundary trade-off:

- weight 1.0: prediction loss 0.0367; same-content 0.188; same-tag 0.775;
  reachability accuracy 0.465 versus 0.303 with shuffled actions;
- weight 0.3: prediction loss 0.2776; same-content 0.776; same-tag 0.189;
  control loss improved 0.710 to 0.529; reachability accuracy 0.406 versus
  0.368 shuffled, with a positive 0.197 action margin;
- weight 0.0: prediction loss diverged to 1.944; same-content 0.967; same-tag
  0.009; reachability accuracy 0.413 versus 0.368 shuffled.

Thus absolute prediction weight 1.0 re-established the episode-tag shortcut,
whereas removing it entirely destroyed the predictive world model. Weight 0.3
was the only useful boundary candidate but showed weak action reliance. The
next 10000-step pilot scans weights 0.3, 0.5, and 0.7; the 0 and 1 boundaries
should not be extended.
