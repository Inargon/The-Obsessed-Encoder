# Paper Positioning: Prediction Should Not Unconditionally Shape the Encoder

Date: 2026-09-07

## Proposed thesis

Predictive JEPA training conflates two roles: fitting a predictor to an existing
representation and deciding which predictable factors should shape the shared
encoder. Under a tight latent bottleneck, a low-information nuisance can win
the second competition merely because it is easy to predict. Action or inverse
dynamics supervision makes control information decodable, but does not by
itself regulate how the prediction objective continues to reshape the encoder.

Our working method, control-guided prediction-gradient routing, keeps the full
prediction gradient for the predictor while directionally attenuating the
prediction gradient entering the encoder. The strongest defensible claim is
not that this solves every allocation failure. It is that predictable-nuisance
failure involves control-insensitive as well as negatively conflicting updates,
and that managing those updates can restore content-dominated representation
geometry without abandoning latent prediction.

## Closest literature and the remaining gap

### Enigma: The Obsessed Encoder

[The Obsessed Encoder](https://www.enigma.inc/posts/obsessed-encoder) shows
that small, predictable features can dominate large JEPA latent spaces while
anti-collapse statistics remain healthy. Its matched random-per-frame control
isolates predictability, rather than pixel corruption, as the cause. RandGoal
also demonstrates why removing all static or slow information is unsafe: the
task goal itself may be static. It diagnoses feature misallocation but does not
provide the optimization mechanism studied here.

### LeWorldModel and its analysis

[LeWorldModel](https://arxiv.org/abs/2603.19312) combines next-latent
prediction with SIGReg and plans in the resulting latent space. It establishes
the base system and its compact, stable recipe.

[When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379)
proves linear identifiability for stationary additive-noise worlds with
Gaussian latent variables, proves a converse for non-Gaussian alternatives,
and connects orthogonal linear identifiability to latent-space planning. Our
paper should complement rather than imitate this theory: study what happens
when finite-capacity nonlinear training and predictable nuisance factors cause
the empirical representation to violate the useful geometry that planning
needs.

[What Drives Success in Physical Planning with JEPA World Models?](https://arxiv.org/abs/2512.24497)
systematically studies architecture, objective, and planner choices and shows
that planning success is not explained by prediction loss alone. It is the
best stylistic model for an analysis-heavy empirical paper and a source of
matched planning protocols.

### Action-based representation objectives

[Sensorimotor World Models](https://arxiv.org/abs/2606.20104) uses inverse
dynamics as the sole anti-collapse mechanism. It supplies the cleanest IDM
baseline and the closest high-level argument that perception should be shaped
for action.

[Delta-JEPA](https://arxiv.org/abs/2606.31232) decodes actions from latent
differences rather than concatenated endpoints. Its central claim is
action-sensitive transition geometry. It must be compared or at least
implemented as a matched auxiliary objective; our distinction is that action
supervision also controls how the forward-prediction gradient enters the
encoder.

[No Gaussian Required / AC-MTM](https://arxiv.org/abs/2608.17542) uses
contrastive inverse dynamics as a distribution-free anti-collapse signal. It
matches SIGReg on average across four standard tasks and reports 80.0 +/- 2.0
versus 58.0 +/- 2.0 on OGBench Visual Scene. It is a critical baseline because
it may resist nuisance through stronger action discrimination without gradient
routing.

### Physical grounding, factorization, and counterfactual structure

[PhyLatent](https://arxiv.org/abs/2608.05720) identifies physical invariance,
identifiability, and counterfactual-dynamics failures and uses physical state
grounding, future alignment, static invariance, counterfactual separation, and
denoising. It reports OGBench-Cube MPC improvement from 70.0% to 78.1% and
TwoRooms from 81.0% to 98.0%. It is broader and more strongly grounded; our
advantage should be a smaller intervention requiring actions but no physical
state labels or handcrafted nuisance definition.

[Toward Physically Grounded JEPA World Models](https://arxiv.org/abs/2609.03565)
combines IDM with state alignment, reporting 100% on TwoRoom, 98% on PushT, and
87% on OGBench-Cube. This sets a privileged-grounding reference rather than a
fair target for an action-only method.

[Subspace-Decomposed JEPAs](https://arxiv.org/abs/2605.31111) explicitly
assigns progression and content to orthogonal latent subspaces and includes a
load-bearing subspace falsifier. It is closely related to allocation, but fixes
roles architecturally. Our method instead modifies online optimization of a
shared latent without preassigning coordinates.

[Causal-JEPA](https://arxiv.org/abs/2602.11389) uses object-level masking as
latent interventions to force interaction reasoning. It is relevant as a
structural way to remove shortcut solutions, but assumes an object-centric
representation and changes the prediction task rather than routing its encoder
gradient.

### Large policy-facing JEPA systems

[JEPA-WAM](https://arxiv.org/abs/2608.09381) shares a predictor between latent
transition modeling and continuous action generation and reports 79.2% on
LIBERO-Plus without large robot-policy pretraining. [VLA-JEPA](https://arxiv.org/abs/2602.10098)
uses leakage-free future-state prediction before action-head fine-tuning. These
show the broader relevance of separating predictive representation learning
from policy use, but are not the primary matched baselines for the compact
LeWM/PushT study.

### Gap after this review

Among the reviewed papers, existing methods primarily change what the latent
must encode (inverse dynamics, state grounding, counterfactual structure), how
coordinates are assigned, or how prediction is constructed. We did not find a
method that explicitly separates the prediction loss's update to the predictor
from its update to the shared encoder, then uses a control gradient to manage
both negatively conflicting and near-orthogonal prediction updates under a
predictable-nuisance failure. This is the candidate novelty; it must be stated
as a scoped finding rather than an exhaustive priority claim until the search
is expanded.

## Questions the paper should answer

1. Does action supervision alone prevent a predictable nuisance from
   dominating the representation geometry?
2. Is the failure explained by negative gradient conflict, or by the much
   larger set of control-insensitive prediction updates?
3. Can predictor learning be preserved while selectively limiting prediction's
   ability to reshape the encoder?
4. Is any gain due only to adaptive prediction down-weighting, or does the
   control-conditioned direction carry additional information?
5. Does the learned representation preserve physical and action-effect
   geometry, rather than merely remove the watermark?
6. Does the predictor genuinely influence planning decisions, or is it only an
   auxiliary representation regularizer?

## Propositions rather than a broad theory

Let `g_p` and `g_c` be prediction and control gradients with respect to the
embedding, and decompose `g_p = alpha g_c + g_perp`. The routed prediction
gradient is

```text
g_route = [alpha]_+ g_c + [cos(g_p, g_c)]_+ g_perp.
```

The paper can state and prove the following limited propositions.

1. Non-negative control projection:
   `g_c^T g_route = [g_p^T g_c]_+ >= 0`.
2. Explicit orthogonal attenuation:
   `||g_route||^2 = [alpha]_+^2 ||g_c||^2 + gamma^2 ||g_perp||^2`.
3. Under direct Euclidean optimization of the embedding, the routed prediction
   term cannot worsen the current control objective to first order.
4. None of these propositions guarantees improvement under the encoder
   parameter Jacobian, Adam, long-horizon training, or an incomplete control
   objective. Long-term representation claims remain empirical.

This is enough formal structure for a solid analysis paper without claiming a
general allocation theorem.

## Essential experimental matrix

### Matched methods

- LeWM/SIGReg.
- Ordinary IDM with identical head, weight, data, and planning protocol.
- Delta-style latent-difference decoder.
- Prediction weight 0.3.
- Negative-conflict-only routing.
- Full aligned routing.
- Parallel-only routing.
- Norm-matched scalar prediction gradient.
- Retention-matched shuffled control axis.
- Physical-state alignment as a privileged reference when feasible.

### Nuisance and task factors

- Clean PushT.
- Per-episode predictable watermark.
- Pixel-matched per-frame random watermark control.
- Unseen watermark colors, positions, sizes, and intensities.
- RandGoal or another static-but-task-relevant factor, because a method that
  erases every slow feature has not solved the real problem.
- At least one second environment from stable-worldmodel to show that the
  mechanism is not a PushT-specific accident.

### Representation evaluations

- Same-content versus same-tag similarity throughout training.
- Frozen linear probes for physical state and tag identity, reported together.
- Action and multi-step action-sequence decoding.
- Action-effect sensitivity: how much predicted future latent changes when the
  action is changed at fixed observation.
- Physical-distance versus latent-distance rank correlation, used only as a
  privileged evaluation.
- Transition covariance spectrum and effective transition dimension.
- Encoder sensitivity or finite-difference response to independently changing
  content and watermark.
- Gradient cosine, negative-conflict fraction, orthogonal gate, retained norm,
  and control-gradient norm over training.

### Causal-use evaluation

At a fixed observation, replace or scramble the predicted future latent before
the planner/action head consumes it. Measure whether the selected action
changes in a direction consistent with the altered predicted outcome. Compare
against scrambling an equal-norm nuisance direction. If planning is invariant
to future-latent corruption, the learned predictor may be functioning mainly
as a representation regularizer rather than a genuinely decision-causal world
model.

### Downstream reporting

- Use at least 200-500 evaluation episodes for selected checkpoints.
- Report final, best predeclared checkpoint, and a late-window mean; do not use
  the single best 50-episode spike as the main number.
- Use at least three training seeds for the main comparison.
- Keep planner, candidate budget, evaluation seeds, and checkpoint selection
  identical across methods.
- Report prediction quality and planning success jointly, since either can
  improve while representation utility degrades.

## Suggested paper shape

1. Reproduce Enigma and show that ordinary training and IDM can retain
   action-decodable information while the nuisance still dominates distances.
2. Diagnose optimization: negative conflicts are rare, while most prediction
   gradient is nearly control-orthogonal and continues to reshape the encoder.
3. Introduce control-guided prediction-gradient routing and its limited
   propositions.
4. Use norm-matched and shuffled controls to identify directionality rather
   than generic gradient attenuation as the mechanism.
5. Show representation-geometry recovery, physical/action-effect preservation,
   nuisance OOD robustness, and finally planning improvement.
6. Analyze failure boundaries: parallel-only overfilters; weak or converged
   control gradients are incomplete relevance signals; aligned routing is not
   a general certificate of task sufficiency.

## Working title options

- When Prediction Becomes a Distraction: Control-Guided Gradient Routing for
  JEPA World Models
- Predict What Matters: Preventing Nuisance Takeover in Latent World Models
- Who Shapes the Encoder? Separating Prediction from Representation Allocation
  in JEPA World Models

