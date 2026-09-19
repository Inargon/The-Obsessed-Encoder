# Who Shapes the Encoder?

## When a predictive world model learns what is easy instead of what is useful

Predicting the future sounds like a good way to learn a world model. Give a
model the current image and an action, ask it to predict the next latent state,
and the representation should have to capture the underlying dynamics.

But predictive learning contains a loophole: the model is rewarded for
predicting **whatever is easiest to predict**, not necessarily whatever a
planner needs.

We can expose that loophole with a five-pixel-wide square.

We add a small random-colour tag to every frame of a PushT episode. The colour
is constant within the episode and unrelated to the motion of the pusher or
the block. It is useless for choosing actions. It is also extremely easy to
predict.

After ten epochs, a standard joint-embedding predictive model reaches only
**4% control success**. A frozen linear probe explains why: tag colour is
decoded from its backbone with \(R^2=0.98\), while the mean \(R^2\) of pusher
position, block position, and block angle is only \(0.31\). At the projection
used by predictive learning, physical-state decodability falls to \(0.10\),
while the tag remains at \(0.95\).

The representation has not collapsed to a constant. It has collapsed onto
the wrong distinction.

> **Prediction prevented global collapse, but it did not prevent semantic
> capture.**

This is the obsessed-encoder problem: a small predictable nuisance acquires
more representational authority than the state variables needed for control.

<!-- FIGURE 1: Two PushT observations with identical physical content and
different episode tags, followed by JEPA and Full latent-neighbour diagrams. -->

## One loss, two jobs

Let an encoder produce \(z_t=f_\theta(o_t)\), and let a predictor estimate a
future latent under action \(a_t\):

\[
\widehat z_{t+1}=p_\phi(z_t,a_t),\qquad
\mathcal L_p=\|\widehat z_{t+1}-z_{t+1}\|^2.
\]

The prediction loss is doing two jobs at once.

First, it trains the predictor \(p_\phi\). Second, through the encoder, it
decides which visual factors become easy to represent. The predictor may
legitimately need to model every predictable factor. The planner-facing
encoder does not need to give every factor equal authority.

That distinction suggests an asymmetric intervention:

> Let the predictor receive the full prediction gradient, but treat the
> prediction gradient entering the encoder as a proposal.

We obtain a relevance signal from a control objective trained only from
images, actions, and trajectories. It combines short-horizon inverse action
prediction, action-cycle consistency, and reachability. It does not use
simulator state.

At the shared representation, define

\[
g_p=\nabla_z\mathcal L_p,\qquad
g_c=\nabla_z\mathcal L_c.
\]

Decompose the prediction gradient relative to the current control gradient:

\[
g_p=g_\parallel+g_\perp,
\qquad
g_\parallel=
\frac{g_p^\top g_c}{\|g_c\|^2+\epsilon}g_c.
\]

Our encoder-side prediction update is

\[
R(g_p,g_c)
=g_\parallel^+
+\rho g_\perp,
\qquad
\rho=[\cos(g_p,g_c)]_+,
\]

where \(g_\parallel^+\) keeps only a positively aligned parallel component.
The encoder therefore receives

\[
g_{\mathrm{enc}}=g_s+g_c+R(g_p,g_c),
\]

including the statistical anti-collapse gradient \(g_s\). Predictor-only
parameters still receive the complete prediction gradient.

This is not a claim that the control gradient identifies every useful piece of
physics. It is a rule for allocating optimization authority: prediction is
free to learn, but its ability to reorganize the shared encoder is budgeted by
control evidence.

## But aren't random vectors orthogonal in high dimensions?

Yes. That objection is central, not incidental.

For independent isotropic unit vectors in \(d\) dimensions,

\[
\mathbb E[\cos(u,v)]=0,
\qquad
\mathrm{Var}[\cos(u,v)]\approx\frac{1}{d}.
\]

Near-orthogonality is therefore the high-dimensional null expectation. A
small cosine does **not** prove that a prediction direction is useless. It may
contain long-horizon dynamics, a variable already learned by the control
head, or information absent from a short-horizon guide.

Our measurements make the concern concrete. In a fixed-batch audit at 10k
updates, roughly **99% of the prediction-gradient norm is orthogonal** to the
instantaneous control gradient. A method that calls this entire subspace
irrelevant would be making an indefensible semantic claim.

The meaningful comparison is not “large cosine versus zero.” It is the true
pairing versus a null pairing. The mean prediction/control cosine is
\(0.1138\); after shuffling the control axes between samples, it is
\(-0.0010\). The excess alignment of \(0.1148\) says that the control guide
contains sample-specific directional information even though most norm lies
outside its one-dimensional axis.

We therefore interpret cosine as a **confidence and allocation signal**, not
as a classifier that labels every orthogonal direction useless.

<!-- FIGURE 2: High-dimensional null distribution centred at zero, with true
and shuffled cosine markers; alongside the g_parallel/g_perp decomposition. -->

## What happens when we remove each component?

We trained four models for ten epochs under the same tagged PushT condition
and evaluated the same 50 episodes.

| Encoder update | Success | Physical \(R^2\) | Tag \(R^2\) |
|---|---:|---:|---:|
| Standard JEPA | 4% | 0.313 | 0.980 |
| Parallel prediction only | 82% | 0.926 | 0.862 |
| Control + gated orthogonal prediction | 92% | 0.944 | 0.833 |
| Full routing | 86% | 0.941 | 0.604 |

The physical score is the mean backbone linear-probe \(R^2\) over pusher
position, block position, and block orientation. Reported probe variation
comes from three held-out clip splits; all models currently use training seed
zero.

The control-guided variants all recover task state and control performance.
But only the full route substantially reduces tag decodability.

The 92% success of gated orthogonal prediction deserves care. This variant is
not a “pure orthogonal gradient” method. It keeps the full control update and
removes only the prediction component parallel to that update:

\[
g_{\mathrm{enc}}^{\mathrm{GatedOrth}}
=g_s+g_c+\rho g_\perp.
\]

It can be understood as avoiding a duplicated update along the current
control axis. It preserves excellent physical-state decodability and obtains
the highest point estimate for control, but it still makes tag colour highly
linearly accessible.

Full routing obtains 86%, six points below the 92% estimate, but the paired
difference is only three episodes and is not statistically significant
(exact \(p=0.508\)). In contrast, Full beats JEPA on 41 paired episodes while
losing none (exact \(p\approx9.1\times10^{-13}\)).

The distinction matters. Gated orthogonal prediction largely solves control
**despite** the shortcut. Full routing makes the representation itself more
selective: it retains comparable physical state while reducing tag \(R^2\)
from \(0.83\) to \(0.60\).

<!-- FIGURE 3: Three grouped bars per model: success, physical R2, tag R2.
Use a separate axis/panel for success rather than implying identical units. -->

## What the experiment establishes

The result supports a four-link mechanism:

\[
\text{predictable nuisance}
\rightarrow
\text{representation capture}
\rightarrow
\text{loss of physical geometry}
\rightarrow
\text{planning failure}.
\]

Control-guided encoder optimization reverses the middle of this chain. The
standard JEPA allocates nearly perfect linear access to the tag and weak access
to physical state. Full routing restores physical state, reduces tag access,
and changes control success from 4% to 86%.

The component ablations sharpen the conclusion. Control supervision is enough
to recover much of the physical state. Routing determines whether the model
also continues to devote substantial capacity to the nuisance. Task success
alone would hide this difference.

## What it does not establish

Several boundaries are important.

**Linear decodability is not causal use.** A lower tag \(R^2\) shows that the
nuisance is less linearly exposed; it does not prove that all tag information
has disappeared or that CEM never uses it.

**A control gradient is not a semantic oracle.** A short-horizon action loss
can miss static goals and long-horizon consequences. Orthogonal information
must not be declared irrelevant merely because the current guide does not
cover it.

**The model results currently use one training seed.** The three probe seeds
change only the held-out clip split. Episode-level paired tests quantify
evaluation uncertainty for fixed checkpoints, not training variability.

**Gradient surgery is not new.** Projection, conflict removal, and cosine
gating have precedents in multi-task and auxiliary learning. The contribution
here is the failure mode, the asymmetric predictor/encoder interface, and the
matched evidence connecting optimization geometry to a planner-facing
representation.

The next decisive controls are joint training without routing,
control-only encoder training, and additional training seeds. They will
separate the contribution of control supervision from the contribution of
gradient admission itself.

## The broader lesson

World models are often judged by whether they can predict. Planning requires a
different question: **what distinctions does prediction cause the encoder to
care about?**

A predictive loss is not merely a source of information. It is a mechanism
for allocating representational authority. When a nuisance is easier to
predict than the physical state, preventing numerical collapse is not enough.
The encoder can remain diverse, stable, and completely obsessed with the wrong
thing.

The practical principle is modest:

> Preserve the model's ability to learn the predictable world, but measure and
> control which predictive updates are allowed to shape the representation
> used for decisions.

That principle does not make orthogonal gradients useless. It makes their
optimization budget an empirical question.

---

### Experimental notes

- Dataset: `pusht_expert_train.h5` with an episode-constant 5x5 RGB tag.
- Checkpoint: epoch 10, training seed 0.
- Planning: the same 50 tagged episodes and evaluation seed for every model.
- Probes: frozen encoders, ridge regression, 1,024 sampled clips, four frames
  per clip, 80/20 clip-level split, split seeds 17/73/137.
- Exact values and provenance are stored in `results_snapshot.json`.

### Context

This investigation builds on the failure described in
[The Obsessed Encoder](https://www.enigma.inc/posts/obsessed-encoder) and the
latent world-model implementation in
[LeWorldModel](https://github.com/lucas-maes/le-wm). The negative-conflict
comparator is conceptually related to
[PCGrad](https://arxiv.org/abs/2001.06782), but the full rule additionally
budgets the near-orthogonal prediction component.

