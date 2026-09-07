# Enigma / LeWM: Senior Meeting Report Notes

Date: 2026-09-07

## Protocol warning

The old-server results below use the training-time evaluation protocol, which
was estimated to run roughly 0.1 higher than the offline protocol. Most old
results are also seed 0. The current gridfire results use a separate run and
must not be numerically ranked against the old-server table. Use within-server,
matched-protocol comparisons only.

## One-sentence story

Enigma exposes a mismatch between predictability and representational utility.
Our old experiments show that removing the nuisance and making task variables
decodable are each insufficient; the new gradient diagnosis shows that the
dominant prediction updates are usually control-insensitive rather than
negatively conflicting. Control-guided prediction-gradient routing therefore
separates predictor learning from how prediction is allowed to reshape the
shared encoder.

## The old results worth presenting

### 1. A clean four-cell diagnosis: anti-collapse and task reinjection differ

Use the angle/rho quartet, not the full arm leaderboard:

- watermarked LeWM: last-10 success 0.05;
- privileged angle supervision only: 0.08;
- rho anchor only: 0.67;
- privileged angle + rho: 0.91, near the clean baseline at 0.89.

Interpretation: preserving a broad/noncollapsed representation is not enough
to ensure that the missing task variable carries useful geometry; injecting a
task variable is not enough if the predictable nuisance still controls the
representation. The oracle result demonstrates headroom, but rho and angle are
diagnostic instruments rather than the desired final method.

The nonprivileged analogue is directionally consistent: temporal progress only
scores 0.06, while progress + rho reaches 0.77.

### 2. The missing quantity is specific, not generic capacity

Across eleven old-server arms, block-angle linear-probe R-squared has a reported
Spearman correlation of +0.964 with planning success. Pusher state is decoded
at roughly 0.95 almost everywhere and does not explain the performance spread.
Watermark identity can remain decodable in a strong arm, so the issue is not
simply whether nuisance information exists. The stronger claim is that block
orientation fails to become load-bearing in the planning geometry.

Present the correlation as a seed-0 descriptive audit, not a population-level
law. Follow it with multi-seed confirmation if it becomes a headline result.

### 3. Decodable does not mean usable for planning

Three planning-side repair attempts failed:

- linear metric reweighting did not recover a collapsed encoder;
- a nonlinear INTACT probe remained near the collapsed planner result;
- reweighting the quasimetric representation reduced success despite a
  reasonably decodable angle variable.

Interpretation: a variable can be extractable by a probe yet fail to organize
the distances and rollout geometry consumed by the planner. This motivates
measuring representation geometry and causal use, not only linear probes.

Call this an empirical supply-side finding, not a theorem.

### 4. Conditional prediction cost explains why angle loses

The useful formulation is not that slowly changing variables are intrinsically
expensive. It is the residual conditional uncertainty or cost after observing
the current representation and action. Episode watermark is almost free to
predict; block orientation changes sparsely through contact and can be harder
to model. Under a finite optimization and representation budget, the easy
factor receives disproportionate reinforcement.

This connects the angle audit to Enigma without claiming a universal law that
capacity is proportional to predictability.

### 5. The scalar gradient split is the direct precursor to aligned routing

The old lambda-E/lambda-F implementation separated the prediction loss into:

- full predictor/action-encoder gradient (lambda-F = 1);
- reduced encoder gradient (lambda-E below 1).

Its smoke test verified an encoder gradient ratio of 0.300 while leaving the
predictor gradient near full strength. This established that predictor fitting
and encoder shaping can be manipulated separately.

Aligned routing generalizes that scalar idea:

```text
scalar split:  g_pred_to_encoder = lambda_E * g_pred
aligned:       g_pred_to_encoder = R(g_pred | g_control)
```

The narrative is therefore: uniform wage cut first, then direction-aware
admission. The norm-matched scalar experiment is the decisive comparison.

## Current gridfire evidence to present separately

### Full aligned seed 0

- training completed normally at 139,329 steps;
- best 50-episode success: 0.88 at 96k and again at 118k;
- final measured success: 0.80 at 138k;
- final last-10 mean: 0.774;
- final same-content cosine: 0.807;
- final same-tag cosine: 0.176;
- final prediction-gradient retention: 0.106;
- final prediction/control gradient cosine: 0.072.

The robust result is the sustained content/tag reversal. Success exceeds the
old IDM reference at selected checkpoints but cannot yet be claimed superior
because protocol, seed count, and evaluation sample count differ.

### First mechanism ablations at 10k

- aligned at 10k: success 0.36;
- parallel-only: success 0.24, same-content 0.963, same-tag 0.014, prediction
  loss 0.332;
- shuffled guide: success 0.18, retained prediction gradient 0.031.

Parallel-only demonstrates that maximal watermark invariance is not sufficient:
hard deletion also removes useful predictive structure. The first shuffled arm
supports guide correspondence but is confounded by lower retained norm. The
new norm-matched scalar and retention-matched shuffled experiments directly
address that confound.

### Failed methods that belong in an appendix or failure analysis

- prediction weight 0.3: peak 0.72 and final 0.64 on gridfire;
- conditional allocation: peak 0.64;
- masked reachability: peak 0.74;
- action-token routing collapsed planning;
- factorized, local-rank, effect-geometry, and action-binding pilots did not
  justify further compute;
- changing the prediction kernel to a window residual also damaged the clean
  control, showing that preserving predictor learning is necessary.

Do not present these as a long leaderboard. Group them by falsified hypothesis.

## Recommended meeting slide order

1. Enigma failure: lower prediction loss, worse planning, tag-dominated latent.
2. Old four-cell angle/rho diagnosis: anti-collapse and reinjection are distinct.
3. Angle audit: what task information is missing or non-load-bearing?
4. Planning-side repair failures: decodability is not usable geometry.
5. Gradient audit: negative conflict is rare; near-orthogonal updates dominate.
6. Method: preserve predictor gradient, route only prediction-to-encoder.
7. Full seed-0 result: persistent content/tag reversal plus planning curve.
8. Parallel and shuffled falsifiers: soft routing is necessary, but initial
   shuffled comparison has a norm confound.
9. Running decisive experiments: norm-matched scalar, retention-matched
   shuffled guide, exact IDM comparator, seeds 1 and 2.
10. Claim boundary and next milestone.

## Suggested spoken conclusion

The current evidence does not yet prove an abstract capacity-allocation theory
or a stable improvement over IDM. It does show that Enigma's predictable
nuisance failure is not explained by ordinary negative gradient conflict, and
that changing how prediction is allowed to shape the encoder can reverse the
representation geometry without discarding predictor training. The next
matched controls determine whether control-conditioned direction, rather than
adaptive down-weighting alone, is the load-bearing mechanism.

