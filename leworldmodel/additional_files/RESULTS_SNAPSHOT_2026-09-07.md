# Enigma / LeWM Results Snapshot

Date: 2026-09-07

## Protocol boundary

The old-server and gridfire numbers are not one leaderboard.

- Most old-server results are seed 0 and use training-time evaluation. The
  project record estimates that protocol to be roughly 0.1 higher than the
  offline protocol.
- Gridfire planning evaluations use 50 episodes per checkpoint, normally every
  2,000 optimizer steps. A difference of 0.02 is one successful episode, so a
  single peak is noisy.
- Compare methods numerically only within the same server and matched protocol.

## 1. Old-server results: diagnostic reference

| Method | Privileged signal | Late / last-10 SR | Peak SR | Main use |
|---|---:|---:|---:|---|
| angle + rho=0.77 | angle, rho | 0.91 | 0.96 | Oracle headroom; both axes repaired |
| rho clean | rho | 0.91 | 0.96 | Clean control under rho anchoring |
| clean baseline | none | 0.89 | 0.94 | Approximate attainable clean level |
| focus MSID | none beyond action | 0.81 | 0.88 | Best old nonprivileged action baseline |
| temporal progress + rho | rho | 0.77 | 0.84 | Nonprivileged reinjection plus anchor |
| rho=0.77 only | rho | 0.67 | 0.74 | Removes dominance but angle remains weak |
| LDAD | no state label | 0.67 | 0.72 | Partial recovery |
| quasimetric + rho | rho | 0.57 | 0.68 | Relation objective plus anchor |
| quasimetric | none | 0.50 | 0.60 | Relation objective alone |
| reach-probability clean | none | 0.14 | 0.18 | Auxiliary relation loss can harm a healthy representation |
| gamma + rho | rho | 0.10 | 0.20 | Weak combination |
| angle only | angle | 0.08 | 0.12 | Decodability/reinjection alone is insufficient |
| temporal progress only | none | 0.06 | 0.16 | Progress alone is insufficient |
| watermarked LeWM | none | 0.05 | 0.12 | Enigma failure condition |
| episode reach-probability | none | 0.03 | 0.06 | Failed relation auxiliary |

### Old-server representation audit

- Across eleven arms, block-angle linear-probe R-squared had reported Spearman
  correlation `+0.964` with planning success.
- Pusher-state R-squared stayed around `0.95` for nearly every arm and did not
  explain the performance spread.
- Watermark identity could remain decodable in a strong arm. The relevant
  failure is therefore not nuisance presence alone, but nuisance dominance and
  the failure of block orientation to become load-bearing geometry.
- Linear metric reweighting, a nonlinear INTACT probe, and quasimetric
  reweighting did not repair a collapsed planner. Decodable information was not
  automatically usable by latent-distance planning.

### Clean four-cell diagnosis

| Condition | Late SR |
|---|---:|
| watermarked | 0.05 |
| angle only | 0.08 |
| rho only | 0.67 |
| angle + rho | 0.91 |

This separates two requirements: resist predictable-nuisance dominance and
make the missing task variable organize the representation. Neither alone was
sufficient.

## 2. Gridfire full runs

| Method | Steps | Best SR | Final SR | Late mean | Same-content | Same-tag | Geometry margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditional allocation | 139,329 | 0.64 | 0.54 | 0.56 | 0.327 | 0.639 | -0.312 |
| masked reachability | 139,329 | 0.74 | 0.66 | 0.66 | 0.470 | 0.517 | -0.047 |
| masked sequence, pred=0.3 | 139,329 | 0.72 at 54k | 0.64 | 0.624 last 10 | not final-copied here | not final-copied here | — |
| control-aligned routing | 139,329 | 0.88 at 96k and 118k | 0.80 | 0.774 last 10 | 0.807 | 0.176 | +0.631 |

For aligned at the final training step:

- prediction loss: approximately `0.0281`;
- control loss: approximately `0.1198`;
- prediction/control gradient cosine: approximately `0.0715`;
- orthogonal gate: approximately `0.0752`;
- retained prediction-gradient norm fraction: approximately `0.1057`;
- fraction with reversed/negative relation: approximately `0.125`.

The robust aligned result is the sustained content/tag reversal. The `0.88`
value is a best 50-episode checkpoint, not yet a stable multi-seed estimate.

## 3. Matched 10k mechanism ablations

All arms below use seed 0 and the same 50-episode evaluation cadence.

| Arm | SR at 10k | Best SR through 10k | Pred loss | Control loss | Retained norm | Content | Tag | Margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| norm-matched scalar | **0.42** | **0.42** | 0.110 | 0.258 | 0.153 | 0.828 | 0.146 | +0.682 |
| aligned routing | 0.36 | 0.36 | **0.090** | **0.220** | 0.161 | 0.865 | 0.110 | +0.755 |
| retention-matched shuffled | 0.28 | 0.30 | 0.138 | 0.265 | 0.149 | 0.844 | 0.129 | +0.716 |
| parallel-only | 0.24 | 0.34 | 0.332 | 0.272 | 0.085 | **0.963** | **0.014** | **+0.949** |
| unmatched shuffled | 0.18 | 0.18 | 0.262 | 0.275 | 0.031 | 0.928 | 0.046 | +0.882 |

Matched-control validity:

- norm-matched scalar relative norm error: `0.00197`;
- retention-matched shuffled relative norm error: `0.00173`;
- norm-matched scalar direction cosine to aligned: `0.754`;
- retention-matched shuffled direction cosine to aligned: `0.539`.

### Full 10k curves

- aligned: `0.02, 0.04, 0.22, 0.30, 0.36` at 2k--10k;
- norm-matched scalar: `0.08, 0.14, 0.20, 0.40, 0.42`;
- retention-matched shuffled: `0.02, 0.14, 0.20, 0.30, 0.28`;
- parallel-only: `0.02, 0.06, 0.02, 0.34, 0.24`;
- unmatched shuffled: `0.04, 0.10, 0.16, 0.12, 0.18`.

The difference between `0.42` and `0.36` is three successes out of 50 at the
last checkpoint, so it is not conclusive alone. The scalar arm is nevertheless
strong across several early checkpoints and warrants a full run.

## 4. Other explored methods and failure boundaries

- Local-rank, conditional-covariance, and hybrid allocation regularizers passed
  smoke/pilot tests but did not establish a planning advantage. In the 500-step
  pilot, conditional had prediction loss `0.179`, while hybrid and local-rank
  were both around `0.265`.
- Factorized reachability at 500 steps showed dynamic-static leak increasing
  from `0.645` to `0.828`, dynamic variance collapsing to `0`, and only
  `0.323` reachability accuracy.
- Action-token routing collapsed planning: at 80k its latest SR was `0.04` and
  its best observed SR was `0.12` at 4k. The full job was cancelled.
- Effect-geometry training was unstable/weak early: SR through 12k was
  `0.02, 0.06, 0.02, 0.20, 0.08, 0.08` despite its Gram loss decreasing.
- Action-binding 10k pilots were weak: bank-IDM was `0.10` at 4k,
  counterfactual binding `0.02`, and the oracle arm became NaN.
- Ordinary negative-conflict routing removed very little prediction gradient.
  Around 58k it reached `0.68` versus `0.62` for pred03, but the audit showed
  mean gradient cosine remained positive and negative conflict was rare.

These runs should be grouped as falsified hypotheses rather than displayed as
an undifferentiated method leaderboard.

## 5. Current state

- Full norm-matched scalar seed 0: Slurm job `86250`, submitted and pending due
  to the shared account's running-job QOS limit.
- Frozen representation probe: implementation complete and `4 passed`; Slurm
  submission was rejected because the shared account already had ten jobs.
  No probe result exists yet.
- The planned probe compares aligned near 96k/final, pred03 near 54k/final,
  and masked-reachability final on backbone and projection R-squared for
  pusher XY, block XY, block-angle sin/cos, and tag RGB.
- Future-latent intervention has not been implemented. Current action-shuffle
  tests only establish that an auxiliary head uses action, not that CEM uses
  world prediction causally.

## 6. Important conclusions, ordered by evidential strength

### Strongly supported

1. Lower prediction loss is not a reliable proxy for planning utility under a
   predictable nuisance.
2. Avoiding statistical collapse is not sufficient; task variables must become
   load-bearing in the geometry used by planning.
3. Aligned training strongly and persistently reverses watermark-dominated pair
   geometry while retaining useful planning performance.
4. Maximum nuisance invariance is not the objective: parallel-only produces the
   cleanest content/tag separation but worse prediction and planning.
5. Ordinary negative gradient conflict is not the main observed failure mode;
   most prediction/control updates are weakly positive or near-orthogonal.

### Supported but still provisional

1. Control-conditioned prediction budgeting may be more important than rotating
   the prediction gradient. The 10k norm-matched scalar arm currently exceeds
   aligned (`0.42` versus `0.36`) while using a matched update norm.
2. Correct control correspondence still appears useful: matched shuffled reaches
   `0.28`, below aligned, but this needs longer runs and more seeds.
3. The simplest emerging method is therefore not necessarily directional
   projection. It may be an adaptive scalar admission rule: use control/prediction
   alignment to decide how much prediction may reshape the encoder, while the
   predictor still receives its full learning signal.

### Not yet established

1. Stable superiority over IDM: protocols, seed counts, and evaluation counts
   are not yet matched.
2. That aligned or scalar routing specifically restores block angle in current
   gridfire checkpoints: the frozen probe is ready but has not run.
3. That predicted future latent is decision-causal for CEM: no future-latent
   replacement or candidate-ranking intervention has run.
4. A general capacity-allocation theorem: current evidence supports a mechanism
   hypothesis about representation geometry and encoder updates, not a conserved
   bit-capacity law.

## 7. Current one-sentence paper claim

Predictable nuisances expose a mismatch between what is easy to predict and
what should shape a planning representation. Separating predictor learning from
prediction's update to the shared encoder, then allocating that encoder update
according to control alignment, can reverse nuisance-dominated geometry without
discarding world prediction; matched controls currently suggest that adaptive
gradient budgeting may be the simpler load-bearing mechanism.
