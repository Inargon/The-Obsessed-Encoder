# Action-effect ratio pilot

Status: implemented as an opt-in training objective; **no environment result has
been measured**. Do not treat the unit tests as evidence of nuisance invariance.

## Fast synthetic check

`python leworldmodel/additional_files/smoke_action_effect_ratio.py` runs in
about 13 seconds on a CPU. It generates a 2D position that changes with the
action and a separate 3D tag that stays fixed across each transition. Over
three seeds, adding score consistency at weight 0.1 reduced the held-out mean
absolute tag-only score change from `0.1869 +/- 0.0774` to
`0.1513 +/- 0.0361`. The true-vs-shuffled action score margin remained
positive (`163.05 +/- 5.94` versus `150.92 +/- 17.70`). The improvement is
modest relative to seed variation, and the margin scale depends on the
Gaussian likelihood model. This check supports only that the code can learn
an action-sensitive score without eliminating the tag intervention effect;
it does **not** establish a benefit on Tagged PushT.

The existing `masked_reachability` objective already has action-conditioned
query and endpoint-key heads. Merely renaming those heads forward/backward is
not a new method. This pilot instead estimates

`r(s,a,s') = log q(a | s,s') - log q(a | s)`.

If both densities are calibrated to the same behavior distribution, Bayes'
rule makes this `log p(s' | s,a) - log p(s' | s)`. In a factorized system where
the nuisance transition is independent of action, the nuisance likelihood
factor cancels algebraically. This is a density *ratio*, not goal-reaching
probability, and it must not be substituted for the CEM distance without a
separate planning experiment. With finite data or a nuisance-dependent
behavior policy, the learned networks need not be invariant. The optional
counterfactual loss directly tests and penalizes score changes when only the
5x5 tag is altered.

## Opt-in configuration

Add this block to a Tagged PushT training config under `loss`:

```yaml
action_effect_ratio:
  enabled: true
  horizon: 1
  hidden_dim: 256
  weight: 0.1
  consistency_weight: 0.1
```

The consistency term requires `pixel_tag.counterfactual_key` to be
`pixels_counterfactual`, as in the existing counterfactual planning campaign.
Set `consistency_weight: 0` for the ratio-only ablation. Keep the same data,
seed, backbone, prediction weight, and planning protocol across arms.

## Minimum comparisons

1. Existing inverse + reachability + Bloop baseline.
2. Baseline plus action-effect ratio, consistency weight zero.
3. Baseline plus action-effect ratio and matched-tag score consistency.

Before closed-loop evaluation, compare held-out posterior/prior action NLL,
score changes under tag-only intervention, and action-shuffle sensitivity.
Then evaluate fixed-candidate ranking changes and Tagged PushT success over
the same episodes. A useful ratio must improve intervention stability without
destroying action sensitivity. The first pilot does not change the CEM cost;
using the score for planning is a separate subsequent ablation.

Implementation: `action_effect_ratio.py` and the opt-in hook in `train.py`.
