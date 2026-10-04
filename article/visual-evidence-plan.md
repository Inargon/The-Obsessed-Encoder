# Visual evidence plan

This note fixes the visual-evidence protocol before looking at individual
examples. It takes two complementary perspectives:

- **The Obsessed Encoder:** test whether a predictable but task-irrelevant
  variable dominates representation geometry or planning decisions.
- **When Does LeJEPA Learn a World Model?:** test whether the learned latent
  preserves physical variables, transitions, and planning-relevant geometry.

## Figure 1 — Method

Show the shared encoder, control gradient, EMA control direction, prediction
gradient, and the unconditional projection of the prediction gradient onto the
orthogonal complement of the EMA direction. Label this as **EMA-guided
orthogonal gradient repair**, not conflict-only surgery.

## Figure 2 — Representation allocation

For JEPA, Full, Cycle, and EMA repair, report matched frozen-probe performance
for physical state and nuisance variables. Pair the numbers with a latent
geometry panel showing same-physics/different-nuisance and
different-physics/same-nuisance distances. PCA/UMAP may illustrate the result,
but is not treated as evidence without the paired quantitative tests.

## Figure 3 — Counterfactual decision stability

Hold the physical frames, goal, and candidate action bank fixed. Change only
the nuisance tag. Report normalized cost RMSE, pair-order reversal, selected
candidate changes, and selected-action distance. Include a static aggregate
plot over all sampled clips and GIFs that alternate the two nuisance values for
predeclared representative examples.

Example selection must be disclosed. The primary aggregate plot uses every
sample; illustrative GIFs prioritize JEPA-only decision flips and then the
largest JEPA-minus-repair cost-sensitivity gap.

## Figure 4 — Correlation reversal

Train with a nuisance correlated with the target or action, then evaluate on
IID, balanced, and reversed correlations. Candidate tasks include background
colour versus target quadrant in Reacher, wall texture versus goal room in
TwoRoom, table colour versus target direction in Cube, and background colour
versus push direction in PushT. This is stronger causal evidence than a linear
probe because it tests whether the learned shortcut controls behavior.

## Figure 5 — Planning geometry

Align latent coordinates to ground-truth physical state using a linear map or
CCA, then visualize matched physical and latent trajectories. Report linear
identifiability, local transition consistency, neighborhood preservation, and
whether shortest-path or candidate ordering is preserved. Show the same
transition rendered under several nuisance values and plot the corresponding
latent displacement vectors.

## Figure 6 — Search versus amortized control

Show CEM and frozen-encoder GC-IDM rollouts side by side with success rate and
planning latency. Include representative failures, especially contact-rich
PushT cases where direct inverse-dynamics regression may average over multiple
valid actions while online search remains effective.

## Reproducibility rules

- Fix checkpoint, dataset indices, initial state, goal, candidate bank, and
  evaluation seed across methods.
- State exactly which pixels changed in every counterfactual.
- Save the raw JSON behind each plot and include checkpoint hashes.
- Export both GIF/MP4 and a static frame strip suitable for the paper.
- Never present hand-picked examples without the matched aggregate result.
- Separate representation bias, decision sensitivity, and closed-loop success;
  none is a substitute for the other two.

## First runnable artifact

`bloop_counterfactual_visual_campaign.py` produces the first no-retraining
package on tagged PushT: an aggregate decision-sensitivity figure, six paired
static storyboards, six alternating GIFs, and the raw per-clip candidate costs.
