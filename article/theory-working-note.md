# Theory working note: what the present method does and does not establish

Status: derivations for discussion, 2026-09-20. These are conditional statements,
not a proof of global identifiability or the optimality of cosine admission.
This note supersedes stronger informal claims in the preliminary webpage.

## 1. A predictive shortcut is an admissible solution

Let observations be x_t = h(s_t,n), with n constant over a trajectory. Assume
there exists an admissible encoder f(h(s,n)) = psi(n) satisfying the actual
representation regularizer's constraints. If the predictor can implement the
identity, p(z,a) = z, then its squared latent prediction loss is zero.

Proof: both target and prediction equal psi(n). This establishes existence of
a state-discarding solution, not convergence to it. If two observations have
the same n but require different decisions for the same externally supplied
goal, an encoder that identifies them is insufficient for those decisions.

The regularizer assumption must be checked. A finite or low-dimensional tag
does not automatically support an exact full-dimensional Gaussian under a
continuous encoder. Nonzero variance, full-rank covariance, and matching a
Gaussian distribution are distinct requirements. A stochastic physical mode
also need not have transition eigenvalue below one; spectral and optimization
claims require a specified operator, function class, and objective.

## 2. What an action loss adds

For squared action loss with an unrestricted optimal decoder, define

R(V) = E ||A - E[A | V]||^2.

For nuisance N and physical transition U, the conditional-expectation identity
is

R(N) - R(N,U) = E ||E[A | N,U] - E[A | N]||^2 >= 0.

Proof: expand the square around E[A | N,U]; the residual is orthogonal in L2
to measurable functions of (N,U). Strict positivity requires a conditional
MEAN difference, not merely any dependence between action and transition.

If B(f) contains prediction and representation regularization, a candidate
physical encoder f_u beats a nuisance encoder f_n in B + lambda R only if

lambda [R(f_n)-R(f_u)] > B(f_u)-B(f_n).

This pairwise comparison does not characterize every global optimum. Action
decodability does not imply goal-conditioned decision sufficiency. The actual
control objective includes inverse, cycle, and reachability terms; this
identity directly describes only the ideal squared inverse objective.

## 3. Exact geometry of the implemented rule

All vectors below are per-sample embedding gradients, flattened over time and
features. gp already includes the prediction weight; gc includes all control
weights; gr is the weighted SIGReg gradient. Assume gc != 0 and ignore epsilon
and finite-precision effects. Set e = gc/||gc||, a = gp^T e, u = gp-ae,
c = a/||gp||, a+ = max(a,0), c+ = max(c,0).

The prediction contribution is R = a+ e + c+ u. The total embedding gradient
is G = gc + R + gr, plus any separately enabled auxiliary objective.

The component ablations preserve gc and gr:

- parallel-only: G = gc + a+ e + gr;
- gated orthogonal-only: G = gc + c+ u + gr;
- raw orthogonal-only: G = gc + u + gr;
- joint without routing: G = gc + gp + gr.

At the SAME parameters and batch, Full differs from gated orthogonal-only by
(a+/||gc||) gc. It adds an adaptive contribution in the existing guide direction.
This equality does not make their independently trained trajectories identical.

For c > 0:

||c u|| / ||a e|| = sqrt(1-c^2),
||R|| / ||gp|| = c sqrt(2-c^2).

Thus near positive orthogonality the two admitted components have similar
norm, even though the original orthogonal component dominates. Negative c
causes the original rule to admit no prediction gradient. With exactly zero
guide the implementation also returns zero in the original aligned mode.

Prediction-only local properties:

gc^T R >= 0; gp^T R >= 0; ||R|| <= ||gp||.

These are properties of a hypothetical embedding step -eta R. They do not
guarantee control descent under the entire training update: SIGReg, encoder
Jacobian coupling, and Adam preconditioning remain relevant. The surrogate
adds no correction to predictor-only parameters; their original loss
gradients, including applicable control paths, remain present.

For beta in (0,1), a+ e + beta u solves

min_v 0.5||v-gp||^2 + (lambda/2)||Qv||^2,
subject to gc^T v >= 0,

where Q=I-ee^T and lambda=1/beta-1. Decomposition along e and Q proves the
result. This is a variational characterization, not a justification of the
penalty's semantic correctness. beta=c for positive c makes lambda adaptive.

## 4. The unresolved mechanism condition

Under a stylized additive decomposition gp=d+n and the linear operator
A=P+beta Q, define r_h=||Ph||^2/||h||^2. For nonzero d,n,

(||Ad||^2/||An||^2)/(||d||^2/||n||^2)
= [beta^2+(1-beta^2)r_d]/[beta^2+(1-beta^2)r_n].

Therefore r_d>r_n improves this component-energy ratio for 0<beta<1. This
applies to the positive branch of routing with beta held at its evaluated
value. It does not establish such a decomposition in a nonlinear encoder,
r_d>r_n in our data, or an improvement in final representation information.
Two components entirely in Q are scaled equally; the gate cannot distinguish
their semantics within that subspace.

True cosine above a sample-shuffled control null demonstrates an association
under that null, not physical specificity or a universally valid significance
test. The algorithm itself does not subtract the null. A batch- or
trajectory-aware uncertainty analysis is required for population claims.

## 5. Experiments that decide the interpretation

The new matched campaign trains Full, joint-no-route, and norm-matched scalar
from the same source/config and seed list. Scalar matches the reference norm
at its OWN current parameters; it cannot maintain the same norm history as an
independently diverging Full trajectory. This tests a local rule, not a fixed
historical learning-rate schedule.

The checkpoint intervention changes native-pixel tags while holding content
and candidate actions fixed. Shared-tag, context-only, and goal-only changes
separate common-mode changes from mismatches. It reports representation
distances, candidate-cost changes, order reversals, and selected action changes.
This is not a closed-loop planning success test. Sampled training-dataset clips
are not claimed to be held-out episodes; clip-bootstrap intervals do not
measure training-seed variation.

Interpretation rules:

- Similar Full/scalar performance weakens the necessity of directional routing.
- Full over joint alone does not distinguish direction from gradient magnitude.
- Lower linear tag R2 alone does not imply invariance or reduced behavioral use.
- Reduced counterfactual cost/action sensitivity supports behavioral robustness,
  conditional on the tested intervention and candidate set.
- Existing 86/82/92 percent SR does not establish Full superiority or necessity
  of both prediction components. No synergy theorem follows from these numbers.

Next publication-level work is independent training seeds and a pre-specified
common final-checkpoint evaluation. Reuse existing JEPA as an exploratory
reference only until source, data, optimization, and evaluation match are audited.
