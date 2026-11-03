"""Finite-budget first-hit supervision for RECORDED continuations.

This estimates behavior-continuation success, not optimal reachability.
Input hit masks must use a fixed task success predicate, never latent proximity.
Episodes must be split before producing windows. Missing follow-up is censored.
"""
import numpy as np


def first_hit_targets(hit, observed, budgets):
    """hit/observed: N,T booleans including t=0; budgets in raw steps.

    A witnessed hit labels every later budget positive, even if follow-up ends.
    A negative requires a complete recorded prefix through that budget.
    Never interpret a negative as impossible under other actions.
    Returns labels and a mask of known labels, both N,len(budgets).
    """
    hit, observed = np.asarray(hit), np.asarray(observed)
    budgets = np.asarray(budgets)
    if hit.dtype != bool or observed.dtype != bool or hit.ndim != 2:
        raise ValueError('Expected boolean N,T hit and observed masks')
    if hit.shape != observed.shape or hit.shape[1] == 0:
        raise ValueError('Incompatible or empty time axis')
    if np.any(hit & ~observed):
        raise ValueError('Cannot witness a hit at an unobserved step')
    if np.any(observed[:, 1:] & ~observed[:, :-1]):
        raise ValueError('Observation masks must be contiguous prefixes')
    if budgets.ndim != 1 or budgets.dtype.kind not in 'iu' or np.any(budgets < 0):
        raise ValueError('Budgets must be nonnegative integers')
    labels = np.zeros((len(hit), len(budgets)), dtype=bool)
    known = np.zeros_like(labels)
    for j, h in enumerate(budgets):
        end = min(int(h) + 1, hit.shape[1])
        labels[:, j] = hit[:, :end].any(axis=1)
        complete = observed[:, :end].all(axis=1) & (h < hit.shape[1])
        known[:, j] = labels[:, j] | complete
    return labels, known


def hazard_to_cdf(logits):
    """First-hit CDF from conditional hazards; monotone by construction.

    Column 0 includes immediate success. Unknown targets require masked loss.
    NumPy reference implementation used to verify the trainable equivalent.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError('Expected finite N,H logits')
    log_survival = -np.logaddexp(0., logits)
    return -np.expm1(np.cumsum(log_survival, axis=1))
