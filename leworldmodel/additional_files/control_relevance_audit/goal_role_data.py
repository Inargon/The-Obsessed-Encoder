"""Utilities for a controlled, hindsight-goal candidate benchmark."""
import numpy as np


def branch_targets(initial, traces, goal, budgets=(5, 10, 20, 40)):
    from budget_data import pose_hits
    traces = np.asarray(traces)
    if traces.ndim != 3 or traces.shape[-1] < 7 or max(budgets) > traces.shape[1]:
        raise ValueError('Need complete candidate traces through every budget')
    if not np.isfinite(traces).all():
        raise ValueError('Nonfinite branch')
    hit0 = bool(pose_hits(np.asarray(initial)[None], goal)[0])
    labels = []
    for trace in traces:
        hit = pose_hits(trace, goal)
        labels.append([hit0 or hit[:h].any() for h in budgets])
    return np.asarray(labels, bool), hit0


def group_split(groups, seed=73):
    unique = np.unique(groups)
    if len(unique) < 10:
        raise ValueError('Need ten independent initial episodes')
    order = np.random.default_rng(seed).permutation(unique)
    parts = np.split(order, [int(.6*len(order)), int(.8*len(order))])
    return [np.flatnonzero(np.isin(groups, part)) for part in parts]


def evaluate_predictions(pred, labels, initial_hit):
    """Q,G,V,C,H predictions; Q,G,C,H labels. Each Q is an episode."""
    pred, labels = np.asarray(pred), np.asarray(labels)
    if pred.ndim != 5 or pred.shape[:2]+pred.shape[3:] != labels.shape:
        raise ValueError('Mismatched predictions and labels')
    if not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 1)):
        raise ValueError('Invalid probabilities')
    if np.any(np.diff(pred, axis=-1) < -1e-6):
        raise ValueError('Nonmonotone budget probabilities')
    result = []
    for v in range(pred.shape[2]):
        pp = pred[:,:,v]
        selected = pp.argmax(axis=2)
        achieved = np.take_along_axis(labels, selected[:,:,None,:], axis=2).squeeze(2)
        possible = labels.any(axis=2)
        nontrivial = ~np.asarray(initial_hit, bool)
        opportunity = possible & nontrivial[:,:,None]
        result.append({'variant': v, 'brier_by_budget': ((pp-labels)**2).mean((0,1,2)).tolist(),
            'selected_hit_rate_by_budget': achieved.mean((0,1)).tolist(),
            'opportunities_by_budget': opportunity.sum((0,1)).tolist(),
            'missed_opportunities_by_budget': (opportunity & ~achieved).sum((0,1)).tolist(),
            'mean_abs_change_from_original': np.abs(pp-pred[:,:,0]).mean((0,1,2)).tolist()})
    return result
