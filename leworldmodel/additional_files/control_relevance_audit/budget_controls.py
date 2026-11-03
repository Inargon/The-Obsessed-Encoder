"""Matched input ablations for the recorded-continuation pilot."""
import numpy as np


def select_features(x, mode):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 28 or not np.isfinite(x).all():
        raise ValueError('Expected three eight-feature states plus four goal features')
    result = np.zeros_like(x)
    if mode == 'history':
        return x.copy()
    if mode == 'current':
        result[:,16:] = x[:,16:]
    elif mode == 'geometry':
        dx, dy = x[:,24]-x[:,18], x[:,25]-x[:,19]
        s, c = x[:,20], x[:,21]
        result[:,:5] = np.column_stack([c*dx+s*dy, -s*dx+c*dy,
            np.hypot(dx,dy), x[:,26]*c-x[:,27]*s, x[:,27]*c+x[:,26]*s])
    else:
        raise ValueError('Unknown feature mode')
    return result


def evaluation_masks(labels, known, remaining, max_budget):
    labels, known = np.asarray(labels,bool), np.asarray(known,bool)
    if labels.shape != known.shape or len(remaining) != len(labels):
        raise ValueError('Mismatched sample arrays')
    noninitial = ~labels[:,0]
    complete = np.asarray(remaining) >= max_budget
    return {'all_known': known, 'not_initially_hit': known & noninitial[:,None],
            'complete_followup': known & complete[:,None],
            'complete_followup_not_initially_hit': known & (complete & noninitial)[:,None]}
