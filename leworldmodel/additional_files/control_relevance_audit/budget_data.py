"""Episode-disjoint recorded-continuation pilot; no optimal-reachability labels."""
import numpy as np
from budget_reachability import first_hit_targets


def episode_runs(ids, steps):
    ids, steps = np.asarray(ids).reshape(-1), np.asarray(steps).reshape(-1)
    if len(ids) != len(steps) or not len(ids):
        raise ValueError('Empty or mismatched episode/step arrays')
    cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    runs = [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]
    if len(set(ids[a].item() for a, _ in runs)) != len(runs):
        raise ValueError('Episode IDs recur in noncontiguous blocks')
    for a, b in runs:
        if not np.all(np.diff(steps[a:b]) == 1):
            raise ValueError('Nonconsecutive raw steps within episode')
    return runs


def pose_hits(states, goal, xy_tolerance=10., angle_tolerance=.15):
    angle = states[:, 4] - goal[2]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    return ((np.linalg.norm(states[:, 2:4] - goal[:2], axis=1) <= xy_tolerance)
            & (np.abs(angle) <= angle_tolerance))


def build_split(states, runs, samples, seed, budgets, history=3):
    """Goals sampled independently of the anchor continuation, within this split.

    Uniform episode then uniform row sampling. All arrays are recorded state
    diagnostics, not visual encoders or complete simulator states.
    """
    rng = np.random.default_rng(seed)
    usable = [(a, b) for a, b in runs if b-a >= history]
    if not usable:
        raise ValueError('No episodes with sufficient history')
    xs, hits, masks, provenance = [], [], [], []
    horizon = int(max(budgets))
    for _ in range(samples):
        a, b = usable[rng.integers(len(usable))]
        row = int(rng.integers(a+history-1, b))
        ga, gb = runs[rng.integers(len(runs))]
        gr = int(rng.integers(ga, gb))
        goal = states[gr, 2:5]
        hist = states[row-history+1:row+1]
        # Periodic angle representation, retaining recorded agent velocities.
        hist = np.column_stack([hist[:, :4], np.sin(hist[:, 4]),
                                np.cos(hist[:, 4]), hist[:, 5:7]])
        xs.append(np.r_[hist.ravel(), goal[:2], np.sin(goal[2]), np.cos(goal[2])])
        end = min(b, row+horizon+1)
        hit, observed = np.zeros(horizon+1, bool), np.zeros(horizon+1, bool)
        observed[:end-row] = True
        hit[:end-row] = pose_hits(states[row:end], goal)
        hits.append(hit); masks.append(observed); provenance.append((row, gr))
    y, known = first_hit_targets(np.asarray(hits), np.asarray(masks), budgets)
    return np.asarray(xs, np.float32), y, known, np.asarray(provenance)
