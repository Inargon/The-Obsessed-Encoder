"""Equal-capacity endpoint/common-contrast regression with candidate decision audit.

Privileged-state diagnostic, not a new representation-learning method.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from audit import split_episodes
from effect_set_pilot import MLP, costs_to_goal
from response_difference import pair_differences


def basis(k):
    """Orthonormal Helmert basis: common mode then k-1 independent contrasts."""
    if k < 2:
        raise ValueError('At least two candidates required')
    b = np.zeros((k, k))
    b[:, 0] = 1 / np.sqrt(k)
    for j in range(1, k):
        b[:j, j] = 1 / np.sqrt(j * (j + 1))
        b[j, j] = -j / np.sqrt(j * (j + 1))
    return b


def encode(y, b, weights):
    return np.einsum('nkd,kj->njd', y, b) * np.sqrt(weights)[None, :, None]


def decode(y, b, weights):
    return np.einsum('njd,kj->nkd', y / np.sqrt(weights)[None, :, None], b)


def pose(y):
    return np.concatenate((y[..., :2], np.arctan2(y[..., 2:3], y[..., 3:4])), -1)


def decision_metrics(pred, true, goals):
    predicted_costs = costs_to_goal(pose(pred), goals[:, None, :])
    true_costs = costs_to_goal(pose(true), goals[:, None, :])
    chosen = predicted_costs.argmin(-1)
    regret = true_costs[np.arange(len(true)), chosen] - true_costs.min(-1)
    delta = pair_differences(pred) - pair_differences(true)
    return {'mean_candidate_regret': float(regret.mean()),
            'optimal_within_1e_6_fraction': float(np.mean(regret <= 1e-6)),
            'difference_mse': float(np.mean(delta ** 2)),
            'endpoint_embedding_mse': float(np.mean((pred - true) ** 2)),
            'xy_rmse_px': float(np.sqrt(np.mean((pred[..., :2] - true[..., :2]) ** 2))),
            'scene_regrets': regret.tolist(), 'chosen_candidates': chosen.tolist()}


def fit(x, y, goals, splits, mode, seed, epochs=500, width=64):
    tr, va, te = splits
    k, d = y.shape[1:]
    b = np.eye(k) if mode == 'endpoint' else basis(k)
    weights = np.ones(k)
    if mode == 'contrast_x2':
        weights[1:] = 2
    elif mode not in ('endpoint', 'decomposed'):
        raise ValueError(mode)
    xm, xs = x[tr].mean(0), x[tr].std(0)
    xs[xs < 1e-8] = 1
    x = (x - xm) / xs
    ym = y[tr].mean(axis=(0, 1), keepdims=True)
    scale = max(float(np.sqrt(np.mean((y[tr] - ym) ** 2))), 1e-8)
    target = encode((y - ym) / scale, b, weights).reshape(len(y), -1)
    model = MLP(x.shape[1], k * d, width, seed)
    # Transform output weights, preserving the same initial physical predictions.
    model.params[2] = encode(model.params[2].reshape(width, k, d), b, weights).reshape(width, -1)
    model.params[3] = encode(model.params[3].reshape(1, k, d), b, weights).reshape(-1)

    def predict(rows):
        return decode(model.predict(x[rows]).reshape(len(rows), k, d), b, weights) * scale + ym

    best, chosen = float('inf'), None
    for epoch in range(1, epochs + 1):
        loss = model.update(x[tr], target[tr], False, .003)
        if epoch % 10 == 0 or epoch == epochs:
            value = decision_metrics(predict(va), y[va], goals[va])['mean_candidate_regret']
            if value < best:
                best, chosen = value, (epoch, copy.deepcopy(model.params))
    model.params = chosen[1]
    return {'selected_epoch': chosen[0], 'validation_candidate_regret': best,
            'parameters': sum(p.size for p in model.params), 'target_scale': scale,
            'last_training_loss_not_comparable': loss,
            **decision_metrics(predict(te), y[te], goals[te])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()
    if args.out.exists() or args.epochs < 1:
        parser.error('New output and positive epochs required')
    source = args.input_dir / 'branches.npz'
    with np.load(source, allow_pickle=False) as data:
        starts, states = data['initial_states'], data['states']
        goals, actions, costs = data['goals'], data['actions'], data['candidate_costs']
    if starts.ndim != 3 or starts.shape[1:] != (2, 7):
        raise ValueError('Expected paired seven-state fixture')
    n = len(starts)
    if states.ndim != 5 or states.shape[:2] != (n, 2) or states.shape[-1] != 7:
        raise ValueError('Invalid trajectory layout')
    k = states.shape[2]
    if k < 2 or actions.shape != (k, 2) or goals.shape != (n, 3):
        raise ValueError('Invalid action bank/goals')
    initial = starts.reshape(-1, 7)
    x = np.column_stack((initial[:, :4], np.sin(initial[:, 4]),
                         np.cos(initial[:, 4]), initial[:, 5:]))
    final = states[:, :, :, -1, 2:5].reshape(n * 2, k, 3)
    y = np.concatenate((final[..., :2], 30 * np.sin(final[..., 2:3]),
                        30 * np.cos(final[..., 2:3])), -1)
    goals = np.repeat(goals, 2, axis=0)
    if not all(np.isfinite(a).all() for a in (x, y, goals, actions, costs)):
        raise ValueError('Nonfinite fixture')
    np.testing.assert_allclose(costs.reshape(n * 2, k), costs_to_goal(final, goals[:, None]),
                               atol=1e-7, rtol=1e-10)
    ids = np.repeat(np.arange(n), 2)
    report = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'protocol': {'epochs': args.epochs, 'width': 64, 'actions': actions.tolist(),
            'selection': 'Minimum validation declared-goal candidate regret; earliest tie',
            'contrast_weight': 2, 'learning_rate': .003,
            'limitations': ['Privileged state; not visual encoder training',
                'Only three fixed actions and existing goal distribution, no closed-loop evaluation',
                'Overlapping splits and repeated seeds are not independent experiments',
                'Same parameter count and initial physical predictions; Adam is not rotation invariant',
                'Decomposed unweighted MSE equals endpoint MSE algebraically',
                'Contrast x2 is loss reweighting, not a novelty claim',
                'Previous tests informed this pilot; confirmation needs a fresh fixture']},
        'splits': [], 'runs': []}
    for split_seed in (73, 74, 75):
        splits = split_episodes(ids, split_seed)
        tr, va, te = splits
        report['splits'].append({'seed': split_seed, **{
            name: np.unique(ids[ix]).tolist() for name, ix in
            zip(('train', 'validation', 'test'), splits)}})
        baseline = np.repeat(y[tr].mean(0, keepdims=True), len(te), axis=0)
        baseline_regret = decision_metrics(baseline, y[te], goals[te])['mean_candidate_regret']
        for seed in (901, 902, 903):
            for mode in ('endpoint', 'decomposed', 'contrast_x2'):
                result = fit(x, y, goals, splits, mode, seed, args.epochs)
                report['runs'].append({'split': split_seed, 'seed': seed, 'mode': mode,
                                      'train_mean_candidate_regret': baseline_regret, **result})
                print('RESPONSE_PLANNING', split_seed, seed, mode, json.dumps({key: result[key]
                    for key in ('mean_candidate_regret', 'difference_mse', 'xy_rmse_px')}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print('RESPONSE_PLANNING_COMPLETE', args.out, flush=True)


if __name__ == '__main__':
    main()
