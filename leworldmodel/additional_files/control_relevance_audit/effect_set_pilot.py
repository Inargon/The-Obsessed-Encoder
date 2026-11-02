"""Privileged-state pilot: identity labels vs outcome sets vs forward prediction.

This does not train a visual encoder or establish novelty. All fixture branches
of a paired initial scene stay in one split. Test outcomes only score predictions.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from audit import split_episodes


def pose_features(pose):
    return np.concatenate((pose[..., :2], np.sin(pose[..., 2:3]),
                           np.cos(pose[..., 2:3])), axis=-1)


def costs_to_goal(final, goal):
    """Fixed inherited fixture metric, not tuned on held-out outcomes."""
    angle = final[..., 2] - goal[..., 2]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    return ((final[..., :2] - goal[..., :2]) ** 2).sum(-1) + (30 * angle) ** 2


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def outcome_targets(final, train):
    distance = costs_to_goal(final[:, None, :, :], final[:, :, None, :])
    positive = distance[train][distance[train] > 1e-8]
    temperature = float(np.median(positive)) if len(positive) else 1.0
    return softmax(-distance / temperature), temperature


class MLP:
    def __init__(self, n_in, n_out, width, seed):
        rng = np.random.default_rng(seed)
        self.params = [rng.normal(0, np.sqrt(1 / n_in), (n_in, width)),
                       np.zeros(width),
                       rng.normal(0, np.sqrt(1 / width), (width, n_out)),
                       np.zeros(n_out)]
        self.m = [np.zeros_like(w) for w in self.params]
        self.v = [np.zeros_like(w) for w in self.params]
        self.step = 0

    def predict(self, x):
        w, b, v, c = self.params
        return np.tanh(x @ w + b) @ v + c

    def update(self, x, y, classification, lr):
        w, b, v, c = self.params
        h = np.tanh(x @ w + b)
        out = h @ v + c
        if classification:
            prob = softmax(out)
            loss = -np.mean(np.sum(y * np.log(np.maximum(prob, 1e-300)), axis=1))
            grad = (prob - y) / len(x)
        else:
            loss = np.mean((out - y) ** 2)
            grad = 2 * (out - y) / out.size
        dh = (grad @ v.T) * (1 - h * h)
        gradients = [x.T @ dh, dh.sum(0), h.T @ grad, grad.sum(0)]
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients):
            raise ValueError('Nonfinite optimization')
        self.step += 1
        for i, g in enumerate(gradients):
            self.m[i] = .9 * self.m[i] + .1 * g
            self.v[i] = .999 * self.v[i] + .001 * g * g
            self.params[i] -= lr * (self.m[i] / (1 - .9 ** self.step)) / (
                np.sqrt(self.v[i] / (1 - .999 ** self.step)) + 1e-8)
        return float(loss)


def fit_arm(state, final, splits, mode, seed, epochs, width):
    tr, va, te = splits
    n, k, _ = final.shape
    q, temperature = outcome_targets(final, tr)
    classification = mode != 'forward'
    if classification:
        raw = np.concatenate((np.repeat(state[:, None, :], k, axis=1),
                              pose_features(final)), axis=-1)
        y = q if mode == 'effect_set' else np.broadcast_to(np.eye(k), (n, k, k)).copy()
    else:
        raw = state[:, None, :]
        y = pose_features(final).reshape(n, 1, k * 4)
    xm = raw[tr].reshape(-1, raw.shape[-1]).mean(0)
    xs = raw[tr].reshape(-1, raw.shape[-1]).std(0)
    xs[xs < 1e-8] = 1
    x = (raw - xm) / xs
    if classification:
        ym, ys = 0., 1.
    else:
        ym = y[tr].mean(axis=(0, 1))
        ys = y[tr].std(axis=(0, 1))
        ys[ys < 1e-8] = 1
    target = (y - ym) / ys
    model = MLP(x.shape[-1], y.shape[-1], width, seed)

    def predict_scores(rows, goals):
        # goals: [scene, query, pose]; never supply realized branch outcomes here.
        if classification:
            query = np.concatenate((np.repeat(state[rows, None, :], goals.shape[1], axis=1),
                                    pose_features(goals)), axis=-1)
            return -model.predict(((query - xm) / xs).reshape(-1, x.shape[-1])).reshape(
                len(rows), goals.shape[1], k)
        pred = model.predict(x[rows, 0]) * ys + ym
        pred = pred.reshape(len(rows), k, 4)
        pose = np.concatenate((pred[..., :2],
                               np.arctan2(pred[..., 2:3], pred[..., 3:4])), axis=-1)
        return costs_to_goal(pose[:, None, :, :], goals[:, :, None, :])

    true_validation = costs_to_goal(final[va, None, :, :], final[va, :, None, :])
    best = float('inf')
    chosen = None
    for epoch in range(1, epochs + 1):
        loss = model.update(x[tr].reshape(-1, x.shape[-1]),
                            target[tr].reshape(-1, target.shape[-1]), classification, .003)
        if epoch % 10 == 0 or epoch == epochs:
            scores = predict_scores(va, final[va])
            selected = scores.argmin(-1)
            regret = np.take_along_axis(true_validation, selected[..., None], -1)[..., 0]
            value = float(np.mean(regret - true_validation.min(-1)))
            if value < best:
                best, chosen = value, (epoch, copy.deepcopy(model.params))
    model.params = chosen[1]
    info = {'selected_epoch': chosen[0], 'validation_regret': best,
            'last_training_loss': loss, 'parameters': sum(w.size for w in model.params),
            'temperature_train_only': temperature,
            'train_target_entropy': float(-np.mean(np.sum(q[tr] * np.log(np.maximum(q[tr], 1e-300)), -1)))}
    return predict_scores, info


def metrics(final, goals, scores, pair_ids):
    true = costs_to_goal(final[:, None, :, :], goals[:, :, None, :])
    selected = scores.argmin(-1)
    regret = np.take_along_axis(true, selected[..., None], -1)[..., 0] - true.min(-1)
    return {'mean_candidate_regret': float(regret.mean()),
            'optimal_within_1e_6_fraction': float((regret <= 1e-6).mean()),
            'regret_per_pair': {str(int(p)): float(regret[pair_ids == p].mean())
                                for p in np.unique(pair_ids)},
            'selected_candidates': selected.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', default=[901, 902, 903])
    args = parser.parse_args()
    if args.out.exists() or args.epochs < 1 or args.width < 1:
        parser.error('Output must be new; epochs and width must be positive')
    source = args.input_dir / 'branches.npz'
    with np.load(source, allow_pickle=False) as f:
        starts = f['initial_states']
        trajectories = f['states']
        goals = f['goals']
        actions = f['actions']
        costs = f['candidate_costs']
    if starts.ndim != 3 or starts.shape[1:] != (2, 7):
        raise ValueError('Expected paired seven-state fixture')
    n = len(starts)
    if trajectories.ndim != 5 or trajectories.shape[:2] != (n, 2) or trajectories.shape[-1] != 7:
        raise ValueError('Invalid trajectory layout')
    k = trajectories.shape[2]
    if k < 2 or actions.shape != (k, 2) or goals.shape != (n, 3):
        raise ValueError('Expected fixed bank of at least two constant 2D commands')
    final = trajectories[:, :, :, -1, 2:5].reshape(n * 2, k, 3)
    initial = starts.reshape(n * 2, 7)
    state = np.column_stack((initial[:, :4], np.sin(initial[:, 4]),
                             np.cos(initial[:, 4]), initial[:, 5:]))
    goals = np.repeat(goals, 2, axis=0)[:, None, :]
    for arr in (state, final, goals, actions, costs):
        if not np.isfinite(arr).all():
            raise ValueError('Nonfinite fixture')
    np.testing.assert_allclose(costs.reshape(n * 2, k),
        costs_to_goal(final, goals), atol=1e-7, rtol=1e-10)
    ids = np.repeat(np.arange(n), 2)
    splits = split_episodes(ids, 73)
    tr, va, te = splits
    report = {'fixture_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'split_pair_ids': {name: np.unique(ids[ix]).tolist()
                                 for name, ix in zip(('train', 'validation', 'test'), splits)},
              'protocol': {'epochs': args.epochs, 'width': args.width, 'actions': actions.tolist(),
                'selection': 'Minimum validation regret on achieved object goals; train-only scaling',
                'angle_scale': 'Fixed 30 pixels/radian inherited from fixture metric',
                'limitations': ['Privileged state pilot; no visual encoder or closed-loop control training',
                    'Three fixed candidate commands do not test unseen-action generalization',
                    'All arms access the same branch outcomes; input dimensions and parameter counts differ',
                    'Same hidden width and optimizer steps, not exact FLOP matching',
                    'One pair-group split; training seeds share test scenes',
                    'Achieved goals are supplied as requested targets, not candidate outcome vectors',
                    'No claims of novelty or causal identification']}, 'runs': []}
    for seed in args.seeds:
        run = {'seed': seed, 'arms': {}}
        for mode in ('identity', 'forward', 'effect_set'):
            predict, fit = fit_arm(state, final, splits, mode, seed, args.epochs, args.width)
            results = {'fit': fit}
            for label, target_goals in (('achieved', final[te]), ('declared', goals[te])):
                results[label] = metrics(final[te], target_goals, predict(te, target_goals), ids[te])
            run['arms'][mode] = results
            print('EFFECT_SET_PILOT', seed, mode, json.dumps({
                label: results[label]['mean_candidate_regret'] for label in ('achieved', 'declared')}), flush=True)
        report['runs'].append(run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print('EFFECT_SET_PILOT_COMPLETE', args.out, flush=True)


if __name__ == '__main__':
    main()
