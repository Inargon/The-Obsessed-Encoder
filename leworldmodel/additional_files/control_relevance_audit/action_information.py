"""Frozen linear action readouts; predictive input recoverability is not causality."""
import json
import argparse
from pathlib import Path
import numpy as np
from audit import split_episodes, score


def fit(x, y, train, alpha):
    mean = x[train].mean(0)
    std = x[train].std(0)
    std[std < 1e-8] = 1
    a = (x[train] - mean) / std
    ym = y[train].mean(0)
    target = y[train] - ym
    if a.shape[1] <= len(a):
        w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ target)
    else:
        w = a.T @ np.linalg.solve(a @ a.T + alpha * np.eye(len(a)), target)
    return mean, std, ym, w


def predict(model, x):
    mean, std, ym, w = model
    return ((x - mean) / std) @ w + ym


def readout(x, y, splits, transfer=None):
    train, val, test = splits
    alphas = (.01, .1, 1., 10., 100.)
    models = [fit(x, y, train, alpha) for alpha in alphas]
    losses = [np.mean((predict(m, x[val]) - y[val]) ** 2) for m in models]
    i = int(np.argmin(losses))
    result = dict(score(y[test], predict(models[i], x[test])), alpha=alphas[i])
    if transfer is not None:
        result['transfer_same_frozen_readout'] = score(y[test], predict(models[i], transfer[test]))
    return result


def evaluate(data, seed=73):
    keys = ('history', 'past_actions', 'real', 'predicted', 'action')
    ids = np.asarray(data['episode_id'])
    n = len(ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu':
        raise ValueError('Expected integer episode IDs')
    for k in keys:
        x = np.asarray(data[k])
        if x.ndim != 2 or len(x) != n or not np.isfinite(x).all():
            raise ValueError('Invalid array: ' + k)
    if data['real'].shape != data['predicted'].shape:
        raise ValueError('Endpoint shapes differ')
    splits = split_episodes(ids, seed)
    context = np.column_stack([data['history'], data['past_actions']])
    real = np.column_stack([context, data['real']])
    predicted = np.column_stack([context, data['predicted']])
    residual = data['predicted'] - data['real']
    shuffled_residual = residual.copy()
    rng = np.random.default_rng(seed + 1000)
    for indices in splits:
        shuffled_residual[indices] = residual[rng.permutation(indices)]
    inputs = {'history_and_past_actions': context, 'real_endpoint': real,
              'predicted_endpoint': predicted,
              'real_plus_residual': np.column_stack([real, residual]),
              'real_plus_shuffled_residual': np.column_stack([real, shuffled_residual]),
              'residual': np.column_stack([context, residual])}
    results = {k: readout(v, data['action'], splits,
                         transfer=predicted if k == 'real_endpoint' else None)
               for k, v in inputs.items()}
    train, _, test = splits
    results['train_mean'] = score(data['action'][test],
        np.broadcast_to(data['action'][train].mean(0), data['action'][test].shape))
    return {'seed': seed, 'counts': [len(s) for s in splits],
            'episodes': [np.unique(ids[s]).tolist() for s in splits], 'readouts': results,
            'endpoint_mse': float(np.mean(residual[test] ** 2)),
            'limitations': [
                'Predictor directly receives the target action; recoverability alone is not hidden coding.',
                'Linear probes measure accessibility, not total mutual information or irreversible loss.',
                'Added residual dimensions change ridge geometry; shuffled control is a diagnostic, not a conditional independence test.',
                'History includes all predictor image context and earlier actions, not the target action.',
                'Episode holdout applies to probes, not necessarily pretrained encoders.',
                'No intervention, closed-loop planning, or novelty conclusion follows from this audit.']}


def save_report(data, out):
    reports = [evaluate(data, seed) for seed in (73, 74, 75)]
    with out.open('x', encoding='utf-8') as f:
        json.dump(reports, f, indent=2, allow_nan=False)
    for report in reports:
        print('ACTION_READOUT', report['seed'], json.dumps(report['readouts']), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Re-read cached endpoints; no GPU extraction.')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bundles = {}
    for label in ('jepa', 'ours'):
        with np.load(args.cache / (label + '.npz'), allow_pickle=False) as f:
            bundles[label] = dict(f)
    for key in ('episode_id', 'row_index', 'clip_index', 'action', 'past_actions'):
        np.testing.assert_array_equal(bundles['jepa'][key], bundles['ours'][key])
    args.out.mkdir(parents=True, exist_ok=False)
    for label, bundle in bundles.items():
        print('MODEL', label, flush=True)
        reports = [evaluate(bundle, seed) for seed in (73, 74, 75)]
        with (args.out / (label + '-report.json')).open('x', encoding='utf-8') as f:
            json.dump(dict(cache=str(args.cache.resolve()), reports=reports), f, indent=2, allow_nan=False)
        for report in reports:
            print('CONDITIONAL_READOUT', label, report['seed'],
                  {k: v['r2'] for k, v in report['readouts'].items()}, flush=True)
    print('CONDITIONAL_ACTION_AUDIT_COMPLETE', args.out, flush=True)


if __name__ == '__main__': main()
