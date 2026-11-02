"""Matched privileged-state pilot; evaluates response differences, not planning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from audit import split_episodes
from effect_set_pilot import MLP


def contrast(y):
    return y - y.mean(axis=-2, keepdims=True)


def pair_differences(y):
    i, j = np.triu_indices(y.shape[-2], 1)
    return y[..., i, :] - y[..., j, :]


def fit(x, y, splits, mode, seed, epochs=500, width=64):
    tr, va, te = splits
    xm, xs = x[tr].mean(0), x[tr].std(0)
    xs[xs < 1e-8] = 1
    x = (x - xm) / xs
    # Shared scalar scale preserves relative coordinate weighting in BOTH arms.
    ym = y[tr].mean(axis=(0, 1), keepdims=True)
    scale = max(float(np.sqrt(np.mean((y[tr] - ym) ** 2))), 1e-8)
    target = (y - ym) / scale
    if mode == 'contrast':
        target = contrast(target)
    elif mode != 'endpoint':
        raise ValueError(mode)
    model = MLP(x.shape[1], y.shape[1] * y.shape[2], width, seed)
    best, chosen = float('inf'), None
    for epoch in range(1, epochs + 1):
        loss = model.update(x[tr], target[tr].reshape(len(tr), -1), False, .003)
        if epoch % 10 == 0 or epoch == epochs:
            pred = model.predict(x[va]).reshape(y[va].shape) * scale
            error = float(np.mean((pair_differences(pred) - pair_differences(y[va])) ** 2))
            if error < best:
                best, chosen = error, (epoch, copy.deepcopy(model.params))
    model.params = chosen[1]
    pred = model.predict(x[te]).reshape(y[te].shape) * scale
    delta = pair_differences(pred) - pair_differences(y[te])
    return {'selected_epoch': chosen[0], 'validation_difference_mse': best,
            'test_difference_mse': float(np.mean(delta ** 2)),
            'test_difference_mse_per_coordinate': np.mean(delta ** 2, axis=(0, 1)).tolist(),
            'test_scene_difference_mse': np.mean(delta ** 2, axis=(1, 2)).tolist(),
            'parameters': sum(p.size for p in model.params), 'target_scale': scale,
            'last_train_loss_not_comparable_across_arms': loss}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=500)
    args = p.parse_args()
    if args.epochs < 1 or args.out.exists():
        p.error('Positive epochs and new output required')
    source = args.input_dir / 'branches.npz'
    with np.load(source, allow_pickle=False) as data:
        starts, states, actions = data['initial_states'], data['states'], data['actions']
    if starts.ndim != 3 or starts.shape[1:] != (2, 7):
        raise ValueError('Expected paired seven-state fixture')
    n = len(starts)
    if states.ndim != 5 or states.shape[:2] != (n, 2) or states.shape[-1] != 7:
        raise ValueError('Invalid trajectory layout')
    k = states.shape[2]
    if k < 2 or actions.shape != (k, 2):
        raise ValueError('Invalid action bank')
    initial = starts.reshape(-1, 7)
    x = np.column_stack((initial[:, :4], np.sin(initial[:, 4]),
                         np.cos(initial[:, 4]), initial[:, 5:]))
    final = states[:, :, :, -1, 2:5].reshape(n * 2, k, 3)
    # Fixed task-space embedding: chordal angle features, NOT wrapped angle cost.
    y = np.concatenate((final[..., :2], 30 * np.sin(final[..., 2:3]),
                        30 * np.cos(final[..., 2:3])), axis=-1)
    if not all(np.isfinite(a).all() for a in (x, y, actions)):
        raise ValueError('Nonfinite fixture')
    ids = np.repeat(np.arange(n), 2)
    report = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'protocol': {'epochs': args.epochs, 'width': 64, 'actions': actions.tolist(),
                'target': '[block_x, block_y, 30*sin(angle), 30*cos(angle)]',
                'selection': 'Same validation pairwise response-difference MSE',
                'limitations': ['Privileged state, no learned visual representation',
                    'Fixed action bank; no unseen-action test or closed-loop planning',
                    'Contrast drops common-mode outcome; insufficient alone for absolute goals',
                    'Splits overlap and training seeds share scenes; not independent replicates',
                    'Equal parameters, initialization and updates; objectives have different difficulty',
                    'No sample-efficiency or nuisance-robustness claim from this experiment']},
              'splits': [], 'runs': []}
    for split_seed in (73, 74, 75):
        splits = split_episodes(ids, split_seed)
        report['splits'].append({'seed': split_seed, **{
            name: np.unique(ids[rows]).tolist() for name, rows in
            zip(('train', 'validation', 'test'), splits)}})
        tr, va, te = splits
        baseline = pair_differences(y[tr].mean(axis=0, keepdims=True))
        baseline_mse = float(np.mean((pair_differences(y[te]) - baseline) ** 2))
        for seed in (901, 902, 903):
            for mode in ('endpoint', 'contrast'):
                result = fit(x, y, splits, mode, seed, args.epochs)
                row = {'split': split_seed, 'seed': seed, 'mode': mode,
                       'train_mean_difference_mse': baseline_mse, **result}
                report['runs'].append(row)
                print('RESPONSE_DIFFERENCE', split_seed, seed, mode,
                      result['test_difference_mse'], 'BASELINE', baseline_mse, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print('RESPONSE_DIFFERENCE_COMPLETE', args.out, flush=True)


if __name__ == '__main__':
    main()
