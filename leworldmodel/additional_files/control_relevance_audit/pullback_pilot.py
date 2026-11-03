"""Frozen JEPA directional pullback audit. D=identity, not a task decoder.

Finite differences estimate J(z)(z2-z1); no full metric matrix is materialized.
Existing simulator outcomes are evaluation labels only. No model fitting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np


def rank_corr(x, y):
    def ranks(a):
        a = np.asarray(a, dtype=float).ravel()
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite scores')
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts + 1) / 2)[inv]
    x, y = ranks(x), ranks(y)
    if len(x) != len(y):
        raise ValueError('Score length mismatch')
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def physical_differences(states, horizon):
    """Same action across two scenes, terminal pose AND displacement differences."""
    poses = states[..., horizon, 2:5]
    delta = poses[:, 1] - poses[:, 0]
    delta[..., 2] = np.arctan2(np.sin(delta[..., 2]), np.cos(delta[..., 2]))
    movement = poses - states[..., 0, 2:5]
    movement[..., 2] = np.arctan2(np.sin(movement[..., 2]), np.cos(movement[..., 2]))
    md = movement[:, 1] - movement[:, 0]
    md[..., 2] = np.arctan2(np.sin(md[..., 2]), np.cos(md[..., 2]))
    return {'terminal_xy': np.linalg.norm(delta[..., :2], axis=-1),
            'terminal_angle': np.abs(delta[..., 2]),
            'displacement_xy': np.linalg.norm(md[..., :2], axis=-1),
            'displacement_angle': np.abs(md[..., 2])}


def directional_response(fn, z, delta, epsilon):
    if epsilon <= 0:
        raise ValueError('Positive finite-difference step required')
    return (fn(z + epsilon * delta) - fn(z - epsilon * delta)) / (2 * epsilon)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch-pairs', type=int, default=8)
    p.add_argument('--models', nargs='+', choices=['jepa', 'ours'], default=['jepa', 'ours'])
    args = p.parse_args()
    if args.out.exists() or args.batch_pairs < 1:
        p.error('New output and positive batch size required')
    from state_terminal_readout import aligned_visual_inputs
    source = args.fixture / 'branches.npz'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    meta = json.loads((args.cache / 'report.json').read_text())
    if meta['input_sha256'] != digest:
        raise ValueError('Fixture/cache mismatch')
    with np.load(source, allow_pickle=False) as f:
        states, initial, goals, bank = (f[k] for k in ('states', 'initial_states', 'goals', 'actions'))
        reference_image = f['initial_pixels'][0, 0].copy()
    n, scenes, k, times, dim = states.shape
    if scenes != 2 or dim != 7 or bank.shape != (k, 2) or times <= 40:
        raise ValueError('Expected paired forty-step constant-command fixture')
    if not all(np.isfinite(a).all() for a in (states, initial, goals, bank)):
        raise ValueError('Nonfinite fixture')
    with np.load(args.cache / 'features.npz', allow_pickle=False) as f:
        inputs = aligned_visual_inputs(dict(f), initial.reshape(-1, 7),
                                       np.repeat(goals, 2, axis=0), np.repeat(np.arange(n), 2))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    import stable_worldmodel as swm
    from hydra.utils import instantiate
    from utils import get_column_normalizer, get_img_preprocessor
    from additional_files.mechanism_metrics import normalize_action_blocks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = swm.data.load_dataset(str(Path(os.environ['STABLEWM_HOME']) / 'datasets/pusht_expert_train.h5'),
        cache_dir=os.environ['LOCAL_DATASET_DIR'], num_steps=4, frameskip=5,
        keys_to_load=['action'], keys_to_cache=['action'])
    norm = get_column_normalizer(dataset, 'action', 'action')
    packed = np.tile(bank, (1, 5)).astype(np.float32)
    normalized = normalize_action_blocks(torch.from_numpy(packed), norm, 2).to(device)
    preprocess = get_img_preprocessor('pixels', 'pixels', img_size=224)
    image = torch.from_numpy(reference_image).permute(2, 0, 1)[None]
    image = preprocess({'pixels': image})['pixels'].to(device)
    report = {'fixture_sha256': digest, 'pairs': n, 'runs': [], 'protocol': {
        'decoder': 'identity; no task-state decoder, no new training',
        'horizons_native_steps': [5, 20, 40], 'frameskip': 5,
        'history': 'one real initial frame then autoregressive predictions; max context 3',
        'finite_difference': 'central difference at both pair endpoints; epsilon .01 and .005 times pair delta',
        'limitations': ['Previously inspected constructed fixture, exploratory not held-out confirmation',
            'All pairs retained; no outcome-based filtering or tuning',
            'Interpolated latents need not be physical states; local derivative can be off manifold',
            'Pairs may be far apart; finite-displacement comparison tests locality, not reachability',
            'Raw latent output includes task-irrelevant information; not task-conditioned geometry',
            'Only three constant action sequences; no policy, no unseen-action or speed claim',
            'Flattened observations are dependent; correlations have no independence/significance claim']}}
    args.out.mkdir(parents=True, exist_ok=False)
    for name in args.models:
        cp = Path(meta['checkpoints'][name]['checkpoint'])
        if hashlib.sha256(cp.read_bytes()).hexdigest() != meta['checkpoints'][name]['sha256']:
            raise ValueError('Checkpoint changed')
        model = instantiate(json.loads((cp.parent / 'config.json').read_text()))
        weights = torch.load(cp, map_location='cpu', weights_only=True)
        model.load_state_dict({key: value for key, value in weights.items()
                               if not key.startswith('control_objective.')}, strict=True)
        model = model.to(device).float().eval().requires_grad_(False)
        z = torch.as_tensor(inputs[name].reshape(n, 2, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            encoded = model.encode({'pixels': image[:, None]})['emb'][0, 0]
            torch.testing.assert_close(encoded, z[0, 0], rtol=1e-4, atol=1e-4)
        for horizon in (5, 20, 40):
            collected = {key: [] for key in ('euclidean', 'direct', 'local_01', 'local_005',
                                            'fd_relative_change', 'endpoint_derivative_disagreement')}
            for start in range(0, n, args.batch_pairs):
                b = z[start:start + args.batch_pairs]
                z0, z1 = (b[:, i, None].expand(-1, k, -1).reshape(-1, z.shape[-1]) for i in (0, 1))
                delta = z1 - z0
                a = normalized.repeat(len(b), 1)
                def forward(x):
                    history = x[:, None]
                    for _ in range(horizon // 5):
                        context = history[:, -3:]
                        ae = model.action_encoder(a[:, None].expand(-1, context.shape[1], -1))
                        nxt = model.predict(context, ae)[:, -1:]
                        history = torch.cat((context, nxt), dim=1)
                    return history[:, -1]
                with torch.no_grad():
                    if start == 0:
                        native = model.rollout(
                            {'pixels': image[:, None, None].expand(-1, k, -1, -1, -1, -1)},
                            normalized[None, :, None].expand(1, k, horizon // 5, -1),
                            history_size=3)['predicted_emb'][0, :, -1]
                        torch.testing.assert_close(forward(z0)[:k], native, rtol=1e-4, atol=1e-4)
                        print('NATIVE_ROLLOUT_VERIFIED', name, horizon, flush=True)
                    direct = forward(z1) - forward(z0)
                    d0 = directional_response(forward, z0, delta, .01)
                    d1 = directional_response(forward, z1, delta, .01)
                    fine0 = directional_response(forward, z0, delta, .005)
                    fine1 = directional_response(forward, z1, delta, .005)
                    length = lambda x: torch.linalg.vector_norm(x, dim=-1)
                    values = {'euclidean': length(delta), 'direct': length(direct),
                        'local_01': torch.sqrt((length(d0)**2 + length(d1)**2) / 2),
                        'local_005': torch.sqrt((length(fine0)**2 + length(fine1)**2) / 2),
                        'fd_relative_change': (length(fine0-d0)+length(fine1-d1)) /
                            (length(fine0)+length(fine1)).clamp_min(1e-8),
                        'endpoint_derivative_disagreement': length(fine0-fine1) /
                            (length(fine0)+length(fine1)).clamp_min(1e-8)}
                for key, value in values.items():
                    collected[key].append(value.cpu().numpy().reshape(len(b), k))
            scores = {key: np.concatenate(value) for key, value in collected.items()}
            if not all(np.isfinite(v).all() for v in scores.values()):
                raise ValueError('Nonfinite output')
            truth = physical_differences(states, horizon)
            row = {'model': name, 'horizon': horizon,
                'spearman': {key: {target: rank_corr(scores[key], val) for target, val in truth.items()}
                             for key in ('euclidean', 'direct', 'local_01', 'local_005')},
                'fd_relative_change_median': float(np.median(scores['fd_relative_change'])),
                'endpoint_derivative_disagreement_median': float(np.median(scores['endpoint_derivative_disagreement'])),
                'checkpoint_sha256': meta['checkpoints'][name]['sha256']}
            np.savez_compressed(args.out / f'{name}-h{horizon}.npz', **scores, **truth, pair_ids=np.arange(n))
            report['runs'].append(row)
            print('PULLBACK_PILOT', json.dumps(row, allow_nan=False), flush=True)
            (args.out / 'report.partial.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        del model
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('PULLBACK_PILOT_COMPLETE', args.out, flush=True)


if __name__ == '__main__':
    main()
