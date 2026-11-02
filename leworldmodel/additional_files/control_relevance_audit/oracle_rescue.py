"""Frozen-feature diagnostic with equally sized initial-state oracle additions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit import split_episodes
from state_terminal_readout import aligned_visual_inputs, encode_pose, terminal_cost, fit_mlp
from readout_delayed_pairs import choice_metrics


def oracle_inputs(features, initial, ids, splits, seed):
    block = encode_pose(initial[:, 2:5])
    agent = initial[:, [0, 1, 5, 6]]
    shuffled = block.copy()
    rng = np.random.default_rng(seed)
    seen = []
    for rows in splits:
        groups = np.unique(ids[rows])
        if len(groups) < 2:
            raise ValueError('Need at least two pairs per split for shuffle control')
        order = rng.permutation(groups)
        for source, dest in zip(order, np.roll(order, 1)):
            src = np.flatnonzero(ids == source)
            dst = np.flatnonzero(ids == dest)
            if len(src) != 2 or len(dst) != 2 or not np.isin(src, rows).all() or not np.isin(dst, rows).all():
                raise ValueError('Pair groups must be intact and disjoint across splits')
            shuffled[dst] = block[src]
        seen.extend(rows.tolist())
    if sorted(seen) != list(range(len(ids))):
        raise ValueError('Splits must partition all rows exactly once')
    addons = {'zero': np.zeros_like(block), 'block': block,
              'agent': agent, 'shuffled_block': shuffled}
    return {name + '+' + label: np.column_stack((x, extra))
            for name, x in features.items() for label, extra in addons.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): p.error('Refusing to overwrite output directory')
    source = args.fixture / 'branches.npz'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    meta = json.loads((args.cache / 'report.json').read_text())
    if meta['input_sha256'] != digest:
        raise ValueError('Cached features and fixture have different provenance')
    with np.load(source, allow_pickle=False) as f:
        initial = f['initial_states'].reshape(-1, 7)
        n = len(initial)
        final = f['states'][:, :, :, -1, 2:5].reshape(n, -1, 3)
        goals = np.repeat(f['goals'], 2, axis=0)
        costs = f['candidate_costs'].reshape(n, -1)
        eligible = f['delayed_mask'] & f['candidate_choice_diff']
    ids = np.repeat(np.arange(n // 2), 2)
    with np.load(args.cache / 'features.npz', allow_pickle=False) as f:
        visual = aligned_visual_inputs(dict(f), initial, goals, ids)
    visual = {name: visual[name] for name in ('jepa', 'ours')}
    target = encode_pose(final).reshape(n, -1)
    if not all(np.isfinite(a).all() for a in (initial, target, goals, costs)):
        raise ValueError('Nonfinite fixture')
    np.testing.assert_allclose(terminal_cost(encode_pose(final), goals), costs, rtol=1e-10, atol=1e-7)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'fixture_sha256': digest, 'checkpoints': meta['checkpoints'],
              'protocol': {'primary_split': 73, 'sensitivity_splits': [74, 75],
                 'training_seed': 901, 'readout': 'Existing fit_mlp: widths 64/128, decay 0/.001, 600 epochs',
                 'selection': 'Train-only normalization, validation-only normalized terminal-pose MSE',
                 'oracle': 'Initial block pose only; no terminal state as input; agent x,y,vx,vy control',
                 'shuffle': 'Non-self permutation of whole scene pairs, independently within each split',
                 'capacity': 'Each visual model has four appended dimensions in every arm',
                 'limitations': ['Frozen features only; privileged oracle unavailable at visual deployment',
                    'Oracle gains do not prove information was absent from original features',
                    'Single training seed; three overlapping scene splits, not independent replications',
                    'Three fixed actions in a constructed distribution; no closed-loop planning',
                    'Original images may contain a goal outline despite stripping explicit goal suffix']},
              'runs': []}
    for seed in (73, 74, 75):
        splits = split_episodes(ids, seed)
        te = splits[-1]
        inputs = oracle_inputs(visual, initial, ids, splits, seed)
        inputs['recorded_state'] = np.column_stack((initial[:, :4], np.sin(initial[:, 4]),
                                                    np.cos(initial[:, 4]), initial[:, 5:]))
        run = {'split': seed, 'test_pairs': np.unique(ids[te]).tolist(), 'models': {}}
        for name, x in inputs.items():
            config, pred = fit_mlp(x, target, splits)
            encoded = pred.reshape(len(te), final.shape[1], 4)
            inferred = terminal_cost(encoded, goals[te])
            result = choice_metrics(costs[te], inferred, ids[te], eligible)
            result['xy_rmse_px'] = float(np.sqrt(np.mean((encoded[..., :2] - final[te, ..., :2])**2)))
            angle = np.arctan2(encoded[..., 2], encoded[..., 3]) - final[te, ..., 2]
            result['angle_mae_rad'] = float(np.mean(np.abs(np.arctan2(np.sin(angle), np.cos(angle)))))
            run['models'][name] = {'fit': config, 'metrics': result}
            print('ORACLE_RESCUE', seed, name, json.dumps({k: result[k] for k in
                ('mean_candidate_regret', 'xy_rmse_px', 'angle_mae_rad')}), flush=True)
        report['runs'].append(run)
        with (args.out / ('split-' + str(seed) + '.json')).open('x') as f:
            json.dump(run, f, indent=2, allow_nan=False)
    with (args.out / 'report.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print('ORACLE_RESCUE_COMPLETE', args.out / 'report.json', flush=True)


if __name__ == '__main__': main()
