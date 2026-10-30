"""Read-only inspection of the installed PushT implementation and HDF5 alignment."""
import inspect
import json
import os
from pathlib import Path
import h5py
import numpy as np
import stable_worldmodel as swm


def main():
    home = Path(os.environ['STABLEWM_HOME'])
    path = home / 'datasets/pusht_expert_train.h5'
    dataset = swm.data.load_dataset(str(path), cache_dir=str(home), num_steps=7,
                                    frameskip=1, keys_to_cache=['state', 'action'])
    print('DATASET_CLASS', type(dataset), flush=True)
    for name in ('__getitem__', 'get_col_data'):
        print('DATASET_SOURCE', name)
        print(inspect.getsource(getattr(type(dataset), name)))
    package = Path(swm.__file__).resolve().parent
    # Print the installed environment source, without instantiating a renderer.
    for source in sorted(package.rglob('*.py')):
        if 'pusht' not in str(source.relative_to(package)).lower().replace('_', ''):
            continue
        lines = source.read_text(errors='replace').splitlines()
        if not any('class ' in line for line in lines):
            continue
        print('PUSHT_SOURCE_FILE', source)
        if len(lines) <= 650:
            for i, line in enumerate(lines, 1):
                print(f'{i}: {line}')
        else:
            print('SOURCE_TOO_LONG', len(lines), 'showing relevant contexts')
            selected = set()
            for i, line in enumerate(lines):
                if any(term in line for term in ('def step', 'def _get_obs', 'def get_state', 'def _get_state', 'def _set_state', 'action_type', 'action_space')):
                    selected.update(range(max(0, i-2), min(len(lines), i+65)))
            for i in sorted(selected):
                print(f'{i+1}: {lines[i]}')
    # Match the dataset loader's SWMR read mode for simultaneous handles.
    with h5py.File(path, 'r', swmr=True) as f:
        print('H5_ATTRIBUTES', {k: str(v) for k, v in f.attrs.items()})
        for key in ('state', 'action', 'proprio'):
            print('FIELD', key, 'SHAPE', f[key].shape, 'ATTRIBUTES', {k: str(v) for k, v in f[key].attrs.items()})
        episodes = f['episode_idx'][:].reshape(-1)
        steps = f['step_idx'][:].reshape(-1)
        valid = np.arange(min(len(dataset), len(episodes)-6))
        valid = valid[(episodes[valid] == episodes[valid+6]) & (steps[valid+6]-steps[valid] == 6)]
        rows = np.sort(np.random.default_rng(73).choice(valid, min(64, len(valid)), replace=False))
        if len(rows) < 5:
            raise ValueError('Insufficient windows')
        for n, row in enumerate(rows):
            item = dataset[int(row)]
            for key in ('state', 'action'):
                actual = np.asarray(item[key])
                expected = f[key][int(row):int(row)+7]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0,
                    err_msg=f'Loader index is not raw H5 row: {key}, row={row}')
            if n < 3:
                np.testing.assert_array_equal(np.asarray(item['pixels']), f['pixels'][int(row):int(row)+7],
                    err_msg='Pixel window differs from raw H5')
        print('RAW_ROW_AND_WINDOW_ALIGNMENT_VERIFIED', len(rows), flush=True)
        # Descriptive diagnostics only: no inferred physical labels or causal claims.
        chosen = np.sort(np.random.default_rng(74).choice(valid, min(4096, len(valid)), replace=False))
        state = f['state'][chosen].astype(float)
        following = f['state'][chosen+1].astype(float)
        action = f['action'][chosen].astype(float)
        summary = {'samples': len(chosen), 'state_min': state.min(0).tolist(),
                   'state_max': state.max(0).tolist(), 'action_min': action.min(0).tolist(),
                   'action_max': action.max(0).tolist()}
        if action.shape[1] == 2 and state.shape[1] >= 2:
            summary['hypotheses_mse'] = {
                'action_equals_next_first_two_state_columns': float(np.mean((action-following[:, :2])**2)),
                'action_equals_first_two_state_increments': float(np.mean((action-(following[:, :2]-state[:, :2]))**2)),
                'action_equals_current_first_two_state_columns': float(np.mean((action-state[:, :2])**2))}
        print('NUMERICAL_DIAGNOSTICS', json.dumps(summary))
    print('SCHEMA_INSPECTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
