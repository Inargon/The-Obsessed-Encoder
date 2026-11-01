"""Native matched PushT predictor audit: history=3, frameskip=5, num_preds=1.

Protocol pinned to submit_pusht_clean_{jepa,ours}_matched.sh and train defaults.
Uses the training image/action transforms, checks native loader vs raw H5 rows,
and removes only auxiliary control-objective parameters when loading JEPA.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np


def verify_window(item, state, action, row, history=3, skip=5):
    frames = history + 1
    np.testing.assert_array_equal(np.asarray(item['state']), state[row:row+frames*skip:skip],
                                  err_msg='Native state/frame alignment differs')
    # Loader may expose packed or unpacked action blocks, but order must match.
    actual = np.asarray(item['action'])
    expected = action[row:row+frames*skip]
    if actual.size != expected.size:
        raise ValueError('Native action window length differs')
    np.testing.assert_array_equal(actual.reshape(expected.shape), expected,
                                  err_msg='Native action/block alignment differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=1024)
    args = p.parse_args()
    if args.samples < 32: p.error('Need at least 32 samples')
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    import stable_worldmodel as swm
    import stable_pretraining as spt
    from hydra.utils import instantiate
    from utils import get_column_normalizer, get_img_preprocessor
    from action_information import save_report

    home = Path(os.environ['STABLEWM_HOME'])
    source = home / 'datasets/pusht_expert_train.h5'
    dataset = swm.data.load_dataset(str(source), cache_dir=os.environ['LOCAL_DATASET_DIR'],
        num_steps=4, frameskip=5, keys_to_load=['pixels', 'action', 'state'],
        keys_to_cache=['state', 'action'])
    state = np.asarray(dataset.get_col_data('state'))
    action = np.asarray(dataset.get_col_data('action'))
    episodes = np.asarray(dataset.get_col_data('episode_idx')).reshape(-1)
    steps = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    chosen = np.sort(np.random.default_rng(73).choice(len(dataset), min(args.samples, len(dataset)), replace=False))
    rows, ids = [], []
    for clip in chosen:
        ep, start = map(int, dataset.clip_indices[int(clip)])
        row = int(dataset.offsets[ep]) + start
        if start < 0 or start + int(dataset.span) > int(dataset.lengths[ep]):
            raise ValueError('Window crosses episode boundary')
        if row + 20 > len(episodes) or not np.all(episodes[row:row+20] == episodes[row]):
            raise ValueError('Raw action block crosses episode boundary')
        np.testing.assert_array_equal(steps[row:row+20], steps[row]+np.arange(20))
        verify_window(dataset[int(clip)], state, action, row)
        rows.append(row); ids.append(int(episodes[row]))
    print('NATIVE_PREDICTOR_ALIGNMENT_VERIFIED', len(rows), flush=True)
    # Same full-dataset normalization as training, not fit on the probe split.
    # Probe input scaling itself is fit only on probe-training episodes.
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor('pixels', 'pixels', img_size=224),
        get_column_normalizer(dataset, 'action', 'action'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reference = None
    for label in ('jepa', 'ours'):
        run = 'pusht_clean_' + label + '_matched_full_seed0'
        root = home / 'checkpoints' / run
        checkpoint = root / 'weights_epoch_10.pt'
        config = json.loads((root / 'config.json').read_text())
        model = instantiate(config)
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict({k: v for k, v in weights.items() if not k.startswith('control_objective.')}, strict=True)
        model.to(device).eval().requires_grad_(False)
        torch.manual_seed(73)
        fields = {k: [] for k in ('history', 'past_actions', 'real', 'predicted', 'action')}
        with torch.inference_mode():
            for j, clip in enumerate(chosen):
                item = dataset[int(clip)]
                pixels = torch.as_tensor(item['pixels']).unsqueeze(0).to(device)
                actions = torch.as_tensor(item['action']).unsqueeze(0).to(device)
                if pixels.shape[:2] != (1, 4) or actions.shape != (1, 4, 10):
                    raise ValueError(f'Unexpected native layout: {pixels.shape}, {actions.shape}')
                if not torch.isfinite(actions).all(): raise ValueError('Nonfinite normalized actions')
                encoded = model.encode({'pixels': pixels, 'action': actions})
                emb = encoded['emb']
                predicted = model.predict(emb[:, :3], encoded['act_emb'][:, :3])
                if predicted.shape != emb[:, 1:].shape: raise ValueError('Training target shape mismatch')
                values = dict(history=emb[0, :3].flatten(), past_actions=actions[0, :2].flatten(),
                              real=emb[0, 3], predicted=predicted[0, -1], action=actions[0, 2])
                for key, value in values.items(): fields[key].append(value.float().cpu().numpy())
                if (j+1) % 64 == 0: print('EXPORTED_ACTION', label, j+1, flush=True)
        bundle = {k: np.asarray(v) for k, v in fields.items()}
        bundle.update(episode_id=np.asarray(ids), row_index=np.asarray(rows), clip_index=chosen)
        if reference is not None:
            for key in ('episode_id', 'row_index', 'clip_index', 'action', 'past_actions'):
                np.testing.assert_array_equal(bundle[key], reference[key])
        reference = bundle
        with (args.out / (label+'.npz')).open('xb') as f: np.savez_compressed(f, **bundle)
        digest = hashlib.sha256()
        with checkpoint.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''): digest.update(chunk)
        provenance = dict(checkpoint=str(checkpoint), sha256=digest.hexdigest(), dataset=str(source),
            history_size=3, frameskip=5, num_preds=1, image_size=224,
            target='last context action block, raw rows offset+10:offset+15; endpoint at offset+15',
            normalization='official full-dataset training normalizer; probes use train-only scaling',
            protocol_source='matched submission scripts and config/train/lewm.yaml, data/pusht.yaml',
            model_config=config, torch_version=torch.__version__, numpy_version=np.__version__)
        (args.out/(label+'-provenance.json')).write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        save_report(bundle, args.out/(label+'-report.json'))
        del model, weights
        if device.type == 'cuda': torch.cuda.empty_cache()
    print('ACTION_INFORMATION_AUDIT_COMPLETE', args.out, flush=True)


if __name__ == '__main__': main()
