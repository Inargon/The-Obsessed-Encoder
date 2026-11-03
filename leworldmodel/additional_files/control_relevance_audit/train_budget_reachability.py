"""Train a recorded-state diagnostic before committing to encoder training.

Independent goals prevent hindsight selection from defining the target policy.
Reports are conditional on this goal sampler and observed follow-up; censoring
may be informative. This is NOT optimal reachability, official PushT SR, or a
claim of novelty. Split membership is fixed before all window/goal sampling.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from budget_data import episode_runs, build_split
from budget_controls import select_features, evaluation_masks


def main():
    import h5py
    import torch
    from torch import nn
    import torch.nn.functional as F
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=12000)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--model-seed', type=int)
    p.add_argument('--feature-mode', choices=['geometry','current','history'], default='history')
    p.add_argument('--reference', type=Path, help='Original run; verify identical rows/labels/masks before training')
    args = p.parse_args()
    if args.samples < 100 or args.epochs < 1:
        raise ValueError('Need >=100 samples and >=1 epoch')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed if args.model_seed is None else args.model_seed)
    budgets = [0, 5, 10, 20, 40, 80]
    with h5py.File(args.dataset, 'r') as f:
        states = f['state'][:]
        ids = f['episode_idx'][:].reshape(-1)
        runs = episode_runs(ids, f['step_idx'][:])
    if states.shape != (len(ids), 7) or not np.isfinite(states).all():
        raise ValueError('Expected finite verified seven-column PushT states')
    if len(runs) < 10:
        raise ValueError('Need >=10 episodes')
    order = np.random.default_rng(args.seed).permutation(len(runs))
    partitions = np.split(order, [int(.6*len(order)), int(.8*len(order))])
    arrays = [build_split(states, [runs[i] for i in part],
                         args.samples if k == 0 else args.samples//3,
                         args.seed+100+k, budgets) for k, part in enumerate(partitions)]
    if args.reference:
        for name, (_, y, m, rows) in zip(('train','validation','test'), arrays):
            with np.load(args.reference/(name+'.npz')) as previous:
                for key, value in [('rows',rows),('labels',y),('known',m)]:
                    np.testing.assert_array_equal(value,previous[key],err_msg='Reference mismatch: '+name+'/'+key)
        print('BUDGET_REFERENCE_MATCHED',str(args.reference),flush=True)
    arrays = [(select_features(x,args.feature_mode),y,m,rows) for x,y,m,rows in arrays]
    end_rows = np.empty(len(ids),dtype=np.int64)
    for a,b in runs:
        end_rows[a:b] = b-1
    mean, std = arrays[0][0].mean(0), arrays[0][0].std(0)
    std[std < 1e-6] = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensors = [tuple(torch.as_tensor(v, dtype=torch.float32, device=device)
                     for v in ((x-mean)/std, y, m)) for x, y, m, _ in arrays]
    model = nn.Sequential(nn.Linear(len(mean), 128), nn.SiLU(),
                          nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, len(budgets))).to(device)
    # Low initial cumulative probability helps with rare independently sampled hits.
    nn.init.constant_(model[-1].bias, -4.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def predict(x):
        return -torch.expm1(torch.cumsum(F.logsigmoid(-model(x)), dim=1))

    # Numerical equivalence to the tested NumPy reference and actual gradient check.
    from budget_reachability import hazard_to_cdf
    probe = torch.tensor([[-1000., 0., 1000.], [0., 0., 0.]], device=device)
    np.testing.assert_allclose((-torch.expm1(F.logsigmoid(-probe).cumsum(1))).cpu(),
                               hazard_to_cdf(probe.cpu().numpy()), atol=1e-6)
    best, selected = float('inf'), None
    tx, ty, tm = tensors[0]
    vx, vy, vm = tensors[1]
    with (out/'curve.jsonl').open('w') as curve:
        for epoch in range(1, args.epochs+1):
            model.train()
            for batch in torch.randperm(len(tx), device=device).split(512):
                pred = predict(tx[batch]).clamp(1e-6, 1-1e-6)
                loss = (F.binary_cross_entropy(pred, ty[batch], reduction='none')*tm[batch]).sum()/tm[batch].sum().clamp_min(1)
                optimizer.zero_grad(); loss.backward()
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('Nonfinite loss/gradient')
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val = float((((predict(vx)-vy)**2)*vm).sum()/vm.sum())
            curve.write(json.dumps({'epoch': epoch, 'validation_brier': val})+'\n'); curve.flush()
            if val < best:
                best, selected = val, epoch
                torch.save({'model': model.state_dict(), 'mean': mean.tolist(),
                            'std': std.tolist(), 'budgets': budgets}, out/'best.pt')
            if epoch % 20 == 0:
                print('BUDGET_EPOCH', epoch, 'VAL_BRIER', val, flush=True)
    model.load_state_dict(torch.load(out/'best.pt', map_location=device, weights_only=True)['model'])
    train_y, train_m = arrays[0][1:3]
    prior = (train_y*train_m).sum(0)/np.maximum(train_m.sum(0), 1)

    def metrics(pred, y, mask):
        result = []
        for j, h in enumerate(budgets):
            valid = mask[:, j].astype(bool)
            a, b = pred[valid, j], y[valid, j]
            result.append({'budget': h, 'known': int(valid.sum()), 'positives': int(b.sum()),
                'brier': float(np.mean((a-b)**2)) if len(b) else None,
                'mean_prediction': float(a.mean()) if len(b) else None,
                'positive_brier': float(np.mean((a[b]-1)**2)) if b.any() else None,
                'negative_brier': float(np.mean(a[~b]**2)) if (~b).any() else None})
        return result

    reports = {}
    for name, (x, y, mask), (_, yn, mn, provenance) in zip(('train','validation','test'), tensors, arrays):
        with torch.no_grad():
            pred = predict(x).cpu().numpy()
        if np.any(np.diff(pred, axis=1) < -1e-6):
            raise AssertionError('Nonmonotone CDF')
        reports[name] = {'model': metrics(pred, yn, mn),
                         'train_prior': metrics(np.broadcast_to(prior, pred.shape), yn, mn)}
        remaining = end_rows[provenance[:,0]]-provenance[:,0]
        subsets = evaluation_masks(yn,mn,remaining,max(budgets))
        reports[name]['subsets'] = {subset:{'model':metrics(pred,yn,mask),
            'train_prior':metrics(np.broadcast_to(prior,pred.shape),yn,mask)}
            for subset,mask in subsets.items()}
        np.savez_compressed(out/(name+'.npz'), rows=provenance, labels=yn, known=mn, predictions=pred)
    report = {'scope': __doc__, 'args': vars(args), 'selected_epoch': selected,
              'input': args.feature_mode,
              'ablation': '28 input slots and same MLP; excluded inputs zero padded; three model seeds do not constitute independent dataset splits',
              'censoring_note': 'Complete-followup subsets avoid outcome-dependent label availability but select longer remaining trajectories; no full-population calibration claim',
              'predicate': 'block translation <=10px AND wrapped angle <=0.15rad; diagnostic only',
              'episode_ids': [[int(ids[runs[i][0]]) for i in part] for part in partitions],
              'budgets_raw_steps': budgets, 'results': reports}
    report['args']['reference'] = str(args.reference) if args.reference else None
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for subset,values in reports['test']['subsets'].items():
        print('BUDGET_CONTROL',args.feature_mode,args.model_seed,subset,
              json.dumps(values['model']),flush=True)
    print('BUDGET_REACHABILITY_COMPLETE', out/'report.json', flush=True)


if __name__ == '__main__':
    main()
