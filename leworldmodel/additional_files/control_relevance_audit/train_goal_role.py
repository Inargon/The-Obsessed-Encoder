"""Matched frozen-feature heads: original targets versus target-only augmentation.

Same examples, optimization steps, initialization and architecture per seed.
Original-target validation Brier selects checkpoints for BOTH arms. Test query
episodes and vertical goal-agent shifts are never used for optimization.
This is a candidate-bank first-hit diagnostic, not MPC or official success rate.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from goal_role_data import group_split, evaluate_predictions


def main():
    import torch
    from torch import nn
    import torch.nn.functional as F
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=200)
    args = p.parse_args()
    if args.epochs < 1:
        p.error('epochs must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.features) as d:
        hist, actions, goals, labels, initial_hit, groups = [d[k].copy() for k in
            ('history','actions','goals','labels','initial_hit','episode')]
    q,g,v,e = goals.shape
    c = actions.shape[1]; h = labels.shape[-1]
    if v != 5 or labels.shape != (q,g,c,h) or len(np.unique(groups)) != q:
        raise ValueError('Unexpected feature layout or repeated source episodes')
    if not all(np.isfinite(a).all() for a in (hist,actions,goals)):
        raise ValueError('Nonfinite features')
    train,val,test = group_split(groups)
    common = np.concatenate([np.broadcast_to(hist[:,None],(q,c,hist.shape[-1])),
                              actions.reshape(q,c,-1)],axis=-1)
    xx = np.concatenate([np.broadcast_to(common[:,None,None],(q,g,v,c,common.shape[-1])),
                         np.broadcast_to(goals[:,:,:,None],(q,g,v,c,e))],axis=-1).astype(np.float32)
    # Both arms receive identical train-only normalization including allowed augmentations.
    fit = xx[train,:,:3].reshape(-1,xx.shape[-1])
    mean, std = fit.mean(0), fit.std(0); std[std<1e-6] = 1
    xx = (xx-mean)/std
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.as_tensor(xx,device=device)
    yy = torch.as_tensor(labels.astype(np.float32),device=device)
    n = len(train)*g*c
    # Order samples as query, goal, candidate; variant is a separate axis.
    tx = x[train,:,:3].permute(0,1,3,2,4).reshape(n,3,-1)
    ty = yy[train].reshape(n,h)
    vx, vy = x[val,:,0].reshape(-1,x.shape[-1]), yy[val].reshape(-1,h)
    reports = []
    for seed in (901,902,903):
        for mode in ('original','goal_augmented'):
            torch.manual_seed(seed)
            model = nn.Sequential(nn.Linear(x.shape[-1],128),nn.SiLU(),nn.Linear(128,128),
                                  nn.SiLU(),nn.Linear(128,h)).to(device)
            nn.init.constant_(model[-1].bias,-3.)
            opt = torch.optim.AdamW(model.parameters(),lr=1e-3)
            def predict(a):
                return -torch.expm1(F.logsigmoid(-model(a)).cumsum(-1))
            order_rng = np.random.default_rng(seed)
            variant_rng = np.random.default_rng(seed+10000)
            best, best_state, selected = float('inf'), None, None
            with (args.output/f'{mode}-{seed}-curve.jsonl').open('w') as curve:
                for epoch in range(1,args.epochs+1):
                    model.train()
                    for batch in np.array_split(order_rng.permutation(n), max(1,int(np.ceil(n/128)))):
                        index = torch.as_tensor(batch,device=device)
                        variants = (variant_rng.integers(0,3,len(batch)) if mode == 'goal_augmented'
                                    else np.zeros(len(batch),dtype=int))
                        pred = predict(tx[index,torch.as_tensor(variants,device=device)]).clamp(1e-6,1-1e-6)
                        loss = F.binary_cross_entropy(pred,ty[index])
                        opt.zero_grad(); loss.backward()
                        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                            raise RuntimeError('Nonfinite training loss/gradient')
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        vb = float(((predict(vx)-vy)**2).mean())
                    curve.write(json.dumps({'epoch':epoch,'validation_original_brier':vb})+'\n')
                    if vb < best:
                        best, selected = vb, epoch
                        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                curve.flush()
            model.load_state_dict(best_state)
            with torch.no_grad():
                pred = predict(x[test].reshape(-1,x.shape[-1])).reshape(len(test),g,v,c,h).cpu().numpy()
            result = {'seed':seed,'mode':mode,'selected_epoch':selected,'validation_brier':best,
                      'test':evaluate_predictions(pred,labels[test],initial_hit[test]),
                      'test_by_goal_source': [evaluate_predictions(pred[:,gi:gi+1],
                          labels[test,gi:gi+1],initial_hit[test,gi:gi+1]) for gi in range(g)]}
            reports.append(result)
            np.savez_compressed(args.output/f'{mode}-{seed}-test.npz',predictions=pred,
                                labels=labels[test],episodes=groups[test],initial_hit=initial_hit[test])
            torch.save({'model':best_state,'mean':mean.tolist(),'std':std.tolist()},
                       args.output/f'{mode}-{seed}.pt')
            print('GOAL_ROLE_RESULT',json.dumps({'seed':seed,'mode':mode,
                'selected_epoch':selected,'validation_brier':best,
                'original':result['test'][0],
                'unseen_up':result['test'][3],'unseen_down':result['test'][4]}),flush=True)
    prior = labels[train].mean((0,1,2))
    report = {'scope':__doc__,'feature_file':str(args.features.resolve()),'budgets':[5,10,20,40],
        'episodes':{k:groups[idx].tolist() for k,idx in zip(('train','validation','test'),(train,val,test))},
        'goal_sources':['branch1_endpoint','branch2_endpoint','recorded_future60'],
        'baseline_train_prior':evaluate_predictions(np.broadcast_to(prior,(len(test),g,v,c,h)),labels[test],initial_hit[test]),
        'runs':reports}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('GOAL_ROLE_TRAIN_COMPLETE',args.output/'report.json',flush=True)


if __name__ == '__main__':
    main()
