"""Recorded-state terminal-pose diagnostic; goals enter only analytic scoring."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from audit import split_episodes
from multisplit_readout import fit_readout
from readout_delayed_pairs import choice_metrics


def encode_pose(pose):
    return np.concatenate([pose[...,:2],np.sin(pose[...,2:3]),np.cos(pose[...,2:3])],axis=-1)


def terminal_cost(encoded,goals):
    angle=np.arctan2(encoded[...,2],encoded[...,3])
    delta=np.arctan2(np.sin(angle-goals[:,None,2]),np.cos(angle-goals[:,None,2]))
    return np.sum((encoded[...,:2]-goals[:,None,:2])**2,axis=-1)+(30*delta)**2


def fit_mlp(x,y,splits):
    import torch
    tr,va,te=splits
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm=x[tr].mean(0);xs=x[tr].std(0);xs[xs<1e-8]=1
    ym=y[tr].mean(0);ys=y[tr].std(0);ys[ys<1e-8]=1
    xx=torch.as_tensor((x-xm)/xs,dtype=torch.float32,device=device)
    yy=torch.as_tensor((y-ym)/ys,dtype=torch.float32,device=device)
    best=float('inf');chosen=None;configuration=None
    # Same training seed for each configuration; validation selects width, penalty and epoch.
    for width in (64,128):
        for decay in (0.,.001):
            torch.manual_seed(901)
            model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],width),torch.nn.SiLU(),
                torch.nn.Linear(width,width),torch.nn.SiLU(),torch.nn.Linear(width,y.shape[1])).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=decay)
            for epoch in range(1,601):
                model.train();optimizer.zero_grad(set_to_none=True)
                loss=torch.mean((model(xx[tr])-yy[tr])**2)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();optimizer.step()
                if epoch%10==0:
                    model.eval()
                    with torch.no_grad():value=float(torch.mean((model(xx[va])-yy[va])**2))
                    if value<best:
                        best=value
                        chosen=(width,copy.deepcopy(model.state_dict()))
                        configuration={'width':width,'weight_decay':decay,'epoch':epoch,'validation_normalized_mse':value}
    width,weights=chosen
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],width),torch.nn.SiLU(),
        torch.nn.Linear(width,width),torch.nn.SiLU(),torch.nn.Linear(width,y.shape[1])).to(device)
    model.load_state_dict(weights);model.eval()
    with torch.no_grad():pred=model(xx[te]).cpu().numpy()*ys+ym
    return configuration,pred


def aligned_visual_inputs(bundle, initial, goals, ids):
    """Check cached row identity and strip only the explicitly appended goal suffix."""
    expected=np.column_stack([goals[:,:2],np.sin(goals[:,2]),np.cos(goals[:,2])])
    np.testing.assert_array_equal(bundle['pair_ids'],ids,err_msg='Pair IDs do not match fixture')
    np.testing.assert_array_equal(bundle['recorded_state'],np.column_stack([initial,expected]),
                                  err_msg='Cached state/goal rows do not match fixture')
    result={}
    for name in ('jepa','ours'):
        x=bundle[name]
        if x.ndim!=2 or x.shape[0]!=len(ids) or x.shape[1]<=4 or not np.isfinite(x).all():
            raise ValueError('Invalid cached visual features')
        np.testing.assert_array_equal(x[:,-4:],expected,err_msg='Unexpected explicit goal suffix')
        result[name]=x[:,:-4].copy()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--features-dir',type=Path)
    args=p.parse_args()
    if args.out.exists():p.error('Refusing to overwrite report')
    with np.load(args.input_dir/'branches.npz',allow_pickle=False) as f:
        initial=f['initial_states'];final=f['states'][:,:,:,-1,2:5]
        goals=f['goals'];costs=f['candidate_costs'];eligible=f['delayed_mask']&f['candidate_choice_diff']
    n=len(initial);actions=final.shape[2]
    initial=initial.reshape(2*n,7);goals=np.repeat(goals,2,axis=0)
    final=final.reshape(2*n,actions,3);costs=costs.reshape(2*n,actions)
    target=encode_pose(final).reshape(2*n,-1)
    # Periodic input angle; do not supply goal to the dynamics predictor.
    x=np.column_stack([initial[:,:4],np.sin(initial[:,4]),np.cos(initial[:,4]),initial[:,5:]])
    ids=np.repeat(np.arange(n),2)
    np.testing.assert_allclose(terminal_cost(encode_pose(final),goals),costs,rtol=1e-10,atol=1e-7,
        err_msg='Analytic scoring does not reproduce stored true costs')
    if not np.isfinite(x).all() or not np.isfinite(target).all():raise ValueError('Nonfinite input')
    inputs={'recorded_state':x}
    feature_provenance=None
    if args.features_dir is not None:
        path=args.features_dir/'features.npz'
        meta=json.loads((args.features_dir/'report.json').read_text())
        digest=hashlib.sha256((args.input_dir/'branches.npz').read_bytes()).hexdigest()
        if meta.get('input_sha256')!=digest:raise ValueError('Cached features originate from a different fixture')
        with np.load(path,allow_pickle=False) as f:bundle=dict(f)
        inputs.update(aligned_visual_inputs(bundle,initial,goals,ids))
        feature_provenance={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'fixture_sha256':digest,'checkpoints':meta['checkpoints']}
        print('CACHED_VISUAL_ROWS_AND_GOAL_SUFFIX_VERIFIED',len(ids),flush=True)
    stationary=np.repeat(encode_pose(initial[:,2:5])[:,None,:],actions,axis=1)
    report={'protocol':{'primary_split':73,'sensitivity_splits':[74,75],
        'target':'terminal block x,y,sin(angle),cos(angle), separately for each fixed candidate sequence',
        'input':'recorded state or frozen single-frame features; explicit goal used only after prediction',
        'visual_caveat':'Initial rendered image contains a goal outline; removing appended goal coordinates does not remove visual goal information',
        'capacity':'Same hidden widths/search budget; input dimensions and parameter counts differ',
        'selection':'train-only scaling/fitting; validation-only probe and epoch selection',
        'mlp_training_seed':901,'mlp_widths':[64,128],'weight_decay':[0,.001],'epochs':600,
        'limitations':['Recorded seven-state is not asserted to be a complete simulator state',
            'Recorded block velocities are zero under fixture reset; goal-independent dynamics assumed by this diagnostic',
            'Same constructed distribution, three overlapping pair-group splits, not independent replications',
            'Candidate selection only; no new encoder training or closed-loop planning',
            'Improvement over baselines is not guaranteed; retain negative results']},'feature_provenance':feature_provenance,'runs':[]}
    for seed in (73,74,75):
        splits=split_episodes(ids,seed);tr,va,te=splits
        run={'seed':seed,'test_pairs':np.unique(ids[te]).tolist(),'models':{}}
        baseline=np.broadcast_to(costs[tr].mean(0),costs[te].shape)
        for name,predicted in [('train_mean_cost',baseline),('stationary_block',terminal_cost(stationary[te],goals[te]))]:
            metrics=choice_metrics(costs[te],predicted,ids[te],eligible)
            run['models'][name]=metrics
            print('TERMINAL_BASELINE',seed,name,json.dumps({k:v for k,v in metrics.items() if k!='pairs'}),flush=True)
        for name,x in inputs.items():
            for kind in ('linear','rff','mlp'):
                if kind=='mlp':config,pred=fit_mlp(x,target,splits)
                else:config,pred=fit_readout(x,target,splits,kind,width=256)
                encoded=pred.reshape(len(te),actions,4)
                inferred=terminal_cost(encoded,goals[te])
                metrics=choice_metrics(costs[te],inferred,ids[te],eligible)
                xy=encoded[...,:2]-final[te,...,:2]
                pa=np.arctan2(encoded[...,2],encoded[...,3]);da=pa-final[te,...,2]
                da=np.arctan2(np.sin(da),np.cos(da))
                errors={'xy_rmse_px':float(np.sqrt(np.mean(xy**2))),
                    'angle_mae_rad':float(np.mean(np.abs(da))),
                    'angle_vector_mean_norm':float(np.linalg.norm(encoded[...,2:4],axis=-1).mean())}
                run['models'][name+'_'+kind]={'fit':config,'metrics':metrics,'errors':errors}
                print('TERMINAL_READOUT',seed,name,kind,json.dumps({**{k:v for k,v in metrics.items() if k!='pairs'},**errors}),flush=True)
        report['runs'].append(run)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('STATE_TERMINAL_AUDIT_COMPLETE',args.out,flush=True)


if __name__=='__main__':main()
