"""Episode-disjoint frozen probes. NumPy only; no training/checkpoint mutation."""
import argparse
import json
from pathlib import Path
import numpy as np


def split_episodes(ids, seed):
    groups = np.unique(ids)
    if len(groups) < 5:
        raise ValueError('Need at least five complete episode IDs')
    groups = np.random.default_rng(seed).permutation(groups)
    a = min(max(1, int(.6*len(groups))),len(groups)-2)
    b = min(max(a+1, int(.8*len(groups))),len(groups)-1)
    return [np.flatnonzero(np.isin(ids,g)) for g in (groups[:a],groups[a:b],groups[b:])]


def score(y, pred):
    ss = np.sum((y-y.mean(axis=0))**2,axis=0)
    values = np.full(y.shape[1], np.nan)
    valid = ss > 1e-12
    values[valid] = 1-np.sum((y-pred)**2,axis=0)[valid]/ss[valid]
    return {'r2':float(np.mean(values[valid])) if valid.any() else None,
            'r2_per_dimension':[float(v) if np.isfinite(v) else None for v in values],
            'mse':float(np.mean((y-pred)**2))}


def fit_predict(x,y,train,test,alpha):
    mean=x[train].mean(0);std=x[train].std(0);std[std<1e-8]=1
    a=(x[train]-mean)/std;b=(x[test]-mean)/std
    ym=y[train].mean(0);target=y[train]-ym
    if a.shape[1] <= a.shape[0]:
        w=np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@target)
    else:
        w=a.T@np.linalg.solve(a@a.T+alpha*np.eye(len(a)),target)
    return b@w+ym


def probe(x,y,splits):
    train,val,test=splits
    alphas=(.01,.1,1.,10.,100.)
    losses=[np.mean((fit_predict(x,y,train,val,a)-y[val])**2) for a in alphas]
    alpha=alphas[int(np.argmin(losses))]
    # Test is untouched by alpha selection; fit on train only for clear protocol.
    return dict(score(y[test],fit_predict(x,y,train,test,alpha)),alpha=alpha)


def state_targets(s):
    theta=s[:,4];rel=s[:,:2]-s[:,2:4];c=np.cos(theta);q=np.sin(theta)
    return {'pusher_xy':s[:,:2], 'block_xy':s[:,2:4],
            'block_angle_sincos':np.stack([q,c],1),
            'pusher_in_block_frame':np.stack([c*rel[:,0]+q*rel[:,1],-q*rel[:,0]+c*rel[:,1]],1)}


def validate(data):
    needed=('features','state','previous_state','next_state','action','episode_id')
    for k in needed:
        if k not in data: raise ValueError('Missing '+k)
    n=len(data['episode_id'])
    if data['episode_id'].ndim != 1: raise ValueError('episode_id must be one-dimensional')
    for k in needed[:-1]:
        v=data[k]
        if v.ndim!=2 or len(v)!=n or not np.isfinite(v).all():
            raise ValueError(k+' must be a finite N by D numeric array')
    if data['state'].shape[1]<1 or any(data[k].shape != data['state'].shape for k in ('previous_state','next_state')):
        raise ValueError('All state windows must have equal nonempty shapes')
    if data['episode_id'].dtype.kind in 'fc' and not np.isfinite(data['episode_id']).all():
        raise ValueError('Nonfinite episode IDs')
    if len(np.unique(data['episode_id']))<5:raise ValueError('Need >=5 episodes')


def evaluate(data,seed=73):
    validate(data);splits=split_episodes(data['episode_id'],seed)
    s=data['state'];prev=data['previous_state'];future=data['next_state'];a=data['action']
    raw = str(data.get('state_schema', 'pusht_pose5' if s.shape[1]==5 else 'raw')) == 'raw'
    # Semantic targets are disabled for raw, unverified state columns.
    physical=s if raw else np.column_stack(list(state_targets(s).values()))
    velocity=s-prev;velocity=velocity.copy()
    if not raw: velocity[:,4]=np.arctan2(np.sin(velocity[:,4]),np.cos(velocity[:,4]))
    targets={f'raw_state_col_{i}':s[:,i:i+1] for i in range(s.shape[1])} if raw else state_targets(s)
    if raw:
        outcomes={f'raw_delta_col_{i}':future[:,i:i+1]-s[:,i:i+1] for i in range(s.shape[1])}
    else:
        delta_angle=np.arctan2(np.sin(future[:,4]-s[:,4]),np.cos(future[:,4]-s[:,4]))
        outcomes={'block_delta_xy':future[:,2:4]-s[:,2:4], 'block_delta_angle':delta_angle[:,None]}
    result={'protocol':{'split':'60/20/20 disjoint episodes; alpha chosen on validation only',
        'seed':seed,'counts':[len(x) for x in splits],
        'episodes':[[str(x) for x in np.unique(data['episode_id'][idx])] for idx in splits],
        'interpretation':'Observational linear probes, not causal identification, forward-model evaluation or closed-loop SR',
        'history':'Features should encode previous/current images; see extraction provenance',
        'state_schema':'raw columns; angular deltas not wrapped; semantic interpretation pending' if raw else 'pusht_pose5',
        'geometry':'No contact labels. Raw mode makes no coordinate/velocity assumptions',
        'oracle':'Recorded state history only; not asserted complete simulator state'},
        'state_probes':{k:probe(data['features'],v,splits) for k,v in targets.items()},
        'outcome_probes':{}}
    inputs={'action_only':a, 'features_and_action':np.column_stack([data['features'],a]),
            ('recorded_state_history_and_action' if raw else 'pose_history_and_action'):np.column_stack([physical,velocity,a])}
    for key,y in outcomes.items():
        result['outcome_probes'][key]={k:probe(x,y,splits) for k,x in inputs.items()}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=73);args=p.parse_args()
    if args.out.exists():p.error('Refusing to overwrite existing results')
    with np.load(args.input,allow_pickle=False) as f:result=evaluate(dict(f),args.seed)
    result['input']=str(args.input.resolve())
    provenance=args.input.with_suffix('.json')
    if provenance.is_file():result['extraction_provenance']=json.loads(provenance.read_text(encoding='utf-8'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('Saved',args.out)

if __name__=='__main__': main()
