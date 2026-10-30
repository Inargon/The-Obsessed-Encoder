"""Matched multi-split linear and random-Fourier nonlinear frozen readouts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit import split_episodes, validate, score


def design_matrices(x, train, val, test, kind, width, bandwidth, seed):
    mean=x[train].mean(0);std=x[train].std(0);std[std<1e-8]=1
    parts=[(x[idx]-mean)/std for idx in (train,val,test)]
    if kind=='linear':return parts
    rng=np.random.default_rng(seed)
    w=rng.normal(size=(x.shape[1],width))/(np.sqrt(x.shape[1])*bandwidth)
    phase=rng.uniform(0,2*np.pi,size=width)
    mapped=[np.sqrt(2/width)*np.cos(v@w+phase) for v in parts]
    # A linear skip makes the nonlinear class include the linear baseline.
    return [np.column_stack([v,m]) for v,m in zip(parts,mapped)]


def fit_readout(x,y,splits,kind,width=256,feature_seed=901):
    tr,va,te=splits
    ym=y[tr].mean(0);ys=y[tr].std(0);ys[ys<1e-8]=1
    target=(y[tr]-ym)/ys
    best=np.full(y.shape[1],np.inf);pred=np.zeros_like(y[te],dtype=float)
    selected=[None]*y.shape[1]
    bandwidths=(1.,) if kind=='linear' else (.5,1.,2.)
    for bandwidth in bandwidths:
        a,b,c=design_matrices(x,tr,va,te,kind,width,bandwidth,feature_seed)
        # Fit intercept without penalizing it; all centering uses train only.
        center=a.mean(0);a=a-center;b=b-center;c=c-center
        primal=a.shape[1]<=len(tr)
        gram=a.T@a if primal else a@a.T
        rhs=a.T@target if primal else target
        for alpha in (.01,.1,1.,10.,100.):
            solution=np.linalg.solve(gram+alpha*np.eye(len(gram)),rhs)
            weights=solution if primal else a.T@solution
            vp=b@weights*ys+ym
            loss=np.mean(((vp-y[va])/ys)**2,axis=0)
            better=loss<best
            if better.any():
                tp=c@weights*ys+ym
                pred[:,better]=tp[:,better];best[better]=loss[better]
                for j in np.flatnonzero(better):
                    selected[j]={'alpha':alpha,'bandwidth':bandwidth if kind!='linear' else None}
    return {'per_column':[dict(score(y[te,j:j+1],pred[:,j:j+1]),**selected[j]) for j in range(y.shape[1])]},pred


def load_matched(root):
    data={}
    for name in ('jepa','ours'):
        path=root/(name+'.npz')
        with np.load(path,allow_pickle=False) as f:data[name]=dict(f)
        validate(data[name])
        if 'clip_index' not in data[name]:raise ValueError('Old uncorrected exports are not accepted')
    for key in ('clip_index','row_index','episode_id','state','previous_state','next_state','action'):
        np.testing.assert_array_equal(data['jepa'][key],data['ours'][key],err_msg='Unmatched '+key)
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[73,74,75]);p.add_argument('--width',type=int,default=256)
    args=p.parse_args()
    if args.out.exists():p.error('Refusing to overwrite report')
    if len(set(args.seeds))!=len(args.seeds) or args.width<1:p.error('Use distinct split seeds and positive width')
    data=load_matched(args.input_dir)
    report={'protocol':{'split_seeds':args.seeds,'training_checkpoints':'one existing training seed per model',
        'split':'60/20/20 episode-disjoint; same splits, targets and search budget for both models',
        'nonlinear':'Gaussian random Fourier features with linear skip; train-only scaling and centering',
        'width':args.width,'feature_seed':901,'alphas':[.01,.1,1,10,100], 'bandwidths':[.5,1,2],
        'selection':'Per-target validation normalized MSE; train-only fit; test excluded from selection',
        'limitations':['Raw columns, angular deltas unwrapped; angle scores are not physical rotation errors',
                       'Readouts are observational, not causal identification or closed-loop control',
                       'These are probe-held-out episodes, not confirmed encoder-training-held-out episodes',
                       'Repeated splits share data; spread is descriptive, not an independent confidence interval',
                       'Failure of finite random-feature probe does not prove information absent']},'inputs':{},'runs':[]}
    for name in data:
        path=args.input_dir/(name+'.npz')
        report['inputs'][name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        meta=path.with_suffix('.json')
        if meta.exists():report['inputs'][name]['extraction']=json.loads(meta.read_text())
    for seed in args.seeds:
        splits=split_episodes(data['jepa']['episode_id'],seed)
        run={'seed':seed,'counts':[len(i) for i in splits],
             'episodes':[[str(i) for i in np.unique(data['jepa']['episode_id'][s])] for s in splits], 'models':{}}
        # State/history and action-only inputs are identical, so fit their shared baselines once.
        common={}
        for name,d in data.items():
            tasks={'state':(d['state'],{'features':d['features']}),
                   'delta':(d['next_state']-d['state'],{
                       'action_only':d['action'],
                       'features_and_action':np.column_stack([d['features'],d['action']]),
                       'recorded_state_history_and_action':np.column_stack([d['state'],d['state']-d['previous_state'],d['action']])})}
            run['models'][name]={}
            for task,(y,inputs) in tasks.items():
                run['models'][name][task]={}
                for label,x in inputs.items():
                    metrics={}
                    for kind in ('linear','rff'):
                        key=(task,label,kind)
                        shared=label in ('action_only','recorded_state_history_and_action')
                        if shared and key in common:result=common[key]
                        else:
                            result,_=fit_readout(x,y,splits,kind,args.width)
                            if shared:common[key]=result
                        metrics[kind]=result
                        print('DONE',seed,name,task,label,kind,flush=True)
                    run['models'][name][task][label]=metrics
        report['runs'].append(run)
    for name in data:
        for task,label in [('state','features'),('delta','features_and_action')]:
            for kind in ('linear','rff'):
                values=np.array([[c['r2'] if c['r2'] is not None else np.nan for c in run['models'][name][task][label][kind]['per_column']] for run in report['runs']])
                print('SUMMARY',name,task,kind,'MEAN_R2',values.mean(0).tolist(),'SPLIT_SD',values.std(0).tolist(),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('REPORT',args.out,flush=True)
    print('MULTISPLIT_READOUT_COMPLETE',flush=True)


if __name__=='__main__':main()
