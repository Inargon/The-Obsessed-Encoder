"""Single-frame frozen readouts for grouped delayed-interaction fixtures."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from audit import split_episodes
from multisplit_readout import fit_readout


def choice_metrics(costs, predicted, pair_ids, eligible):
    if costs.shape!=predicted.shape:raise ValueError('Cost prediction shape mismatch')
    selected=np.argmin(predicted,axis=1);truth=np.argmin(costs,axis=1)
    regret=costs[np.arange(len(costs)),selected]-costs.min(1)
    correct=selected==truth
    pairs=[]
    for pair in np.unique(pair_ids):
        idx=np.flatnonzero(pair_ids==pair)
        if len(idx)!=2:raise ValueError('Both scenes of a pair must be held out together')
        pairs.append({'pair':int(pair),'eligible':bool(eligible[pair]),
            'truth':truth[idx].tolist(),'selected':selected[idx].tolist(),
            'both_correct':bool(correct[idx].all()),'regret':regret[idx].tolist()})
    subset=[p for p in pairs if p['eligible']]
    return {'scene_accuracy':float(correct.mean()),'mean_candidate_regret':float(regret.mean()),
        'pair_both_correct_fraction':float(np.mean([p['both_correct'] for p in pairs])),
        'eligible_test_pairs':len(subset),
        'eligible_pair_both_correct_fraction':float(np.mean([p['both_correct'] for p in subset])) if subset else None,
        'pairs':pairs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('Refusing to overwrite output directory')
    path=args.input_dir/'branches.npz'
    with np.load(path,allow_pickle=False) as f:d=dict(f)
    n=len(d['initial_states']);costs=d['candidate_costs'].reshape(n*2,-1).astype(float)
    pixels=d['initial_pixels']
    if pixels.ndim!=5 or pixels.shape[:2]!=(n,2) or pixels.shape[-1]!=3:
        raise ValueError('Expected pair,scene,H,W,RGB images')
    pair_ids=np.repeat(np.arange(n),2)
    if n<10:raise ValueError('Too few independent pairs even for smoke evaluation')
    eligible=d['delayed_mask']&d['candidate_choice_diff']
    target=costs-costs.mean(1,keepdims=True)
    args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    import torch
    from hydra.utils import instantiate
    from utils import get_img_preprocessor
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess=get_img_preprocessor('pixels','pixels',img_size=224)
    home=Path(os.environ['STABLEWM_HOME'])
    features={};provenance={}
    runs={'jepa':'pusht_clean_jepa_matched_full_seed0','ours':'pusht_clean_ours_matched_full_seed0'}
    for name,run in runs.items():
        checkpoint=home/'checkpoints'/run/'weights_epoch_10.pt'
        model=instantiate(json.loads((checkpoint.parent/'config.json').read_text()))
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict({k:v for k,v in weights.items() if not k.startswith('control_objective.')},strict=True)
        model=model.to(device).eval().requires_grad_(False)
        encoded=[]
        with torch.no_grad():
            for image in pixels.reshape(-1,*pixels.shape[2:]):
                batch=torch.from_numpy(image.copy()).permute(2,0,1).unsqueeze(0)
                batch=preprocess({'pixels':batch})['pixels'].to(device)
                z=model.encoder(batch,interpolate_pos_encoding=True).last_hidden_state[:,0].float()
                encoded.append(model.projector(z).flatten().float().cpu().numpy())
        features[name]=np.asarray(encoded)
        provenance[name]={'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          'feature_dim':int(features[name].shape[1])}
        del model
        print('ENCODED_SINGLE_FRAMES',name,len(encoded),flush=True)
    features['recorded_state']=d['initial_states'].reshape(n*2,-1)
    with (args.out/'features.npz').open('xb') as f:
        np.savez_compressed(f,**features,pair_ids=pair_ids)
    report={'protocol':{'pairs':n,'seed_splits':[73,74,75],
        'representation':'single initial image -> projected CLS; no duplicated or fabricated history',
        'target':'three centered terminal candidate costs; fixed goal from fixture',
        'fitting':'frozen encoder; train-only probe fit/scaling, validation hyperparameters',
        'limitations':['32-pair pipeline check; not a method comparison with adequate statistical power',
            'Candidate action selection is not closed-loop planning or a globally optimal policy',
            'Held-out pairs belong to the same narrowly constructed simulator distribution',
            'RFF is a limited nonlinear probe; failure does not prove information loss',
            'Eligible-pair mask uses true simulator outcomes for diagnostic reporting only, not training selection',
            'Three splits overlap and are not independent training seeds']},
        'input':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'checkpoints':provenance,'runs':[]}
    for seed in (73,74,75):
        splits=split_episodes(pair_ids,seed);tr,va,te=splits
        result={'seed':seed,'test_pairs':np.unique(pair_ids[te]).tolist(),'counts':[len(s) for s in splits], 'models':{}}
        baseline=np.broadcast_to(target[tr].mean(0),target[te].shape)
        result['models']['train_mean_baseline']=choice_metrics(costs[te],baseline,pair_ids[te],eligible)
        for name,x in features.items():
            result['models'][name]={}
            for kind in ('linear','rff'):
                fit,pred=fit_readout(x,target,splits,kind,width=128)
                metrics=choice_metrics(costs[te],pred,pair_ids[te],eligible)
                result['models'][name][kind]={'metrics':metrics,'fit':fit}
                summary={k:v for k,v in metrics.items() if k!='pairs'}
                print('PAIR_READOUT',seed,name,kind,json.dumps(summary),flush=True)
        print('PAIR_BASELINE',seed,json.dumps({k:v for k,v in result['models']['train_mean_baseline'].items() if k!='pairs'}),flush=True)
        report['runs'].append(result)
    with (args.out/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('PAIR_READOUT_COMPLETE',args.out,flush=True)


if __name__=='__main__':main()
