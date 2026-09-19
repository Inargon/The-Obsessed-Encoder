"""Frozen native-pixel tag interventions and fixed-candidate model costs.

No privileged states, optimization, or environment rollout. Candidate selection
is an offline diagnostic, not CEM replanning or a success-rate measurement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', required=True, help='LABEL=RUN/FILE.pt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset', default='pusht_expert_train.h5')
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--num-clips', type=int, default=128)
    p.add_argument('--candidates', type=int, default=32)
    p.add_argument('--history', type=int, default=3)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--frameskip', type=int, default=5)
    p.add_argument('--tag-size', type=int, default=5)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f'Refusing to overwrite {args.out}')
    if min(args.num_clips, args.candidates) < 2 or min(args.history,args.horizon,args.frameskip,args.tag_size)<1:
        p.error('Need >=2 clips/candidates and positive temporal/tag dimensions')

    import numpy as np
    import torch
    import stable_worldmodel as swm
    from additional_files.evaluate_reacher_checkpoint import load_inference_model
    from additional_files.diagnose_frozen_representation import parse_checkpoint
    from additional_files.mechanism_metrics import summarize, cost_changes
    from utils import get_img_preprocessor, get_column_normalizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dataset = swm.data.load_dataset(args.dataset, cache_dir=os.environ['LOCAL_DATASET_DIR'],
        num_steps=args.history+args.horizon, frameskip=args.frameskip, keys_to_cache=['action'])
    dataset.transform = None
    normalizer = get_column_normalizer(dataset, 'action', 'action')
    preprocess = get_img_preprocessor('pixels','pixels',img_size=224)
    ids = rng.choice(len(dataset), min(args.num_clips,len(dataset)), replace=False)
    if len(ids)<2:
        raise ValueError('Dataset must contain >=2 clips')
    pixels, actions = [], []
    for i in ids:
        sample = dataset[int(i)]
        raw = torch.as_tensor(sample['pixels']).clone()
        if raw.dtype != torch.uint8 or raw.ndim != 4 or raw.shape[1]!=3:
            raise ValueError(f'Expected native uint8 TCHW input, got {raw.dtype}/{raw.shape}')
        if raw.shape[0] != args.history+args.horizon or args.tag_size>min(raw.shape[-2:]):
            raise ValueError('Unexpected clip length or tag exceeds native image size')
        pixels.append(raw)
        act = torch.as_tensor(sample['action']).clone()
        if not torch.isfinite(act).all():
            raise ValueError('Nonfinite action clip; verify dataset boundary handling')
        act = normalizer({'action':act})['action']
        actions.append(act.reshape(args.history+args.horizon, -1))
    actions = torch.stack(actions)
    colors = torch.as_tensor(rng.integers(0,256,(len(ids),2,3)),dtype=torch.uint8)
    bank_ids = rng.integers(len(ids), size=(len(ids),args.candidates))
    partners = np.roll(np.arange(len(ids)),1)
    root = Path(os.environ['STABLEWM_HOME'])/'checkpoints'
    result = {'protocol':dict(vars(args), out=str(args.out),
        diagnostic='offline fixed candidate costs; NOT closed-loop SR',
        sampling='random dataset clips; not guaranteed training-held-out or episode-independent',
        bootstrap='clip resampling conditional on frozen checkpoint; not training-seed uncertainty',
        tag_stage='native uint8 before exact training image preprocessing',
        action_units='dataset z-score normalized; candidates from observed action blocks',
        indices=ids.tolist(), candidate_source_indices=ids[bank_ids].tolist(),
        colors=colors.tolist(), device=str(device)), 'runs':{}}

    def image(raw, color):
        tagged=raw.clone()
        tagged[:,:,:args.tag_size,:args.tag_size]=color.view(1,3,1,1)
        return preprocess({'pixels':tagged})['pixels'].to(device)

    with torch.no_grad():
        for spec in args.checkpoint:
            label, run, filename = parse_checkpoint(spec)
            if label in result['runs']:
                raise ValueError(f'Duplicate label: {label}')
            checkpoint=root/run/filename
            digest=hashlib.sha256()
            with checkpoint.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1048576),b''):
                    digest.update(chunk)
            model=load_inference_model(run,filename).to(device).eval().requires_grad_(False)
            metrics={}
            def add(key, value):
                metrics.setdefault(key,[]).append(float(value))
            for i, raw in enumerate(pixels):
                own=image(raw,colors[i,0]); changed=image(raw,colors[i,1])
                other=image(pixels[int(partners[i])],colors[i,0])
                frames=torch.stack([own[0],changed[0],other[0]])
                backbone=model.encoder(frames,interpolate_pos_encoding=True).last_hidden_state[:,0].float()
                for rep, emb in [('backbone',backbone),('projection',model.projector(backbone))]:
                    tag_dist=(emb[0]-emb[1]).square().mean().item()
                    content_dist=(emb[0]-emb[2]).square().mean().item()
                    add(rep+'/tag_intervention_mse',tag_dist)
                    add(rep+'/content_intervention_mse',content_dist)

                # Same history actions, same future candidate bank for all tag conditions.
                plans=actions[bank_ids[i],:args.history+args.horizon-1].clone()
                plans[:,:args.history-1]=actions[i,:args.history-1]
                plans=plans.unsqueeze(0).to(device)
                costs={}
                for condition, ctx, goal in [('reference',own,own),('both_tags',changed,changed),
                                             ('context_tag',changed,own),('goal_tag',own,changed)]:
                    count=args.candidates
                    info={'pixels':ctx[:args.history][None,None].expand(1,count,-1,-1,-1,-1),
                          'goal':goal[-1:][None,None].expand(1,count,-1,-1,-1,-1),
                          'action':plans}
                    costs[condition]=model.get_cost(info,plans.clone()).cpu().numpy()
                first=plans[:,:,args.history-1].cpu().numpy()
                for condition in ('both_tags','context_tag','goal_tag'):
                    for key, value in cost_changes(costs['reference'],costs[condition],first).items():
                        add(condition+'/'+key,value[0])
                if (i+1)%16==0:
                    print(f'{label}: {i+1}/{len(ids)}',flush=True)
            result['runs'][label]={'run_name':run,'checkpoint':filename,'sha256':digest.hexdigest(),
                'metrics':{k:summarize(v,args.seed) for k,v in metrics.items()}, 'per_clip':metrics}
            del model
            if device.type=='cuda': torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(f'TAG_INTERVENTION_COMPLETE {args.out}',flush=True)


if __name__=='__main__':
    main()
