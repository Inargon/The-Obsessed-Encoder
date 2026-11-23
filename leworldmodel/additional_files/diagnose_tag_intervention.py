"""Frozen nuisance/relevance interventions and fixed-candidate model costs.

No privileged states, optimization, or environment rollout. Candidate selection
is an offline diagnostic, not CEM replanning or a success-rate measurement.
Tag changes are the nuisance intervention; scene and action-sequence changes
are positive controls for task-relevant sensitivity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def choose_finite_action_indices(dataset, order, requested):
    """Choose clips whose complete action window is finite.

    Sequence datasets retain rows near episode boundaries and represent the
    unavailable tail actions with NaNs.  Those rows are valid for some
    training losses after masking, but not for this fixed-candidate planning
    diagnostic, which compares complete action plans.
    """
    import torch

    selected = []
    for index in order:
        action = torch.as_tensor(dataset[int(index)]["action"])
        if torch.isfinite(action).all():
            selected.append(int(index))
            if len(selected) == requested:
                break
    if len(selected) < requested:
        raise ValueError(
            f"Requested {requested} finite-action clips but found "
            f"only {len(selected)}"
        )
    return selected


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
    p.add_argument('--candidate-audit', action='store_true',
                   help='Save matched costs and training-anchor gaps on disjoint episode sets')
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
    from additional_files.mechanism_metrics import summarize, cost_changes, normalize_action_blocks
    from additional_files.tag_candidate_audit import (
        CONDITIONS, candidate_gaps, frozen_anchor_gaps, sample_episode_bank,
    )
    from utils import get_img_preprocessor, get_column_normalizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dataset = swm.data.load_dataset(args.dataset, cache_dir=os.environ['LOCAL_DATASET_DIR'],
        num_steps=args.history+args.horizon, frameskip=args.frameskip, keys_to_cache=['action'])
    dataset.transform = None
    normalizer_source = dataset
    if args.candidate_audit and args.dataset != 'pusht_expert_train.h5':
        normalizer_source = swm.data.load_dataset(
            'pusht_expert_train.h5', cache_dir=os.environ['LOCAL_DATASET_DIR'],
            keys_to_load=['action'], keys_to_cache=['action'],
        )
    normalizer = get_column_normalizer(normalizer_source, 'action', 'action')
    native_action_dim = int(dataset.get_dim('action'))
    preprocess = get_img_preprocessor('pixels','pixels',img_size=224)
    if args.candidate_audit:
        query, bank = sample_episode_bank(
            dataset, rng, args.num_clips, args.candidates, args.history + args.horizon,
            args.frameskip, args.history,
        )
        ids = np.asarray([item['clip'] for item in query], dtype=int)
    else:
        requested = min(args.num_clips, len(dataset))
        ids = np.asarray(
            choose_finite_action_indices(dataset, rng.permutation(len(dataset)), requested),
            dtype=int,
        )
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
            raise AssertionError('finite-action sampler returned an invalid clip')
        act = normalize_action_blocks(act, normalizer, native_action_dim)
        actions.append(act.reshape(args.history+args.horizon, -1))
    actions = torch.stack(actions)
    colors = torch.as_tensor(rng.integers(0,256,(len(ids),2,3)),dtype=torch.uint8)
    if args.candidate_audit:
        for clip_id in np.nonzero((colors[:, 0] == colors[:, 1]).all(1).numpy())[0]:
            colors[clip_id, 1, 0] = (int(colors[clip_id, 1, 0]) + 1) % 256
        bank_actions = torch.stack([
            normalize_action_blocks(
                torch.as_tensor(dataset[item['clip']]['action']).clone(), normalizer, native_action_dim,
            ).reshape(args.history + args.horizon, -1)
            for item in bank
        ])
        futures = bank_actions[:, args.history - 1:-1].flatten(1)
        if not torch.isfinite(bank_actions).all() or len(torch.unique(futures, dim=0)) != args.candidates:
            raise ValueError('Normalized future plans must be finite and distinct')
        bank_ids = np.tile(np.arange(args.candidates), (len(ids), 1))
        source_ids = np.asarray([item['clip'] for item in bank])
    else:
        bank_actions = actions
        bank_ids = rng.integers(len(ids), size=(len(ids),args.candidates))
        source_ids = ids
    partners = np.roll(np.arange(len(ids)),1)
    root = Path(os.environ['STABLEWM_HOME'])/'checkpoints'
    result = {'protocol':dict(vars(args), out=str(args.out),
        diagnostic='offline fixed candidate costs; NOT closed-loop SR',
        sampling='random dataset clips; not guaranteed training-held-out or episode-independent',
        bootstrap='clip resampling conditional on frozen checkpoint; not training-seed uncertainty',
        tag_stage='native uint8 before exact training image preprocessing',
        action_units='dataset z-score normalized; candidates from observed action blocks',
        interventions={
            'tag':'change a 5x5 RGB tag while holding scene and candidates fixed',
            'scene':'replace context or goal frames with another sampled clip while holding tag and candidates fixed',
            'action_sequence':'cyclically shift future action blocks within each candidate while holding context and goal fixed',
        },
        indices=ids.tolist(), candidate_source_indices=source_ids[bank_ids].tolist(),
        colors=colors.tolist(), device=str(device)), 'runs':{}}
    if args.candidate_audit:
        result['protocol'].update(
            sampling='one complete clip per query episode, conditional on one fixed disjoint candidate bank',
            bootstrap='paired query-episode resampling conditional on frozen checkpoints and the fixed bank',
            episode_split={'query': query, 'candidate': bank},
            training_independent_episodes_verified=False,
            action_normalization_dataset='pusht_expert_train.h5',
            anchor='frozen one-step expert-vs-other-episode action pair, using the training gap layout',
            anchor_negative_source_indices=source_ids[np.arange(len(ids)) % args.candidates].tolist(),
            normalized_future_plans=futures.tolist(),
        )

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
            all_costs = {name: [] for name in CONDITIONS}
            all_anchors = {name: [] for name in CONDITIONS if name != 'action_sequence'}
            def add(key, value):
                metrics.setdefault(key,[]).append(float(value))
            for i, raw in enumerate(pixels):
                own=image(raw,colors[i,0]); changed=image(raw,colors[i,1])
                views = [
                    ('reference', own, own),
                    ('both_tags', changed, changed),
                    ('context_tag', changed, own),
                    ('goal_tag', own, changed),
                ]
                if not args.candidate_audit:
                    other=image(pixels[int(partners[i])],colors[i,0])
                    frames=torch.stack([own[0],changed[0],other[0]])
                    backbone=model.encoder(frames,interpolate_pos_encoding=True).last_hidden_state[:,0].float()
                    for rep, emb in [('backbone',backbone),('projection',model.projector(backbone))]:
                        tag_dist=(emb[0]-emb[1]).square().mean().item()
                        content_dist=(emb[0]-emb[2]).square().mean().item()
                        add(rep+'/tag_intervention_mse',tag_dist)
                        add(rep+'/content_intervention_mse',content_dist)
                    views += [('context_scene', other, own), ('goal_scene', own, other)]

                # Same history actions, same future candidate bank for all tag conditions.
                plans=bank_actions[bank_ids[i],:args.history+args.horizon-1].clone()
                plans[:,:args.history-1]=actions[i,:args.history-1]
                plans=plans.unsqueeze(0).to(device)
                costs={}
                for condition, ctx, goal in views:
                    count=args.candidates
                    info={'pixels':ctx[:args.history][None,None].expand(1,count,-1,-1,-1,-1),
                          'goal':goal[-1:][None,None].expand(1,count,-1,-1,-1,-1),
                          'action':plans.clone()}
                    costs[condition]=model.get_cost(info,plans.clone()).cpu().numpy()

                # Change the future action order while holding context, goal,
                # action magnitudes, and the candidate bank fixed. This is a
                # positive control for action-conditioned cost sensitivity.
                action_plans=plans.clone()
                future_start=args.history-1
                future=action_plans[:,:,future_start:].clone()
                if future.shape[2] < 2:
                    raise ValueError('Action intervention needs at least two future blocks')
                action_plans[:,:,future_start:]=future.roll(shifts=1,dims=2)
                count=args.candidates
                action_info={
                    'pixels':own[:args.history][None,None].expand(1,count,-1,-1,-1,-1),
                    'goal':own[-1:][None,None].expand(1,count,-1,-1,-1,-1),
                    'action':action_plans.clone(),
                }
                costs['action_sequence']=model.get_cost(
                    action_info, action_plans.clone()
                ).cpu().numpy()
                first=plans[:,:,args.history-1].cpu().numpy()
                changed_first=action_plans[:,:,args.history-1].cpu().numpy()
                for condition in [name for name, _, _ in views[1:]] + ['action_sequence']:
                    action_argument = (
                        changed_first if condition == 'action_sequence' else None
                    )
                    for key, value in cost_changes(
                        costs['reference'], costs[condition], first,
                        changed_first_actions=action_argument,
                    ).items():
                        add(condition+'/'+key,value[0])
                if args.candidate_audit:
                    for condition in CONDITIONS:
                        all_costs[condition].append(costs[condition][0].tolist())
                    for condition, gaps in frozen_anchor_gaps(
                        model, own, changed, actions[i], bank_actions[i % args.candidates], args.history,
                    ).items():
                        all_anchors[condition].append(gaps)
                if (i+1)%16==0:
                    print(f'{label}: {i+1}/{len(ids)}',flush=True)
            result['runs'][label]={'run_name':run,'checkpoint':filename,'sha256':digest.hexdigest(),
                'metrics':{k:summarize(v,args.seed) for k,v in metrics.items()}, 'per_clip':metrics}
            if args.candidate_audit:
                result['runs'][label]['candidate_audit'] = {
                    'costs': all_costs, 'anchor': all_anchors, 'rank': candidate_gaps(all_costs),
                }
            del model
            if device.type=='cuda': torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(f'TAG_INTERVENTION_COMPLETE {args.out}',flush=True)


if __name__=='__main__':
    main()
