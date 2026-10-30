"""Export frozen PushT history features using the existing LeWM environment."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--cache-dir',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--horizon',type=int,default=5)
    p.add_argument('--samples',type=int,default=512);p.add_argument('--seed',type=int,default=73)
    p.add_argument('--representation',choices=['backbone','projection'],default='projection')
    args=p.parse_args()
    if args.horizon<1 or args.samples<5:p.error('Positive horizon and >=5 samples required')
    if args.out.exists() or args.out.with_suffix('.json').exists():p.error('Output already exists')
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    import numpy as np
    import torch
    import stable_worldmodel as swm
    from hydra.utils import instantiate
    def load_inference_model(run,filename):
        config_path=home/'checkpoints'/run/'config.json'
        config=json.loads(config_path.read_text())
        weights=torch.load(home/'checkpoints'/run/filename,map_location='cpu',weights_only=True)
        weights={k:v for k,v in weights.items() if not k.startswith('control_objective.')}
        model=instantiate(config)
        model.load_state_dict(weights,strict=True)
        return model
    from utils import get_img_preprocessor
    home=Path(os.environ.get('STABLEWM_HOME',Path.home()/'.stable_worldmodel'))
    ckpt=home/'checkpoints'/args.run/args.checkpoint
    if not ckpt.is_file():raise FileNotFoundError(ckpt)
    digest=hashlib.sha256()
    with ckpt.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    nsteps=args.horizon+2
    dataset=swm.data.load_dataset(args.dataset,cache_dir=args.cache_dir,
        num_steps=nsteps,frameskip=1,keys_to_cache=['state','action'])
    # Never silently fall back to clip IDs as episode IDs.
    for key in ('episode_idx','ep_idx'):
        try:
            episodes=np.asarray(dataset.get_col_data(key)).reshape(-1)
            break
        except (KeyError,ValueError,AttributeError):pass
    else:raise ValueError('Dataset lacks episode_idx/ep_idx; supply real episode metadata')
    steps=np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    if len(steps)!=len(episodes):raise ValueError('Metadata columns have different lengths')
    if len(steps)<nsteps:raise ValueError('Dataset shorter than requested window')
    valid=np.arange(min(len(dataset),len(steps)-nsteps+1))
    valid=valid[(episodes[valid]==episodes[valid+nsteps-1]) &
                (steps[valid+nsteps-1]-steps[valid]==nsteps-1)]
    rng=np.random.default_rng(args.seed)
    chosen=np.sort(rng.choice(valid,min(args.samples,len(valid)),replace=False))
    if len(np.unique(episodes[chosen]))<5:raise ValueError('Sampled fewer than five episodes; increase --samples')
    dataset.transform=get_img_preprocessor('pixels','pixels',img_size=224)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    model=load_inference_model(args.run,args.checkpoint).to(device).eval().requires_grad_(False)
    fields={k:[] for k in ('features','state','previous_state','next_state','action','episode_id','row_index')}
    with torch.no_grad():
        for row in chosen:
            item=dataset[int(row)];s=torch.as_tensor(item['state']).cpu().numpy()
            pixels=torch.as_tensor(item['pixels']);actions=torch.as_tensor(item['action']).cpu().numpy()
            if s.ndim!=2 or s.shape[0]!=nsteps or s.shape[1]<1 or len(pixels)!=nsteps or len(actions)!=nsteps:
                raise ValueError('Unexpected temporal layout; expected frameskip=1 pose/action windows')
            emb=model.encoder(pixels[:2].to(device),interpolate_pos_encoding=True).last_hidden_state[:,0].float()
            if args.representation=='projection':emb=model.projector(emb)
            values={'features':emb.flatten().float().cpu().numpy(),'state':s[1], 'previous_state':s[0],
                'next_state':s[-1], 'action':actions[1:-1].reshape(-1),
                'episode_id':episodes[row], 'row_index':row}
            for k,v in values.items():fields[k].append(v)
            if len(fields['features'])%32==0:print('EXPORTED',len(fields['features']),flush=True)
    from audit import validate
    bundle={k:np.asarray(v) for k,v in fields.items()};bundle['state_schema']=np.asarray('raw');validate(bundle)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('xb') as f:np.savez_compressed(f,**bundle)
    provenance=dict(vars(args),out=str(args.out),checkpoint_sha256=digest.hexdigest(),device=str(device),
        frameskip=1,history_frames=2,tag='none; export clean dataset only',
        state_order='Unmodified dataset columns; field semantics not yet verified',
        state_dim=int(bundle['state'].shape[1]),numpy_version=np.__version__,torch_version=torch.__version__,
        limitation='Native action labels retained. Observational windows are NOT same-state interventions.')
    with args.out.with_suffix('.json').open('x',encoding='utf-8') as f:json.dump(provenance,f,indent=2)
    print('Exported',len(chosen),'windows')

if __name__=='__main__':main()
