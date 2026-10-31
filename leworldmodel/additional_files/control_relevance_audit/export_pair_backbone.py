"""Add frozen single-frame backbone features to the verified projection cache."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from state_terminal_readout import aligned_visual_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():p.error('Refusing to overwrite output directory')
    source=args.fixture/'branches.npz'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    meta=json.loads((args.cache/'report.json').read_text())
    if meta['input_sha256']!=digest:raise ValueError('Fixture hash mismatch')
    with np.load(source,allow_pickle=False) as f:
        initial=f['initial_states'];pixels=f['initial_pixels'];goals=f['goals']
    n=len(initial);ids=np.repeat(np.arange(n),2);goals=np.repeat(goals,2,axis=0)
    initial=initial.reshape(n*2,7)
    with np.load(args.cache/'features.npz',allow_pickle=False) as f:bundle=dict(f)
    aligned_visual_inputs(bundle,initial,goals,ids)
    if 'jepa_backbone' in bundle:raise ValueError('Input cache already has backbone features')
    goal_suffix=np.column_stack([goals[:,:2],np.sin(goals[:,2]),np.cos(goals[:,2])])
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    import torch
    from hydra.utils import instantiate
    from utils import get_img_preprocessor
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess=get_img_preprocessor('pixels','pixels',img_size=224)
    home=Path(os.environ['STABLEWM_HOME'])
    for name in ('jepa','ours'):
        checkpoint=Path(meta['checkpoints'][name]['checkpoint'])
        # Honor the original artifact path; do not silently switch checkpoints.
        sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if sha!=meta['checkpoints'][name]['sha256']:raise ValueError('Checkpoint changed since feature export')
        model=instantiate(json.loads((checkpoint.parent/'config.json').read_text()))
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict({k:v for k,v in weights.items() if not k.startswith('control_objective.')},strict=True)
        model=model.to(device).eval().requires_grad_(False)
        features=[];projected=[]
        with torch.no_grad():
            for image in pixels.reshape(-1,*pixels.shape[2:]):
                batch=torch.from_numpy(image.copy()).permute(2,0,1).unsqueeze(0)
                batch=preprocess({'pixels':batch})['pixels'].to(device)
                z=model.encoder(batch,interpolate_pos_encoding=True).last_hidden_state[:,0].float()
                features.append(z.flatten().cpu().numpy())
                projected.append(model.projector(z).flatten().float().cpu().numpy())
        # End-to-end guard: the backbone comes from the same image/checkpoint/preprocessing.
        np.testing.assert_allclose(np.asarray(projected),bundle[name][:,:-4],rtol=1e-4,atol=1e-4,
                                    err_msg='Recomputed projection differs from original cache')
        bundle[name+'_backbone']=np.column_stack([np.asarray(features),goal_suffix])
        meta['checkpoints'][name+'_backbone']={'checkpoint':str(checkpoint),'sha256':sha,
            'representation':'CLS before projector','feature_dim':int(np.asarray(features).shape[1])}
        print('BACKBONE_AND_PROJECTION_CACHE_VERIFIED',name,len(features),flush=True)
        del model
    aligned_visual_inputs(bundle,initial,goals,ids)
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/'features.npz').open('xb') as f:np.savez_compressed(f,**bundle)
    metadata={'input_sha256':digest,'checkpoints':meta['checkpoints'],'input':str(source),
        'source_cache':str(args.cache),'protocol':'same single-frame CLS, before and after projector; projected cache reproduced within 1e-4 tolerance'}
    with (args.out/'report.json').open('x') as f:json.dump(metadata,f,indent=2,allow_nan=False)
    print('BACKBONE_EXPORT_COMPLETE',args.out,flush=True)


if __name__=='__main__':main()
