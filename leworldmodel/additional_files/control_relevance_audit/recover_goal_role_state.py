"""Recover PRE-action state by exact replay; future traces verify replay only."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    os.environ.setdefault('SDL_AUDIODRIVER','dummy')
    import h5py
    from stable_worldmodel.envs.pusht.env import PushT
    with np.load(args.data/'features.npz') as d:
        data={k:d[k] for k in d.files}
    protocol=json.loads((args.data/'protocol.json').read_text())
    env=PushT(resolution=224,with_target=False,relative=True)
    histories=[]
    def state():
        return np.array([*env.agent.position,*env.block.position,env.block.angle,
            *env.agent.velocity,*env.block.velocity,env.block.angular_velocity])
    try:
        with h5py.File(Path(os.environ['STABLEWM_HOME'])/'datasets/pusht_expert_train.h5','r') as f:
            for qi,row in enumerate(data['rows']):
                initial=f['state'][int(row)]
                warmup=data['history'][qi,-20:].reshape(10,2)
                saved=None
                for ci,sequence in enumerate(data['actions'][qi]):
                    env.reset(seed=protocol['seed']+qi,options={'state':initial.copy()})
                    past=[state()]
                    for a in warmup:
                        env.step(a.copy());past.append(state())
                    history=np.asarray(past)[[0,5,10]]
                    if saved is None: saved=history.copy()
                    else: np.testing.assert_allclose(history,saved,atol=1e-5,rtol=0)
                    future=[]
                    for a in sequence:
                        env.step(a.copy());future.append(state())
                    np.testing.assert_allclose(future,data['traces'][qi,ci],atol=1e-5,rtol=0,
                        err_msg='Replay mismatch: cannot use recovered state with old labels')
                histories.append(saved)
                print('STATE_REPLAY_VERIFIED',qi+1,flush=True)
    finally:
        env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,state_history=histories,goal_poses=data['goal_poses'],episode=data['episode'])
    print('STATE_CACHE_COMPLETE',args.output,flush=True)


if __name__=='__main__':
    main()
