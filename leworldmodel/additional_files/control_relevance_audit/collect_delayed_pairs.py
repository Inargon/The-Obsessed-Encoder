"""Collect controlled PushT branches to test delayed interaction distinctions.
No encoder training or claims of a new method. All candidate pairs are retained.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def pose_change(states):
    delta=states[...,2:5]-states[...,0:1,2:5]
    delta=delta.copy()
    delta[...,2]=np.arctan2(np.sin(delta[...,2]),np.cos(delta[...,2]))
    return delta


def delayed_mask(states, short, distance=1., long_distance=10.):
    # [pair, scene, action, time, seven-state]
    delta=pose_change(states)
    early=np.linalg.norm(delta[...,:short+1,:2],axis=-1).max(axis=(1,2,3))
    early_angle=np.abs(delta[...,:short+1,2]).max(axis=(1,2,3))
    # Same action, different scene: compare object displacement, not absolute position.
    separation=np.linalg.norm(delta[:,0,:,-1,:2]-delta[:,1,:,-1,:2],axis=-1).max(axis=1)
    return (early<distance)&(early_angle<.02)&(separation>long_distance),early,separation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--pairs',type=int,default=32)
    p.add_argument('--horizon',type=int,default=40);p.add_argument('--short',type=int,default=3)
    p.add_argument('--seed',type=int,default=901);args=p.parse_args()
    if args.out.exists():p.error('Refusing to overwrite output directory')
    if args.pairs<1 or not 0<args.short<args.horizon:p.error('Invalid pair count or horizons')
    os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
    from stable_worldmodel.envs.pusht.env import PushT
    import inspect
    args.out.mkdir(parents=True,exist_ok=False)
    # Preserve the installed implementation for reproducibility and later review.
    (args.out/'installed_pusht.py.txt').write_text(inspect.getsource(inspect.getmodule(PushT)))
    env=PushT(resolution=224,with_target=True,relative=True)
    rng=np.random.default_rng(args.seed)
    actions=np.array([[.35,-.12],[.35,0.],[.35,.12]],dtype=np.float32)
    trajectories=[];images=[];initial=[]
    def state():return np.array([*env.agent.position,*env.block.position,env.block.angle,*env.agent.velocity],dtype=float)
    def rollout(s,action):
        env.reset(seed=args.seed,options={'state':s.copy()})
        env.goal_pose=np.array([350.,256.,0.])
        actual=state();error=actual-s;error[4]=np.arctan2(np.sin(error[4]),np.cos(error[4]))
        if np.max(np.abs(error))>1e-4:raise ValueError('Reset did not restore requested state: '+str(error))
        if np.linalg.norm(np.asarray(env.block.velocity))>1e-6 or abs(env.block.angular_velocity)>1e-6:
            raise ValueError('Initial block velocity not zero')
        first=np.asarray(env.render()).copy()
        track=[actual]
        for _ in range(args.horizon):
            env.step(action.copy());track.append(state())
        return np.asarray(track),first
    try:
        for pair in range(args.pairs):
            x=rng.uniform(245,285);offset=rng.uniform(25,45);angle=rng.uniform(-.6,.6)
            starts=np.array([[80.,256.,x,256.-offset,angle,0.,0.],
                             [80.,256.,x,256.+offset,angle,0.,0.]])
            branches=[];frames=[]
            for s in starts:
                scene=[];first=None
                for a in actions:
                    track,frame=rollout(s,a)
                    if first is None:first=frame
                    else:np.testing.assert_array_equal(first,frame,err_msg='Action branches have different initial images')
                    scene.append(track)
                branches.append(scene);frames.append(first)
            trajectories.append(branches);images.append(frames);initial.append(starts)
            print('COLLECTED_PAIR',pair+1,flush=True)
        # Verify full-branch reproducibility separately, without changing the collected data.
        repeated,_=rollout(initial[0][0],actions[0])
        np.testing.assert_allclose(repeated,np.asarray(trajectories)[0,0,0],rtol=0,atol=1e-5,
                                   err_msg='Repeated branch is not deterministic under reset')
    finally:env.close()
    states=np.asarray(trajectories)
    if not np.isfinite(states).all():raise ValueError('Nonfinite rollout state')
    keep,early,separation=delayed_mask(states,args.short)
    # Fixed physical goal, shared across scenes and actions. Best means best of these 3 candidates only.
    final=states[:,:,:,-1,2:5];angle=np.arctan2(np.sin(final[...,2]),np.cos(final[...,2]))
    cost=np.sum((final[...,:2]-np.array([350.,256.]))**2,axis=-1)+(30*angle)**2
    order=np.argsort(cost,axis=-1);best=order[...,0]
    margin=np.take_along_axis(cost,order[...,1:2],axis=-1)[...,0]-np.take_along_axis(cost,order[...,:1],axis=-1)[...,0]
    choice_diff=(best[:,0]!=best[:,1])&(margin.min(axis=1)>25.)
    with (args.out/'branches.npz').open('xb') as f:
        np.savez_compressed(f,states=states,initial_states=np.asarray(initial),initial_pixels=np.asarray(images),
            actions=actions,delayed_mask=keep,early_motion=early,late_displacement_separation=separation,
            candidate_costs=cost,candidate_best_actions=best,candidate_choice_diff=choice_diff)
    report={'pairs':args.pairs,'short_horizon':args.short,'long_horizon':args.horizon,'seed':args.seed,
        'delayed_pairs':int(keep.sum()),'candidate_choice_difference_pairs':int(choice_diff.sum()),
        'both_conditions_pairs':int((keep&choice_diff).sum()),'repeated_branch_check':'passed',
        'early_translation_threshold_px':1.,'early_rotation_threshold_rad':.02,'late_displacement_difference_px':10.,
        'candidate_choice_min_cost_gap':25.,'candidate_goal':[350.,256.,0.],
        'state_order':['agent_x','agent_y','block_x','block_y','block_angle','agent_vx','agent_vy'],
        'limitations':['All pairs saved, selection mask is explicit; no successful-pair count is guaranteed',
          'Best action is relative to three constant command sequences and one declared goal, not globally optimal',
          'New simulated initial scenes; not evidence about original dataset distribution or learned encoders',
          'Paired scenes share agent state; action branches share full reset state and seed',
          'Short-term similarity concerns object motion, not image similarity or full state equality']}
    with (args.out/'summary.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('PAIR_FIXTURE_SUMMARY',json.dumps(report),flush=True)
    print('PAIR_FIXTURE_COMPLETE',args.out,flush=True)


if __name__=='__main__':main()
