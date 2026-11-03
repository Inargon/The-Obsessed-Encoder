"""Frozen JEPA features and real action branches for goal-role augmentation.

Two goals are hindsight branch endpoints, one is a recorded future pose.
This constructed benchmark measures candidate selection, NOT calibrated real
deployment probabilities or held-out encoder-pretraining generalization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from budget_data import episode_runs
from goal_role_data import branch_targets, set_render_pose
from goal_agent_intervention import render_goal_variants


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--queries', type=int, default=60)
    p.add_argument('--candidates', type=int, default=6)
    p.add_argument('--seed', type=int, default=1801)
    p.add_argument('--run', default='pusht_clean_jepa_matched_full_seed0')
    args = p.parse_args()
    if args.queries < 10 or args.candidates < 3:
        p.error('Need >=10 queries and >=3 candidates')
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    import h5py
    from PIL import Image
    from stable_worldmodel.envs.pusht.env import PushT
    from additional_files.evaluate_reacher_checkpoint import load_inference_model
    from utils import get_img_preprocessor
    args.output.mkdir(parents=True, exist_ok=False)
    home = Path(os.environ['STABLEWM_HOME'])
    with h5py.File(home/'datasets/pusht_expert_train.h5', 'r') as f:
        states, actions = f['state'][:], f['action'][:]
        ids = f['episode_idx'][:].reshape(-1)
        runs = episode_runs(ids, f['step_idx'][:])
    if states.shape[1:] != (7,) or actions.shape[1:] != (2,):
        raise ValueError('Unexpected PushT state/action schema')
    eligible = [(a,b) for a,b in runs if b-a >= 61]
    rng = np.random.default_rng(args.seed)
    if len(eligible) < args.queries+1:
        raise ValueError('Too few long episodes')
    order = rng.permutation(len(eligible))
    query_runs = [eligible[i] for i in order[:args.queries]]
    bank_runs = [eligible[i] for i in order[args.queries:]]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_inference_model(args.run, 'weights_epoch_10.pt').to(device).eval().requires_grad_(False)
    preprocess = get_img_preprocessor('pixels', 'pixels', img_size=224)
    # No target marker: otherwise the fixed marker can leak a privileged goal.
    env = PushT(resolution=224, with_target=False, relative=True)
    target_env = PushT(resolution=224, with_target=False, relative=True)

    def snapshot(e):
        return np.array([*e.agent.position, *e.block.position, e.block.angle,
                         *e.agent.velocity, *e.block.velocity, e.block.angular_velocity])

    def encode(images):
        batch = torch.from_numpy(np.stack(images).copy()).permute(0,3,1,2)
        batch = preprocess({'pixels':batch})['pixels'].to(device)
        with torch.inference_mode():
            return model.encode({'pixels':batch.unsqueeze(0)})['emb'][0].cpu().numpy()

    histories, sequences, goals_emb, targets, initial_hits, rows, traces_all, goal_poses = [],[],[],[],[],[],[],[]
    try:
        for qi, (a,b) in enumerate(query_runs):
            row = int(rng.integers(a,b-60))
            initial = states[row].copy()
            warmup = np.clip(actions[row:row+10], env.action_space.low, env.action_space.high)

            def restore():
                env.reset(seed=args.seed+qi, options={'state':initial.copy()})
                images = [np.asarray(env.render()).copy()]
                for action in warmup:
                    env.step(action.copy()); images.append(np.asarray(env.render()).copy())
                return images, snapshot(env)

            frames, current = restore()
            hist = encode([frames[i] for i in (0,5,10)]).ravel()
            bank = [np.zeros((40,2), np.float32), actions[row+10:row+50].copy()]
            for _ in range(args.candidates-2):
                ba, bb = bank_runs[rng.integers(len(bank_runs))]
                r = int(rng.integers(ba,bb-39))
                bank.append(actions[r:r+40].copy())
            bank = np.clip(np.asarray(bank), env.action_space.low, env.action_space.high)
            traces = []
            for sequence in bank:
                replay, check = restore()
                np.testing.assert_allclose(check,current,atol=1e-5,rtol=0)
                np.testing.assert_array_equal(replay[-1],frames[-1])
                trace = []
                for action in sequence:
                    env.step(action.copy()); trace.append(snapshot(env))
                traces.append(trace)
            traces = np.asarray(traces)
            # Chosen before inspecting success; retain duplicate/initial goals.
            poses = np.asarray([traces[1,-1,2:5],traces[2,-1,2:5],states[row+60,2:5]])
            emb, labels, hits0 = [],[],[]
            for gi, goal in enumerate(poses):
                target_env.reset(seed=args.seed+qi)
                target_env.agent.position = tuple(np.clip(states[row+60,:2],48,464))
                target_env.space.reindex_shapes_for_body(target_env.agent)
                set_render_pose(target_env.block, target_env.space, goal)
                np.testing.assert_allclose(snapshot(target_env)[2:5],goal,atol=1e-6,rtol=0)
                images, _ = render_goal_variants(target_env, lambda:snapshot(target_env))
                np.testing.assert_allclose(snapshot(target_env)[2:5],goal,atol=1e-6,rtol=0)
                emb.append(encode(images))
                label, hit0 = branch_targets(current,traces,goal)
                labels.append(label); hits0.append(hit0)
                if qi < 2:
                    for vi, image in enumerate(images):
                        Image.fromarray(image).save(args.output/f'q{qi}-g{gi}-v{vi}.png')
            histories.append(np.r_[hist,warmup.ravel()]); sequences.append(bank)
            goals_emb.append(emb); targets.append(labels); initial_hits.append(hits0)
            traces_all.append(traces); goal_poses.append(poses); rows.append(row)
            print('GOAL_ROLE_COLLECTED',qi+1,flush=True)
    finally:
        env.close(); target_env.close()
    np.savez_compressed(args.output/'features.npz', history=histories, actions=sequences,
        goals=goals_emb, labels=targets, initial_hit=initial_hits, episode=ids[rows], rows=rows,
        traces=traces_all, goal_poses=goal_poses, budgets=[5,10,20,40])
    checkpoint = home/'checkpoints'/args.run/'weights_epoch_10.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (args.output/'protocol.json').write_text(json.dumps({'scope':__doc__, 'seed':args.seed,
        'checkpoint_sha256':digest,'run':args.run,'target_marker':False,
        'goal_variants':['original','left32','right32','up32','down32'],
        'train_variants':[0,1,2],'unseen_direction_test_variants':[3,4],
        'goal_sources':['candidate1_endpoint','candidate2_endpoint','recorded_future60'],
        'success_predicate':'xy<=10px AND wrapped angle<=0.15rad; not official SR',
        'limitations':['Frozen pretrained encoder may have seen query episodes',
          'Image occlusion can change when moving target agent',
          'Original seven-state resets omit block velocities; branches share realized reset',
          'Hindsight goals and restricted candidates preclude deployment calibration claims']},indent=2))
    print('GOAL_ROLE_FEATURES_COMPLETE',args.output,flush=True)


if __name__ == '__main__':
    main()
