"""Small fixed-bank closed-loop audit: learned vs simulated endpoints, same score.

Not a reproduction of CEM or published success rates. Simulator branches are
restored by reset and full prefix replay, never by incomplete position snapshots.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import numpy as np


def choose(costs):
    costs = np.asarray(costs)
    if costs.ndim != 1 or len(costs) < 2 or not np.isfinite(costs).all():
        raise ValueError('Invalid candidate scores')
    return int(costs.argmin())


def goal_error(state, goal):
    d = np.asarray(state)[2:5] - np.asarray(goal)[2:5]
    return float(np.linalg.norm(d[:2])), float(abs(np.arctan2(np.sin(d[2]), np.cos(d[2]))))


def candidate_diagnostics(current, branches, goal, learned_cost, oracle_cost):
    """Physical short-horizon diagnostics only; never used for action selection."""
    branches = np.asarray(branches, dtype=float)
    if branches.ndim != 3 or branches.shape[2] != 10 or not np.isfinite(branches).all():
        raise ValueError('Expected finite candidate x horizon x full-state traces')
    before = np.asarray(goal_error(current, goal))
    errors = np.asarray([goal_error(branch[-1], goal) for branch in branches])
    tolerance = np.array([1e-5, 1e-7])
    def dominates(a, b):
        return np.all(a <= b + tolerance, axis=-1) & np.any(a < b - tolerance, axis=-1)
    selections = {}
    for name, costs in (('learned', learned_cost), ('oracle', oracle_cost)):
        if len(costs) != len(branches):
            raise ValueError('Candidate score count mismatch')
        selected = choose(costs)
        selections[name] = {'selected': selected,
            'dominating_candidates': np.flatnonzero(dominates(errors, errors[selected])).tolist(),
            'improves_both_over_current': bool(dominates(errors[selected], before))}
    return {'current_error': before.tolist(), 'endpoint_errors': errors.tolist(),
        'improving_candidates': np.flatnonzero(dominates(errors, before)).tolist(),
        'any_branch_reaches_threshold': any(
            xy <= 10 and angle <= .1 for branch in branches
            for xy, angle in (goal_error(state, goal) for state in branch)),
        'selections': selections}


def history_indices(length):
    if length < 11:
        raise ValueError('Need ten actual executed actions for native history')
    return [length - 11, length - 6, length - 1]


def pack_history_actions(prefix, bank):
    prefix, bank = np.asarray(prefix), np.asarray(bank)
    if prefix.ndim != 2 or prefix.shape[1] != 2 or len(prefix) < 10 or bank.ndim != 3 or bank.shape[1:] != (5, 2):
        raise ValueError('Need real ten-action history and five-action candidates')
    past = prefix[-10:].reshape(2, 10)
    return np.concatenate((np.broadcast_to(past, (len(bank), 2, 10)),
                           bank.reshape(len(bank), 1, 10)), axis=1)


class ResetReferences:
    """Pin realized resets, rather than assuming reset is an identity setter."""
    def __init__(self):
        self.references = {}

    def check(self, initial, goal, seed, actual, frame):
        actual = np.asarray(actual, dtype=float)
        if actual.shape != (10,) or not np.isfinite(actual).all():
            raise ValueError('Invalid realized simulator state')
        key = (int(seed), tuple(np.asarray(initial, dtype=float)), tuple(np.asarray(goal, dtype=float)))
        if key not in self.references:
            self.references[key] = (actual.copy(), frame.copy())
            return True
        reference, reference_frame = self.references[key]
        np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=0,
                                   err_msg='Repeated reset changed realized physical state')
        np.testing.assert_array_equal(frame, reference_frame,
                                      err_msg='Repeated reset changed rendered image')
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--goal-agent-audit', action='store_true')
    p.add_argument('--goal-set-policy', action='store_true')
    p.add_argument('--reference-report', type=Path)
    p.add_argument('--queries', type=int, default=2)
    p.add_argument('--budget', type=int, default=25)
    p.add_argument('--candidates', type=int, default=8)
    p.add_argument('--goal-offset', type=int, default=100)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--run', default='pusht_clean_jepa_matched_full_seed0')
    args = p.parse_args()
    if args.goal_set_policy and not args.goal_agent_audit:
        p.error("Goal-set policy requires --goal-agent-audit")
    reference = json.loads(args.reference_report.read_text()) if args.reference_report else None
    if args.out.exists() or args.queries < 1 or args.candidates < 2 or args.budget < 5 or args.budget % 5 or args.goal_offset < 15:
        p.error('New output, valid counts, budget multiple of five required')
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    import stable_worldmodel as swm
    from stable_worldmodel.envs.pusht.env import PushT
    from additional_files.evaluate_reacher_checkpoint import load_inference_model
    from additional_files.mechanism_metrics import normalize_action_blocks
    from utils import get_column_normalizer, get_img_preprocessor

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'installed_pusht.txt').write_text(inspect.getsource(inspect.getmodule(PushT)))
    home = Path(os.environ['STABLEWM_HOME'])
    dataset = swm.data.load_dataset(str(home / 'datasets/pusht_expert_train.h5'),
        cache_dir=os.environ['LOCAL_DATASET_DIR'], num_steps=4, frameskip=5,
        keys_to_load=['state', 'action'], keys_to_cache=['state', 'action'])
    states = np.asarray(dataset.get_col_data('state'))
    actions = np.asarray(dataset.get_col_data('action'))
    episodes = np.asarray(dataset.get_col_data('episode_idx')).reshape(-1)
    if states.shape[1:] != (7,) or actions.shape[1:] != (2,):
        raise ValueError('Unexpected native raw schema')
    norm = get_column_normalizer(dataset, 'action', 'action')
    preprocess = get_img_preprocessor('pixels', 'pixels', img_size=224)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    model = load_inference_model(args.run, 'weights_epoch_10.pt').to(device).eval().requires_grad_(False)
    env = PushT(resolution=224, with_target=True, relative=True)
    rng = np.random.default_rng(args.seed)
    # Fixed query selection before any model/oracle outcome. Unique episodes.
    rows, used = [], set()
    for clip in rng.permutation(len(dataset)):
        ep, start = map(int, dataset.clip_indices[int(clip)])
        row = int(dataset.offsets[ep]) + start
        if ep in used or start + args.goal_offset >= int(dataset.lengths[ep]):
            continue
        if not np.isfinite(states[row:row+args.goal_offset+1]).all() or not np.isfinite(actions[row:row+10]).all():
            continue
        rows.append(row); used.add(ep)
        if len(rows) == args.queries:
            break
    if len(rows) != args.queries:
        raise ValueError('Insufficient query episodes')
    query_episodes = set(episodes[rows].tolist())
    bank_rows = np.flatnonzero(~np.isin(episodes, list(query_episodes)))
    checkpoint = home / 'checkpoints' / args.run / 'weights_epoch_10.pt'
    digest = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    report = {'protocol': {**vars(args), 'out': str(args.out), 'reference_report': str(args.reference_report) if args.reference_report else None,
        'checkpoint_sha256': digest.hexdigest(), 'horizon': 5, 'history': 3, 'frameskip': 5,
        'score': 'Squared projected-latent distance to identical rendered final goal',
        'restore': 'Pin realized post-reset state/image for each request; verify repeated resets and full prefix replay',
        'goal_tolerance': 'Diagnostic only: object xy <=10 pixels AND wrapped angle <=0.1 rad',
        'limitations': ['Not original CEM evaluation; restricted recorded candidate bank',
            'New unfiltered queries, not identified prior failed episodes',
            'Queries from existing training dataset; not a held-out generalization claim',
            'Seven-state initial reset may omit original block velocities; goal reachability not guaranteed',
            'Simulator reset may advance physics; actual post-reset state is recorded, not assumed equal to dataset row',
            'Both arms use rendered simulator history, no fabricated repeated context frames',
            'Oracle is simulator-assisted diagnostic, not deployable policy',
            'Physical candidate diagnostics measure short-horizon progress only; temporary retreat may be necessary',
            'Pareto dominance uses xy and wrapped-angle errors separately, no fitted physical cost weights']}, 'queries': []}

    def snapshot():
        return np.array([*env.agent.position, *env.block.position, env.block.angle,
                         *env.agent.velocity, *env.block.velocity, env.block.angular_velocity], float)

    reset_references = ResetReferences()

    def restore(initial, goal, prefix, seed):
        env.reset(seed=seed, options={'state': initial.copy()})
        if not hasattr(env, '_set_goal_state'):
            raise ValueError('Installed PushT lacks native goal setter')
        env._set_goal_state(goal.copy())
        frames = [np.asarray(env.render()).copy()]
        trace = [snapshot()]
        first = reset_references.check(initial, goal, seed, trace[0], frames[0])
        if first:
            print('RESET_REALIZED', json.dumps({'seed': seed, 'requested': initial.tolist(),
                  'actual': trace[0].tolist(), 'delta_first_seven': (trace[0][:7]-initial).tolist()}), flush=True)
        for action in prefix:
            env.step(np.asarray(action).copy())
            frames.append(np.asarray(env.render()).copy()); trace.append(snapshot())
        return frames, np.asarray(trace)

    def encode(images):
        batch = torch.from_numpy(np.stack(images).copy()).permute(0, 3, 1, 2)
        batch = preprocess({'pixels': batch})['pixels'].to(device)
        return model.encode({'pixels': batch.unsqueeze(0)})['emb'][0]

    try:
        with torch.inference_mode():
            for qi, row in enumerate(rows):
                initial, goal = states[row].copy(), states[row+args.goal_offset].copy()
                seed = args.seed + qi
                # Same candidate sequences for both arms at each decision index.
                banks, sources = [], []
                for _ in range(args.budget // 5):
                    bank, source = [np.zeros((5, 2), dtype=np.float32)], [-1]
                    for _attempt in range(100000):
                        r = int(rng.choice(bank_rows))
                        if r + 5 > len(actions) or not np.all(episodes[r:r+5] == episodes[r]):
                            continue
                        a = actions[r:r+5].copy()
                        if not np.isfinite(a).all():
                            continue
                        bank.append(np.clip(a, env.action_space.low, env.action_space.high)); source.append(r)
                        if len(bank) == args.candidates:
                            break
                    if len(bank) != args.candidates:
                        raise ValueError('Could not form candidate bank')
                    banks.append(np.asarray(bank)); sources.append(source)
                goal_frames, goal_trace = restore(goal, goal, [], seed)
                goal_xy, goal_angle = goal_error(goal_trace[0], goal)
                if goal_xy > 1e-4 or goal_angle > 1e-4:
                    raise ValueError('Rendered goal object pose differs from requested goal; inspect reset source')
                goal_emb = encode(goal_frames)[0]
                goal_variant_embs, goal_variant_records = None, None
                if args.goal_agent_audit:
                    from additional_files.control_relevance_audit.goal_agent_intervention import render_goal_variants, rank_variants, goal_set_cost
                    from PIL import Image
                    goal_images, goal_variant_records = render_goal_variants(env, snapshot)
                    # Encode each target with exactly the baseline batch shape.
                    goal_variant_embs = torch.stack([encode([im])[0] for im in goal_images])
                    torch.testing.assert_close(goal_variant_embs[0], goal_emb, atol=1e-6, rtol=1e-6)
                    for vi, im in enumerate(goal_images):
                        Image.fromarray(im).save(args.out / f'query{qi}-goal-{vi}.png')
                    print('GOAL_AGENT_RENDER', qi, json.dumps(goal_variant_records), flush=True)
                prefix0 = np.clip(actions[row:row+10], env.action_space.low, env.action_space.high).copy()
                record = {'row': int(row), 'episode': int(episodes[row]), 'candidate_source_rows': sources,
                          'initial': initial.tolist(), 'goal': goal.tolist(),
                          'rendered_goal_state': goal_trace[0].tolist(),
                          'goal_agent_variants': goal_variant_records, 'arms': {}}
                for arm in (('learned', 'oracle', 'goal_set') if args.goal_set_policy else ('learned', 'oracle')):
                    prefix = list(prefix0.copy())
                    frames, trace = restore(initial, goal, prefix, seed)
                    xy, angle = goal_error(trace[-1], goal)
                    initial_success = xy <= 10 and angle <= .1
                    success, decisions = initial_success, []
                    for turn, bank in enumerate(banks):
                        if success:
                            break
                        current = trace[-1].copy()
                        hist = encode([frames[i] for i in history_indices(len(frames))])
                        # Each token's action is the five native actions FOLLOWING its frame.
                        packed = pack_history_actions(prefix, bank)
                        normalized = normalize_action_blocks(torch.from_numpy(packed).float(), norm, 2).to(device)
                        predicted = model.predict(hist.unsqueeze(0).expand(len(bank), -1, -1),
                                                  model.action_encoder(normalized))[:, -1]
                        learned_cost = (predicted - goal_emb).square().sum(-1).cpu().numpy()
                        real_images, branch_traces = [], []
                        for candidate in bank:
                            replay_frames, replay_trace = restore(initial, goal, prefix, seed)
                            np.testing.assert_allclose(replay_trace, trace, atol=1e-5, rtol=0,
                                err_msg='Prefix replay is not deterministic')
                            np.testing.assert_array_equal(replay_frames[-1], frames[-1])
                            branch = []
                            for action in candidate:
                                env.step(action.copy()); branch.append(snapshot())
                            branch_traces.append(np.asarray(branch))
                            real_images.append(np.asarray(env.render()).copy())
                        real_emb = encode(real_images)
                        oracle_cost = (real_emb - goal_emb).square().sum(-1).cpu().numpy()
                        diagnostic = candidate_diagnostics(current, branch_traces, goal, learned_cost, oracle_cost)
                        intervention = None
                        if goal_variant_embs is not None:
                            intervention = {}
                            for scorer, candidate_emb in (('learned', predicted), ('oracle', real_emb)):
                                matrix = (candidate_emb[None] - goal_variant_embs[:, None]).square().sum(-1).cpu().numpy()
                                np.testing.assert_allclose(matrix[0], learned_cost if scorer == 'learned' else oracle_cost, atol=1e-4, rtol=1e-5)
                                intervention[scorer] = rank_variants(matrix, diagnostic['endpoint_errors'])
                                result = intervention[scorer]
                                print('GOAL_AGENT_RANK', qi, arm, turn, scorer, json.dumps({k: result[k] for k in
                                    ('selected', 'robust_flips', 'selected_physical_errors', 'advantage_over_original_choice')}), flush=True)
                            if turn == 0:
                                for ci, im in enumerate(real_images):
                                    Image.fromarray(im).save(args.out / f'query{qi}-{arm}-candidate{ci}.png')
                        set_scores, matched_goals = None, None
                        if args.goal_set_policy:
                            set_scores, matched_goals = goal_set_cost(intervention['learned']['costs'])
                        chosen = choose(set_scores if arm == 'goal_set' else
                                        learned_cost if arm == 'learned' else oracle_cost)
                        _, check_trace = restore(initial, goal, prefix, seed)
                        np.testing.assert_allclose(check_trace[-1], current, atol=1e-5, rtol=0)
                        executed = []
                        for action in bank[chosen]:
                            env.step(action.copy()); prefix.append(action.copy())
                            frames.append(np.asarray(env.render()).copy())
                            executed.append(snapshot())
                            xy, angle = goal_error(executed[-1], goal)
                            if xy <= 10 and angle <= .1:
                                success = True
                                break
                        np.testing.assert_allclose(executed, branch_traces[chosen][:len(executed)], atol=1e-5, rtol=0)
                        trace = np.concatenate((trace, executed))
                        decisions.append({'turn': turn, 'selected': chosen,
                            'learned_cost': learned_cost.tolist(), 'oracle_cost': oracle_cost.tolist(),
                            'oracle_scoring_regret_of_learned': float(oracle_cost[choose(learned_cost)] - oracle_cost.min()),
                            'xy_error': xy, 'angle_error': angle,
                            'current_state': current.tolist(),
                            'candidate_actions': bank.tolist(),
                            'candidate_state_traces': np.asarray(branch_traces).tolist(),
                            'physical_diagnostic': diagnostic, 'goal_agent_intervention': intervention,
                            'goal_set_cost': set_scores.tolist() if set_scores is not None else None,
                            'matched_goal_indices': matched_goals.tolist() if matched_goals is not None else None})
                    record['arms'][arm] = {'diagnostic_success': bool(success),
                        'realized_reset_state': trace[0].tolist(),
                        'state_after_warmup': trace[10].tolist(),
                        'initial_success': bool(initial_success), 'executed_actions': len(prefix)-10,
                        'final_xy_error': xy, 'final_angle_error': angle, 'decisions': decisions}
                    print('ORACLE_CLOSED_LOOP', qi, arm, success, xy, angle, flush=True)
                    print('CANDIDATE_DIAGNOSTIC', qi, arm, json.dumps({
                        'decisions': len(decisions),
                        'turns_with_improving_candidate': sum(bool(d['physical_diagnostic']['improving_candidates']) for d in decisions),
                        'learned_selection_dominated': sum(bool(d['physical_diagnostic']['selections']['learned']['dominating_candidates']) for d in decisions),
                        'oracle_selection_dominated': sum(bool(d['physical_diagnostic']['selections']['oracle']['dominating_candidates']) for d in decisions),
                        'turns_with_branch_success': sum(d['physical_diagnostic']['any_branch_reaches_threshold'] for d in decisions)
                    }), flush=True)
                report['queries'].append(record)
                (args.out / 'progress.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    finally:
        env.close()
    if reference is not None:
        from additional_files.control_relevance_audit.goal_agent_intervention import verify_baseline
        verify_baseline(report, reference)
        print('BASELINE_REPLAY_MATCHED', args.reference_report, flush=True)
    if args.goal_agent_audit:
        report['protocol']['goal_agent_intervention'] = {
            'offsets_px': [[0,0],[-32,0],[32,0],[0,-32],[0,32]],
            'selection': 'goal_set arm minimizes learned endpoint distance over five goals; learned/oracle arms retain original goal',
            'goal_set_policy_enabled': args.goal_set_policy,
            'limitations': ['Image interventions need not be dynamically reachable goal states',
                'Agent motion can change object occlusion; inspect saved target renders',
                'Ranking sensitivity does not establish long-term control inferiority or novelty']}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('ORACLE_CLOSED_LOOP_COMPLETE', args.out / 'report.json', flush=True)


if __name__ == '__main__':
    main()
