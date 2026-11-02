"""Goal-image agent-position sensitivity, evaluated without changing policy actions."""
import numpy as np


def render_goal_variants(env, snapshot):
    before = snapshot().copy()
    original = np.asarray(env.render()).copy()
    deltas = [(0, 0), (-32, 0), (32, 0), (0, -32), (0, 32)]
    labels = ['original', 'left32', 'right32', 'up32', 'down32']
    images, records = [], []
    try:
        for label, delta in zip(labels, deltas):
            position = before[:2] + delta
            # This audit is restricted to PushT's 512 x 512 workspace.
            if not np.all((position >= 16) & (position <= 496)):
                raise ValueError('Goal-agent intervention leaves interior workspace')
            env.agent.position = tuple(position)
            after = snapshot()
            np.testing.assert_allclose(after[:2], position, atol=1e-8, rtol=0)
            np.testing.assert_array_equal(after[2:], before[2:])
            frame = np.asarray(env.render()).copy()
            np.testing.assert_array_equal(snapshot(), after)
            changed = int(np.count_nonzero(np.any(frame != original, axis=-1)))
            if label != 'original' and changed == 0:
                raise ValueError('Agent intervention did not change rendered image')
            images.append(frame)
            records.append({'label': label, 'agent_xy': position.tolist(), 'changed_pixels': changed})
    finally:
        env.agent.position = tuple(before[:2])
        np.testing.assert_array_equal(snapshot(), before)
        np.testing.assert_array_equal(np.asarray(env.render()), original)
    return images, records


def rank_variants(cost_matrix, endpoint_errors):
    costs = np.asarray(cost_matrix, dtype=float)
    errors = np.asarray(endpoint_errors, dtype=float)
    if costs.ndim != 2 or costs.shape[0] != 5 or costs.shape[1] < 2 or errors.shape != (costs.shape[1], 2):
        raise ValueError('Expected five goal variants and candidate xy/angle errors')
    if not np.isfinite(costs).all() or not np.isfinite(errors).all():
        raise ValueError('Nonfinite ranking data')
    selected = costs.argmin(axis=1)
    baseline = int(selected[0])
    # Avoid interpreting numerical tie flips as strong sensitivity.
    tolerance = 1e-5 * np.maximum(1., np.abs(costs).max(axis=1))
    gain = costs[:, baseline] - costs[np.arange(5), selected]
    return {'selected': selected.tolist(), 'flips': (selected != baseline).tolist(),
        'robust_flips': ((selected != baseline) & (gain > tolerance)).tolist(),
        'advantage_over_original_choice': gain.tolist(), 'tie_tolerance': tolerance.tolist(),
        'selected_physical_errors': errors[selected].tolist(), 'costs': costs.tolist()}


def verify_baseline(report, reference):
    if len(report['queries']) != len(reference['queries']):
        raise AssertionError('Query count changed')
    for query, old in zip(report['queries'], reference['queries']):
        assert query['row'] == old['row'], 'Query selection changed'
        for arm, value in query['arms'].items():
            previous = old['arms'][arm]
            assert value['diagnostic_success'] == previous['diagnostic_success']
            np.testing.assert_allclose([value['final_xy_error'], value['final_angle_error']],
                [previous['final_xy_error'], previous['final_angle_error']], atol=1e-5, rtol=0)
            assert len(value['decisions']) == len(previous['decisions'])
            for decision, before in zip(value['decisions'], previous['decisions']):
                assert decision['selected'] == before['selected'], 'Baseline decision changed'
                for key in ('candidate_actions', 'candidate_state_traces', 'current_state'):
                    np.testing.assert_allclose(decision[key], before[key], atol=1e-5, rtol=0)
                for key in ('learned_cost', 'oracle_cost'):
                    np.testing.assert_allclose(decision[key], before[key], atol=1e-4, rtol=1e-5)
