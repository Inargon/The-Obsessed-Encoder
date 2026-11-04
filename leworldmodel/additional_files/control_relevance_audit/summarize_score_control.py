"""Paired query-level results, never treating decision turns as independent."""
import numpy as np


def summarize(report):
    queries = report['queries']
    if not queries:
        raise ValueError('No queries')
    episodes = [q['episode'] for q in queries]
    if len(set(episodes)) != len(episodes):
        raise ValueError('Repeated query episodes')
    result = {'queries': len(queries), 'arms': {}, 'paired': {},
              'scope': 'Fixed candidate bank; simulator-assisted diagnostic, not deployable or official PushT SR. New query episodes may belong to encoder training data.'}
    for arm in ('learned', 'oracle', 'physical'):
        rows = [q['arms'][arm] for q in queries]
        result['arms'][arm] = {
            'successes': sum(r['diagnostic_success'] for r in rows),
            'initial_successes': sum(r['initial_success'] for r in rows),
            'mean_final_xy': float(np.mean([r['final_xy_error'] for r in rows])),
            'mean_final_angle': float(np.mean([r['final_angle_error'] for r in rows]))}
    for base, alternative in (('learned', 'oracle'), ('oracle', 'physical'), ('learned', 'physical')):
        a = [q['arms'][base] for q in queries]
        b = [q['arms'][alternative] for q in queries]
        result['paired'][base + '_to_' + alternative] = {
            'gained_successes': sum(not x['diagnostic_success'] and y['diagnostic_success'] for x, y in zip(a, b)),
            'lost_successes': sum(x['diagnostic_success'] and not y['diagnostic_success'] for x, y in zip(a, b))}
    return result
