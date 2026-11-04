"""Read-only, standard-library audit of saved oracle_closed_loop reports.

No refitting, new simulator rollouts, or weights between xy and angle errors.
Results concern fixed candidate banks and immediate endpoints, not optimal control.
"""
import argparse
import json
import math
from pathlib import Path


def dominates(a, b):
    tol = (1e-5, 1e-7)
    return all(x <= y + t for x, y, t in zip(a, b, tol)) and any(
        x < y - t for x, y, t in zip(a, b, tol))


def audit_decision(d, goal):
    learned, actual = d['learned_cost'], d['oracle_cost']
    traces = d['candidate_state_traces']
    n = len(traces)
    if n < 2 or len(learned) != n or len(actual) != n:
        raise ValueError('Candidate count mismatch')
    if len(goal) < 5 or not all(math.isfinite(x) for x in goal):
        raise ValueError('Invalid goal')
    errors = []
    for trace in traces:
        if not trace or any(len(s) != 10 or not all(math.isfinite(x) for x in s) for s in trace):
            raise ValueError('Expected finite full-state branch traces')
        s = trace[-1]
        angle = s[4] - goal[4]
        errors.append((math.hypot(s[2]-goal[2], s[3]-goal[3]),
                       abs(math.atan2(math.sin(angle), math.cos(angle)))))
    if not all(math.isfinite(x) for x in learned + actual):
        raise ValueError('Nonfinite cost')
    li, oi = min(range(n), key=learned.__getitem__), min(range(n), key=actual.__getitem__)
    # Exclude floating-point ties from prediction-induced ranking changes.
    regret = actual[li] - actual[oi]
    ranking_error = regret > 1e-6 + 1e-5 * max(abs(actual[li]), abs(actual[oi]))
    ld = any(dominates(e, errors[li]) for e in errors)
    od = any(dominates(e, errors[oi]) for e in errors)
    return {'learned_choice': li, 'real_endpoint_latent_choice': oi,
            'prediction_changes_choice_beyond_tie': ranking_error,
            'real_latent_score_regret': regret,
            'learned_choice_physically_dominated': ld,
            'real_endpoint_latent_choice_physically_dominated': od,
            'oracle_physically_dominates_learned': dominates(errors[oi], errors[li]),
            'learned_physically_dominates_oracle': dominates(errors[li], errors[oi]),
            'prediction_and_score_flags': bool(ranking_error and od),
            'any_candidate_endpoint_success': any(x <= 10 and a <= .1 for x, a in errors),
            'endpoint_errors': errors}


def summarize(rows):
    keys = ('prediction_changes_choice_beyond_tie', 'learned_choice_physically_dominated',
            'real_endpoint_latent_choice_physically_dominated',
            'oracle_physically_dominates_learned', 'learned_physically_dominates_oracle',
            'prediction_and_score_flags', 'any_candidate_endpoint_success')
    return {'decisions': len(rows), **{k: sum(bool(r[k]) for r in rows) for k in keys}}


def audit_report(report):
    rows = []
    for qi, q in enumerate(report['queries']):
        # Use one factual history per query: the learned arm. Oracle arm histories
        # diverge after acting and must not be counted as independent matched cases.
        for d in q['arms']['learned']['decisions']:
            rows.append({'query': qi, 'episode': q['episode'], 'turn': d['turn'],
                         **audit_decision(d, q['goal'])})
    return {'summary': summarize(rows),
            'first_decision_summary': summarize([r for r in rows if r['turn'] == 0]),
            'unique_episodes': len({r['episode'] for r in rows}), 'rows': rows,
            'scope': ['Diagnostic on saved learned-arm histories only; no new policy evaluation',
                      'Prediction flag measures score ordering, not physical harm by itself',
                      'Real-endpoint score flag is short-horizon xy/angle Pareto dominance, not proof of long-horizon failure',
                      'Flags overlap; no additive causal decomposition',
                      'No endpoint success does not establish task unreachability',
                      'Turns and reports may share episodes; do not treat them as independent samples']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output already exists; choose a new path')
    results = []
    for path in a.reports:
        result = audit_report(json.loads(path.read_text(encoding='utf-8')))
        result['source'] = str(path)
        results.append(result)
        print('CANDIDATE_ATTRIBUTION', path, json.dumps(result['summary']), flush=True)
        print('FIRST_DECISIONS', json.dumps(result['first_decision_summary']), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print('CANDIDATE_ATTRIBUTION_COMPLETE', a.out)


if __name__ == '__main__':
    main()
