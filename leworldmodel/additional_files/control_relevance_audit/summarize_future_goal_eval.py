"""Verify matched evaluation conditions before reporting paired outcomes."""
import argparse
import json
from pathlib import Path


def compare(a, b):
    for key in ('seed', 'num_eval', 'episodes', 'start_steps', 'evaluation_config', 'checkpoint'):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError('Unmatched or missing evaluation field: ' + key)
    aa, bb = a['episode_successes'], b['episode_successes']
    if len(aa) != a['num_eval'] or len(bb) != len(aa) or not aa:
        raise ValueError('Invalid per-task outcomes')
    if any(type(x) is not bool for x in aa + bb):
        raise ValueError('Expected explicit Boolean success values')
    n = len(aa)
    return {'tasks': n, 'detached_successes': sum(aa), 'joint_successes': sum(bb),
            'detached_success_rate': sum(aa)/n, 'joint_success_rate': sum(bb)/n,
            'difference_percentage_points': 100*(sum(bb)-sum(aa))/n,
            'joint_only_success': sum(y and not x for x, y in zip(aa, bb)),
            'detached_only_success': sum(x and not y for x, y in zip(aa, bb)),
            'both_success': sum(x and y for x, y in zip(aa, bb)),
            'both_fail': sum(not x and not y for x, y in zip(aa, bb)),
            'unique_episode_start_pairs': len(set(zip(a['episodes'], a['start_steps']))),
            'unique_episodes': len(set(a['episodes'])),
            'scope': 'One training seed, one evaluation set from training dataset; not a novelty or generalization claim'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    a, b = [json.loads((args.directory / (arm + '.json')).read_text()) for arm in ('detached', 'joint')]
    result = compare(a, b)
    (args.directory / 'paired-summary.json').write_text(json.dumps(result, indent=2))
    print('FUTURE_GOAL_PAIRED_RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
