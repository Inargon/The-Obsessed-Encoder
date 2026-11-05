"""Strict protocol matching and paired replay outcomes (no model dependencies)."""

def verify_reference(reference, current):
    for key in ('run_name', 'checkpoint', 'checkpoint_sha256', 'tagged',
                'num_eval', 'seed', 'episodes', 'start_steps', 'evaluation_config'):
        if key not in reference or reference[key] != current[key]:
            raise ValueError(f'Replay protocol mismatch: {key}')


def failure_manifest(report):
    episodes, starts, successes = (report[k] for k in
                                   ('episodes', 'start_steps', 'episode_successes'))
    if not (len(episodes) == len(starts) == len(successes) == report['num_eval']):
        raise ValueError('Report task lengths differ')
    if any(type(x) is not bool for x in successes):
        raise ValueError('Success flags must be booleans')
    return [dict(task_index=i, episode=ep, start_step=step)
            for i, (ep, step, success) in enumerate(zip(episodes, starts, successes))
            if not success]


def compare_replay(reference, current):
    verify_reference(reference, current)
    failure_manifest(reference)
    failure_manifest(current)
    old, new = reference['episode_successes'], current['episode_successes']
    return {
        'tasks': len(old),
        'persistent_failures': [i for i, (a, b) in enumerate(zip(old, new)) if not a and not b],
        'failure_to_success': [i for i, (a, b) in enumerate(zip(old, new)) if not a and b],
        'success_to_failure': [i for i, (a, b) in enumerate(zip(old, new)) if a and not b],
        'reference_failures': failure_manifest(reference),
        'replay_failures': failure_manifest(current),
    }
