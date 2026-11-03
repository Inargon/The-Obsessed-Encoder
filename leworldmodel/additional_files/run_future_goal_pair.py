"""Matched JEPA training: future-goal head-only gradients vs encoder gradients.

Both arms train the base JEPA normally; 'detached' does not freeze the base model.
Same architecture, head initialization, data and original planning objective.
"""
import copy
import os
import re


def make_pair(config, campaign):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', campaign):
        raise ValueError('Invalid campaign name')
    config = copy.deepcopy(config)
    if len(config['arms']) != 1:
        raise ValueError('Expected one clean baseline')
    template = next(iter(config['arms'].values()))
    arms = {}
    for name, detach in [('detached', 'true'), ('joint', 'false')]:
        arm = copy.deepcopy(template)
        if any('loss.future_goal' in value for value in arm['overrides']):
            raise ValueError('Already contains future goal configuration')
        arm['overrides'] += ['+loss.future_goal.enabled=true',
            '+loss.future_goal.max_horizon=3', '+loss.future_goal.hidden_dim=128',
            '+loss.future_goal.weight=0.1', '+loss.future_goal.temperature=0.1',
            '+loss.future_goal.detach_representation=' + detach]
        arms[f'future_goal_{name}_{campaign}'] = arm
    config['arms'] = arms
    return config


def main():
    if __package__:
        from . import run as base
        from .run_pusht_clean_jepa_matched import load_pusht_clean_jepa_matched_configs
    else:
        import run as base
        from run_pusht_clean_jepa_matched import load_pusht_clean_jepa_matched_configs
    config = make_pair(load_pusht_clean_jepa_matched_configs(), os.environ['FUTURE_GOAL_CAMPAIGN'])
    base.CASE = 'future-goal-pair'
    base.load_configs = lambda: config
    base.render_figures = lambda args: print('Evaluate matched checkpoints with unchanged latent MPC separately')
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
