"""Matched from-scratch cycle on/off; existing gradient router remains enabled.

Cycle also contributes to the router guide: this ablates its total contribution,
not its predictor-only causal effect. Run names require a fresh campaign ID.
"""
import copy
import os
import re


def make_pair(config, campaign):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', campaign):
        raise ValueError('Campaign ID must contain letters, numbers, underscores or hyphens')
    config = copy.deepcopy(config)
    if len(config['arms']) != 1: raise ValueError('Expected one matched Ours arm')
    template = next(iter(config['arms'].values()))
    key = '+loss.control.cycle_weight='
    found = [v for v in template['overrides'] if v.startswith(key)]
    if found != [key + '0.5']: raise ValueError('Unexpected source cycle weight')
    arms = {}
    for label, weight in [('on', '0.5'), ('off', '0.0')]:
        arm = copy.deepcopy(template)
        arm['overrides'] = [key+weight if v.startswith(key) else v for v in arm['overrides']]
        arms[f'cycle_{label}_{campaign}'] = arm
    config['arms'] = arms
    return config


def load_configs():
    if __package__:
        from .run_pusht_clean_ours_matched import load_pusht_clean_ours_matched_configs
    else:
        from run_pusht_clean_ours_matched import load_pusht_clean_ours_matched_configs
    return make_pair(load_pusht_clean_ours_matched_configs(), os.environ['CYCLE_CAMPAIGN'])


def main():
    if __package__:
        from . import run as base
    else:
        import run as base
    base.CASE = 'cycle-ablation'
    base.load_configs = load_configs
    base.render_figures = lambda args: print('Figures skipped; evaluate matched checkpoints separately')
    return base.main()


if __name__ == '__main__': raise SystemExit(main())
