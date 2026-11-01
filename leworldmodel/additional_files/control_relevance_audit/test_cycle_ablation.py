import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from leworldmodel.additional_files.run_cycle_ablation import make_pair
import importlib.util


def fixture():
    return {'arms': {'ours': {'overrides': ['+loss.control.cycle_weight=0.5',
        '+loss.aligned_gradient_routing.enabled=true'], 'eval_overrides': []}}}


class CycleAblationTests(unittest.TestCase):
    def check_pair(self, original):
        pair = make_pair(original, 'smoke123')
        on, off = pair['arms'].values()
        differences = [(a, b) for a, b in zip(on['overrides'], off['overrides']) if a != b]
        self.assertEqual(differences, [('+loss.control.cycle_weight=0.5', '+loss.control.cycle_weight=0.0')])
        self.assertEqual(len(on['overrides']), len(off['overrides']))
        self.assertIn('+loss.aligned_gradient_routing.enabled=true', off['overrides'])
        self.assertFalse(any('pixel_tag' in x for x in off['overrides']))
        self.assertEqual(on['eval_overrides'], off['eval_overrides'])
        self.assertIn('+loss.control.cycle_weight=0.5', next(iter(original['arms'].values()))['overrides'])

    def test_configuration_changes_only_cycle_weight(self):
        self.check_pair(fixture())

    @unittest.skipUnless(importlib.util.find_spec('yaml'), 'Full config needs cluster PyYAML')
    def test_real_configuration(self):
        from leworldmodel.additional_files.run_pusht_clean_ours_matched import load_pusht_clean_ours_matched_configs
        self.check_pair(load_pusht_clean_ours_matched_configs())

    def test_campaign_names_isolated(self):
        c = fixture()
        self.assertFalse(set(make_pair(c, 'job1')['arms']) & set(make_pair(c, 'job2')['arms']))
        with self.assertRaises(ValueError): make_pair(c, '../old')


if __name__ == '__main__': unittest.main()
