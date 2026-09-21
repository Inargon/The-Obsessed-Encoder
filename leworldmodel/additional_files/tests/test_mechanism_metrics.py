import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('mechanism_metrics',Path(__file__).parents[1]/'mechanism_metrics.py')
metrics=importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class TestMetrics(unittest.TestCase):
    def test_normalize_frameskip_five_native_two(self):
        raw = np.arange(80, dtype=float).reshape(8, 10)
        mean, std = np.array([1., 3.]), np.array([2., 4.])
        seen = []
        def normalizer(sample):
            seen.append(sample["action"].shape)
            return {"action": (sample["action"] - mean) / std}
        actual = metrics.normalize_action_blocks(raw, normalizer, 2)
        self.assertEqual(seen, [(40, 2)])
        self.assertEqual(actual.shape, (8, 10))
        np.testing.assert_allclose(actual, (raw - np.tile(mean, 5)) / np.tile(std, 5))

    def test_normalize_unpacked_actions(self):
        raw = np.arange(8, dtype=float).reshape(4, 2)
        actual = metrics.normalize_action_blocks(raw, lambda d: {"action": d["action"] + 1}, 2)
        np.testing.assert_array_equal(actual, raw + 1)

    def test_invalid_packed_action_width(self):
        with self.assertRaises(ValueError):
            metrics.normalize_action_blocks(np.ones((4, 5)), lambda d: d, 2)

    def test_identical_candidates(self):
        costs=np.array([[1.,2.,3.]])
        actions=np.array([[[0.,0.],[1.,0.],[0.,1.]]])
        out=metrics.cost_changes(costs,costs,actions)
        for k in ('cost_rmse','pair_order_reversal','selected_candidate_changed','selected_action_normalized_l2'):
            self.assertEqual(out[k][0],0)

    def test_reversal_and_selected_action(self):
        costs=np.array([[1.,2.,3.]])
        actions=np.array([[[0.,0.],[1.,0.],[0.,2.]]])
        out=metrics.cost_changes(costs,-costs,actions)
        self.assertEqual(out['pair_order_reversal'][0],1)
        self.assertEqual(out['selected_action_normalized_l2'][0],2)

    def test_constant_shift_changes_cost_not_choice(self):
        out=metrics.cost_changes(np.array([[1.,2.]]),np.array([[6.,7.]]),np.zeros((1,2,2)))
        self.assertEqual(out['cost_rmse'][0],5)
        self.assertEqual(out['selected_candidate_changed'][0],0)

    def test_changed_action_bank_is_used_for_selected_action_distance(self):
        costs=np.array([[1.,2.]])
        original=np.array([[[0.,0.],[1.,0.]]])
        changed=np.array([[[0.,2.],[1.,2.]]])
        out=metrics.cost_changes(
            costs, costs, original, changed_first_actions=changed
        )
        self.assertEqual(out['selected_candidate_changed'][0],0)
        self.assertEqual(out['selected_action_normalized_l2'][0],2)

    def test_ties_are_reported(self):
        out=metrics.cost_changes(np.ones((1,3)),np.ones((1,3)),np.zeros((1,3,2)))
        self.assertEqual(out['reference_cost_degenerate'][0],1)
        self.assertEqual(out['comparable_pair_fraction'][0],0)

    def test_shape_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            metrics.cost_changes(np.ones((1,3)),np.ones((2,3)),np.zeros((1,3,2)))
        with self.assertRaises(ValueError):
            metrics.cost_changes(
                np.ones((1,3)), np.ones((1,3)), np.zeros((1,3,2)),
                changed_first_actions=np.zeros((1,2,2)),
            )
        with self.assertRaises(ValueError): metrics.summarize([float('nan')])

    def test_bootstrap_constant(self):
        self.assertEqual(metrics.summarize([.4,.4,.4])['bootstrap_clip_ci95'],[.4000000000000001]*2)


if __name__=='__main__': unittest.main()
