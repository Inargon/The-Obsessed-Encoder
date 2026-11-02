import unittest
import numpy as np
from oracle_rescue import oracle_inputs
from state_terminal_readout import encode_pose
from audit import split_episodes


class OracleRescueTests(unittest.TestCase):
    def setup_data(self):
        s = np.arange(168, dtype=float).reshape(24, 7)
        ids = np.repeat(np.arange(12), 2)
        return s, ids, split_episodes(ids, 73)

    def test_addons_correct_and_capacity_matched(self):
        s, ids, splits = self.setup_data()
        x = np.ones((24, 9))
        out = oracle_inputs({'ours': x}, s, ids, splits, 1)
        self.assertEqual({a.shape for a in out.values()}, {(24, 13)})
        np.testing.assert_array_equal(out['ours+block'][:, -4:], encode_pose(s[:, 2:5]))
        np.testing.assert_array_equal(out['ours+agent'][:, -4:], s[:, [0, 1, 5, 6]])
        np.testing.assert_array_equal(out['ours+zero'][:, -4:], np.zeros((24, 4)))
        for a in out.values(): np.testing.assert_array_equal(a[:, :9], x)

    def test_shuffle_stays_within_split_and_moves_pairs(self):
        s, ids, splits = self.setup_data()
        original = encode_pose(s[:, 2:5])
        out = oracle_inputs({'ours': np.ones((24, 2))}, s, ids, splits, 1)['ours+shuffled_block'][:, -4:]
        for rows in splits:
            np.testing.assert_array_equal(np.sort(out[rows, 0]), np.sort(original[rows, 0]))
            self.assertTrue(np.all(out[rows, 0] != original[rows, 0]))
            for pair in np.unique(ids[rows]):
                r = np.flatnonzero(ids == pair)
                self.assertEqual(out[r[1], 0] - out[r[0], 0], 7)

    def test_heldout_state_changes_do_not_change_training_inputs(self):
        s, ids, splits = self.setup_data()
        features = {'ours': np.ones((24, 2))}
        out = oracle_inputs(features, s, ids, splits, 1)
        s[splits[-1]] += 5000
        other = oracle_inputs(features, s, ids, splits, 1)
        for name in out:
            np.testing.assert_array_equal(out[name][splits[0]], other[name][splits[0]])

    def test_split_pairs_rejected(self):
        s, ids, splits = self.setup_data()
        a, b = splits[0][0], splits[1][0]
        splits[0][0], splits[1][0] = b, a
        with self.assertRaises(ValueError):
            oracle_inputs({'ours': np.ones((24, 2))}, s, ids, splits, 1)


if __name__ == '__main__': unittest.main()
