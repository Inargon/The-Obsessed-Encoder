import unittest
import numpy as np
from oracle_closed_loop import choose, goal_error, history_indices, pack_history_actions


class OracleTests(unittest.TestCase):
    def test_history_alignment_after_replanning(self):
        for length in (11, 16, 21):
            prefix = np.arange((length-1)*2).reshape(-1, 2)
            bank = np.arange(30).reshape(3, 5, 2)
            packed = pack_history_actions(prefix, bank)
            indices = history_indices(length)
            for candidate in range(3):
                for token in range(2):
                    np.testing.assert_array_equal(packed[candidate, token].reshape(5, 2),
                                                   prefix[indices[token]:indices[token]+5])
                np.testing.assert_array_equal(packed[candidate, 2].reshape(5, 2), bank[candidate])

    def test_selection_and_nonfinite(self):
        self.assertEqual(choose([4, 1, 1]), 1)
        with self.assertRaises(ValueError):
            choose([0, np.nan])
        with self.assertRaises(ValueError):
            history_indices(10)

    def test_physical_metric_ignores_agent_and_wraps_angle(self):
        a = np.array([0, 0, 10, 20, np.pi-.01, 0, 0])
        b = np.array([100, 100, 10, 20, -np.pi+.01, 10, 10])
        xy, angle = goal_error(a, b)
        self.assertEqual(xy, 0)
        self.assertAlmostEqual(angle, .02)


if __name__ == '__main__':
    unittest.main()
