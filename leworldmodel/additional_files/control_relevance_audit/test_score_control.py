import unittest
import numpy as np
from oracle_closed_loop import physical_branch_costs, choose
from summarize_score_control import summarize


class ScoreTests(unittest.TestCase):
    def test_success_at_intermediate_step_preferred(self):
        b = np.zeros((2, 2, 10))
        b[0, :, 2] = [30, 20]
        b[1, :, 2] = [5, 100]
        self.assertEqual(choose(physical_branch_costs(b, np.zeros(7))), 1)

    def test_angle_wrap_and_agent_invariance(self):
        b = np.zeros((2, 1, 10))
        b[:, :, 4] = 2 * np.pi
        b[1, :, :2] = 500
        np.testing.assert_allclose(physical_branch_costs(b, np.zeros(7)), 0, atol=1e-12)

    def test_bottleneck_error_and_nonfinite(self):
        b = np.zeros((2, 1, 10))
        b[0, :, 2] = 30
        b[1, :, 4] = .4
        np.testing.assert_allclose(physical_branch_costs(b, np.zeros(7)), [3, 4])
        b[0, 0, 2] = np.nan
        with self.assertRaises(ValueError):
            physical_branch_costs(b, np.zeros(7))

    def test_paired_summary(self):
        def row(success):
            return dict(diagnostic_success=success, initial_success=False, final_xy_error=5., final_angle_error=.1)
        q = {'episode': 9, 'arms': {'learned': row(False), 'oracle': row(False), 'physical': row(True)}}
        result = summarize({'queries': [q]})
        self.assertEqual(result['paired']['oracle_to_physical']['gained_successes'], 1)
        with self.assertRaises(ValueError):
            summarize({'queries': [q, q]})


if __name__ == '__main__':
    unittest.main()
