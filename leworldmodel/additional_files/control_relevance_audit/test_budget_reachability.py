import unittest
import numpy as np
from budget_reachability import first_hit_targets, hazard_to_cdf


class BudgetReachabilityTests(unittest.TestCase):
    def test_late_hit_is_not_short_budget_success(self):
        y, m = first_hit_targets(np.array([[0, 0, 1]], bool),
                                 np.ones((1, 3), bool), [0, 1, 2, 5])
        np.testing.assert_array_equal(y, [[0, 0, 1, 1]])
        self.assertTrue(m.all())

    def test_censored_failure_is_unknown(self):
        y, m = first_hit_targets(np.zeros((1, 3), bool),
                                 np.array([[1, 1, 0]], bool), [0, 1, 2, 5])
        self.assertFalse(y.any())
        np.testing.assert_array_equal(m, [[1, 1, 0, 0]])

    def test_hit_persists_after_leaving_goal(self):
        y, m = first_hit_targets(np.array([[1, 0, 0]], bool),
                                 np.ones((1, 3), bool), [0, 1, 2])
        self.assertTrue(y.all() and m.all())

    def test_holes_rejected(self):
        with self.assertRaises(ValueError):
            first_hit_targets(np.zeros((1, 3), bool),
                              np.array([[1, 0, 1]], bool), [2])

    def test_cdf_monotone_and_stable(self):
        p = hazard_to_cdf(np.array([[-1000, 0, 1000], [0, 0, 0]]))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((np.diff(p, axis=1) >= 0).all())
        np.testing.assert_allclose(p[1], [.5, .75, .875])


if __name__ == '__main__':
    unittest.main()
