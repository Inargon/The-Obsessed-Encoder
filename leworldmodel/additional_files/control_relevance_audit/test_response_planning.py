import unittest
import numpy as np
from response_planning import basis, encode, decode, fit, decision_metrics


class PlanningTests(unittest.TestCase):
    def test_invertible_and_equal_loss(self):
        y = np.random.default_rng(4).normal(size=(12, 3, 4))
        b = basis(3)
        np.testing.assert_allclose(b.T @ b, np.eye(3), atol=1e-14)
        for w in (np.ones(3), np.array([1, 2, 2])):
            np.testing.assert_allclose(decode(encode(y, b, w), b, w), y, atol=1e-14)
        self.assertAlmostEqual(np.mean(y ** 2), np.mean(encode(y, b, np.ones(3)) ** 2))

    def test_regret(self):
        y = np.array([[[0., 0., 0., 30.], [10., 0., 0., 30.]]])
        goals = np.array([[10., 0., 0.]])
        self.assertEqual(decision_metrics(y, y, goals)['mean_candidate_regret'], 0)
        self.assertEqual(decision_metrics(y[:, ::-1], y, goals)['mean_candidate_regret'], 100)

    def test_no_test_target_leak_and_capacity(self):
        rng = np.random.default_rng(8)
        x, y, g = rng.normal(size=(40, 8)), rng.normal(size=(40, 3, 4)), rng.normal(size=(40, 3))
        splits = np.arange(24), np.arange(24, 32), np.arange(32, 40)
        counts = []
        for mode in ('endpoint', 'decomposed', 'contrast_x2'):
            a = fit(x, y, g, splits, mode, 1, 20, 8)
            changed = y.copy()
            changed[32:, :, :2] += 50
            b = fit(x, changed, g, splits, mode, 1, 20, 8)
            for key in ('selected_epoch', 'validation_candidate_regret', 'target_scale'):
                self.assertEqual(a[key], b[key])
            self.assertNotEqual(a['xy_rmse_px'], b['xy_rmse_px'])
            counts.append(a['parameters'])
        self.assertEqual(len(set(counts)), 1)


if __name__ == '__main__':
    unittest.main()
