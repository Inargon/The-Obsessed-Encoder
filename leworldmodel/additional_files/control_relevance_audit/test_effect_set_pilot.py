import unittest
import numpy as np
from audit import split_episodes
from effect_set_pilot import MLP, costs_to_goal, outcome_targets, fit_arm, metrics


class EffectSetTests(unittest.TestCase):
    def test_equivalent_actions_share_probability(self):
        poses = np.array([[[0., 0., 0.], [0., 0., 2*np.pi], [10., 0., 0.]]])
        q, _ = outcome_targets(poses, np.array([0]))
        self.assertAlmostEqual(q[0, 0, 0], q[0, 0, 1])
        self.assertGreater(q[0, 0, 0], q[0, 0, 2])

    def test_test_outcomes_do_not_change_training_targets(self):
        poses = np.arange(54, dtype=float).reshape(6, 3, 3)
        q, t = outcome_targets(poses, np.array([0, 1]))
        poses[2:] *= 1000
        q2, t2 = outcome_targets(poses, np.array([0, 1]))
        self.assertEqual(t, t2)
        np.testing.assert_array_equal(q[:2], q2[:2])

    def test_pair_split_no_leakage(self):
        ids = np.repeat(np.arange(12), 2)
        sets = [set(ids[ix]) for ix in split_episodes(ids, 73)]
        self.assertFalse(sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])

    def test_ties_not_counted_as_failure(self):
        final = np.zeros((2, 3, 3))
        result = metrics(final, np.zeros((2, 1, 3)), np.array([[[1, 0, 2]], [[2, 1, 0]]]), np.array([0, 0]))
        self.assertEqual(result['mean_candidate_regret'], 0)
        self.assertEqual(result['optimal_within_1e_6_fraction'], 1)

    def test_optimizer_learns_regression(self):
        x = np.linspace(-1, 1, 32)[:, None]
        model = MLP(1, 1, 12, 1)
        initial = np.mean((model.predict(x) - 2*x)**2)
        for _ in range(150):
            model.update(x, 2*x, False, .01)
        self.assertLess(np.mean((model.predict(x) - 2*x)**2), initial/10)

    def test_all_arms_finite_and_test_outcomes_not_used(self):
        rng = np.random.default_rng(2)
        state = rng.normal(size=(24, 8))
        final = rng.normal(size=(24, 3, 3))
        splits = split_episodes(np.repeat(np.arange(12), 2), 73)
        te = splits[-1]
        goal = np.zeros((len(te), 1, 3))
        modified = final.copy()
        modified[te] += 1000
        for mode in ('identity', 'forward', 'effect_set'):
            pred, _ = fit_arm(state, final, splits, mode, 7, 20, 8)
            pred2, _ = fit_arm(state, modified, splits, mode, 7, 20, 8)
            a, b = pred(te, goal), pred2(te, goal)
            self.assertEqual(a.shape, (len(te), 1, 3))
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_allclose(a, b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
