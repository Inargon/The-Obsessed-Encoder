import unittest
import numpy as np
from budget_data import episode_runs, pose_hits, build_split


class BudgetDataTests(unittest.TestCase):
    def test_discontinuous_steps_and_recurring_ids_rejected(self):
        for ids, steps in [([0,0,0], [0,1,3]), ([0,1,0], [0,0,1])]:
            with self.assertRaises(ValueError):
                episode_runs(ids, steps)

    def test_periodic_goal_and_agent_independence(self):
        state = np.zeros((2,7)); state[:,4] = 2*np.pi-.01
        state[1, :2] = 1000
        np.testing.assert_array_equal(pose_hits(state, [0,0,.01]), [True,True])

    def test_split_membership_censoring_and_no_cross_episode(self):
        state = np.zeros((30,7))
        state[:10,2] = 0; state[10:20,2] = 100; state[20:,2] = 200
        runs = episode_runs(np.repeat([0,1,2],10), np.tile(np.arange(10),3))
        x, y, m, rows = build_split(state, [runs[0],runs[2]], 100, 4, [0,20])
        self.assertFalse(np.any((rows >= 10)&(rows < 20)))
        same_episode = rows[:,0]//10 == rows[:,1]//10
        np.testing.assert_array_equal(y[:,0], same_episode)
        np.testing.assert_array_equal(y[:,1], same_episode)
        # End-of-episode non-hits are unknown at budget 20, not negatives.
        np.testing.assert_array_equal(m[:,1], same_episode)
        self.assertTrue(m[:,0].all())
        self.assertEqual(x.shape, (100,28))


if __name__ == '__main__':
    unittest.main()
