import copy
import unittest
from summarize_future_goal_eval import compare


class EvalTests(unittest.TestCase):
    def fixture(self):
        return dict(seed=42, num_eval=4, episodes=[1, 2, 3, 4], start_steps=[0]*4,
                    evaluation_config={'budget': 50}, checkpoint='weights_step_10000.pt',
                    episode_successes=[True, False, True, False])

    def test_paired_outcomes(self):
        a = self.fixture(); b = copy.deepcopy(a)
        b['episode_successes'] = [False, True, True, True]
        r = compare(a, b)
        self.assertEqual(r['difference_percentage_points'], 25)
        self.assertEqual(r['joint_only_success'], 2)
        self.assertEqual(r['detached_only_success'], 1)

    def test_reject_mismatched_tasks(self):
        a = self.fixture(); b = copy.deepcopy(a)
        b['start_steps'][0] = 1
        with self.assertRaises(ValueError): compare(a, b)


if __name__ == '__main__': unittest.main()
