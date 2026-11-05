import copy
import unittest
from replay_manifest import compare_replay, failure_manifest, verify_reference


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.report = dict(run_name='model', checkpoint='weights', checkpoint_sha256='abc',
                           tagged=False, num_eval=3, seed=42, episodes=[9, 9, 11],
                           start_steps=[0, 5, 0], evaluation_config={'budget': 50},
                           episode_successes=[True, False, False])

    def test_task_indices_not_episode_ids(self):
        self.assertEqual(failure_manifest(self.report), [
            dict(task_index=1, episode=9, start_step=5),
            dict(task_index=2, episode=11, start_step=0)])

    def test_protocol_changes_rejected(self):
        for key, value in [('checkpoint_sha256', 'other'), ('start_steps', [1, 5, 0]),
                           ('evaluation_config', {'budget': 10})]:
            changed = copy.deepcopy(self.report)
            changed[key] = value
            with self.assertRaises(ValueError):
                verify_reference(self.report, changed)

    def test_flips_retained(self):
        changed = copy.deepcopy(self.report)
        changed['episode_successes'] = [False, True, False]
        result = compare_replay(self.report, changed)
        self.assertEqual(result['persistent_failures'], [2])
        self.assertEqual(result['failure_to_success'], [1])
        self.assertEqual(result['success_to_failure'], [0])

    def test_truncation_rejected(self):
        self.report['episode_successes'] = [True]
        with self.assertRaises(ValueError):
            failure_manifest(self.report)


if __name__ == '__main__':
    unittest.main()
