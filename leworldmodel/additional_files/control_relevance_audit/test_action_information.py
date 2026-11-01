import unittest
import numpy as np
from action_information import evaluate, fit, predict
from export_action_information import verify_window


class ActionInformationTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(12)
        action = rng.normal(size=(240, 2))
        return dict(history=rng.normal(size=(240, 3)), past_actions=rng.normal(size=(240, 2)),
                    real=np.zeros((240, 2)), predicted=action.copy(), action=action,
                    episode_id=np.repeat(np.arange(40), 6))

    def test_synthetic_action_channel_detected_without_claiming_physics(self):
        result = evaluate(self.data())
        self.assertGreater(result['readouts']['predicted_endpoint']['r2'], .99)
        self.assertGreater(result['readouts']['residual']['r2'], .99)
        self.assertGreater(result['readouts']['real_plus_residual']['r2'], .99)
        self.assertLess(result['readouts']['real_plus_shuffled_residual']['r2'], .1)
        self.assertLess(result['readouts']['real_endpoint']['r2'], .1)
        self.assertLess(result['readouts']['real_endpoint']['transfer_same_frozen_readout']['r2'], .1)

    def test_equal_endpoints_transfer_identical(self):
        data = self.data(); data['real'] = data['predicted'].copy()
        r = evaluate(data)['readouts']['real_endpoint']
        self.assertEqual(r['r2'], r['transfer_same_frozen_readout']['r2'])
        result = evaluate(data)
        self.assertAlmostEqual(r['r2'], result['readouts']['real_plus_residual']['r2'])

    def test_holdout_does_not_change_fit(self):
        d = self.data(); train = np.arange(100)
        a = fit(d['history'], d['action'], train, 1.)
        d['history'][100:] += 10000; d['action'][100:] -= 10000
        b = fit(d['history'], d['action'], train, 1.)
        for x, y in zip(a, b): np.testing.assert_array_equal(x, y)

    def test_episode_disjoint(self):
        a, b, c = map(set, evaluate(self.data())['episodes'])
        self.assertFalse(a & b or a & c or b & c)

    def test_invalid_endpoint_rejected(self):
        data = self.data(); data['real'][0, 0] = np.nan
        with self.assertRaises(ValueError): evaluate(data)

    def test_raw_window_alignment_and_shift_rejection(self):
        state = np.arange(210).reshape(30, 7)
        action = np.arange(60).reshape(30, 2)
        item = dict(state=state[2:22:5], action=action[2:22].reshape(4, 10))
        verify_window(item, state, action, 2)
        with self.assertRaises(AssertionError): verify_window(item, state, action, 3)

    def test_action_reordering_rejected(self):
        state = np.arange(210).reshape(30, 7)
        action = np.arange(60).reshape(30, 2)
        item = dict(state=state[:20:5], action=action[:20][::-1].reshape(4, 10))
        with self.assertRaises(AssertionError): verify_window(item, state, action, 0)


if __name__ == '__main__': unittest.main()
