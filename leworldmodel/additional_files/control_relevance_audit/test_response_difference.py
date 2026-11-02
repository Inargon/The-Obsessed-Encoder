import unittest
import numpy as np
from response_difference import contrast, pair_differences, fit


class ResponseTests(unittest.TestCase):
    def test_common_mode_cancels(self):
        rng = np.random.default_rng(4)
        y = rng.normal(size=(20, 3, 4))
        np.testing.assert_allclose(pair_differences(y), pair_differences(contrast(y)), atol=1e-14)
        np.testing.assert_allclose(pair_differences(y),
                                   pair_differences(y + rng.normal(size=(20, 1, 4))), atol=1e-14)

    def test_context_changes_response(self):
        y = np.array([[[-1.], [1.]], [[1.], [-1.]]])
        np.testing.assert_array_equal(pair_differences(y).ravel(), [-2., 2.])

    def test_test_targets_do_not_change_fit(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(48, 3))
        y = rng.normal(size=(48, 3, 4))
        splits = (np.arange(30), np.arange(30, 38), np.arange(38, 48))
        for mode in ('endpoint', 'contrast'):
            a = fit(x, y, splits, mode, 901, 20, 8)
            changed = y.copy()
            changed[38:, 0] += 100
            b = fit(x, changed, splits, mode, 901, 20, 8)
            for key in ('selected_epoch', 'validation_difference_mse', 'target_scale'):
                self.assertEqual(a[key], b[key])
            self.assertNotEqual(a['test_difference_mse'], b['test_difference_mse'])


if __name__ == '__main__':
    unittest.main()
