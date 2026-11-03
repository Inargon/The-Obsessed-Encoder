import unittest
import numpy as np
from pullback_pilot import directional_response, physical_differences, rank_corr


class PullbackTests(unittest.TestCase):
    def test_linear_response_matches_exact_difference(self):
        rng = np.random.default_rng(4)
        matrix = rng.normal(size=(5, 3))
        x, delta = rng.normal(size=(2, 7, 5))
        fn = lambda z: z @ matrix + 12
        for eps in (.01, .005):
            np.testing.assert_allclose(directional_response(fn, x, delta, eps),
                                       fn(x + delta) - fn(x), atol=1e-10)

    def test_nonlinear_locality_is_not_assumed(self):
        x, delta = np.array([1.]), np.array([2.])
        fn = lambda z: z ** 2
        derivative = directional_response(fn, x, delta, .005)
        np.testing.assert_allclose(derivative, [4.])
        self.assertGreater(abs(float((fn(x+delta)-fn(x)-derivative)[0])), 3.)

    def test_ties_and_constant_ranks(self):
        self.assertIsNone(rank_corr([1, 1], [1, 2]))
        self.assertAlmostEqual(rank_corr([1, 1, 3], [8, 8, 4]), -1.)
        with self.assertRaises(ValueError):
            rank_corr([np.nan], [1])

    def test_initial_position_difference_is_not_motion(self):
        states = np.zeros((1, 2, 3, 6, 7))
        states[:, 1, :, :, 2] = 50
        states[:, 0, :, :, 4] = np.pi - .01
        states[:, 1, :, :, 4] = -np.pi + .01
        result = physical_differences(states, 5)
        np.testing.assert_allclose(result['terminal_xy'], 50)
        np.testing.assert_allclose(result['terminal_angle'], .02)
        np.testing.assert_allclose(result['displacement_xy'], 0)
        np.testing.assert_allclose(result['displacement_angle'], 0)


if __name__ == '__main__':
    unittest.main()
