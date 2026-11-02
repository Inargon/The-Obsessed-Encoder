import unittest
from read_oracle_candidates import compare


class ReadoutTests(unittest.TestCase):
    def test_small_gain_tradeoff_and_material(self):
        def state(x, angle):
            return [1, 2, x, 0, angle, 3, 4, 5, 6, 7]
        d = {'candidate_state_traces': [[state(20, .4)], [state(19.999, .4)],
             [state(18, .3)], [state(10, .5)]], 'oracle_cost': [0, 1, 2, 3]}
        rows = compare(d, [0]*10, 'oracle')
        self.assertTrue(rows[1]['dominates'])
        self.assertFalse(rows[1]['material'])
        self.assertTrue(rows[2]['material'])
        self.assertFalse(rows[3]['dominates'])
        self.assertEqual(rows[2]['score_gap'], 2)
        self.assertEqual(rows[2]['agent_vx'], 3)
        self.assertEqual(rows[2]['block_omega'], 7)

    def test_angle_wrap_and_nonfinite(self):
        a = [0]*10; a[4] = -3.141592653589793+.01
        b = [0]*10; b[4] = 3.141592653589793-.01
        goal = [0]*10; goal[4] = 3.141592653589793
        d = {'candidate_state_traces': [[a], [b]], 'oracle_cost': [1, 0]}
        rows = compare(d, goal, 'oracle')
        self.assertAlmostEqual(rows[0]['gain_angle_rad'], 0)
        self.assertFalse(rows[0]['dominates'])
        d['oracle_cost'][0] = float('nan')
        with self.assertRaises(ValueError):
            compare(d, goal, 'oracle')


if __name__ == '__main__':
    unittest.main()
