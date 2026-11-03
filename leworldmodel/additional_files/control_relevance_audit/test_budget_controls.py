import unittest
import numpy as np
from budget_controls import select_features, evaluation_masks


class BudgetControlTests(unittest.TestCase):
    def test_geometry_excludes_history_agent_and_velocity(self):
        x = np.zeros((2,28)); x[:,21] = x[:,27] = 1
        x[:,18:20] = [100,200]; x[:,24:26] = [103,204]
        changed = x.copy(); changed[:,:18] = 999; changed[:,22:24] = 999
        np.testing.assert_array_equal(select_features(x,'geometry'),select_features(changed,'geometry'))
        np.testing.assert_allclose(select_features(x,'geometry')[0,:5],[3,4,5,0,1])

    def test_current_excludes_past_but_keeps_agent(self):
        x = np.zeros((2,28)); x[:,:16] = 99; x[:,16] = 7
        output = select_features(x,'current')
        self.assertFalse(output[:,:16].any())
        self.assertTrue((output[:,16] == 7).all())
        np.testing.assert_array_equal(select_features(x,'history'),x)

    def test_complete_followup_does_not_admit_early_hit_short_trace(self):
        labels = np.array([[0,1],[0,0],[1,1]],bool)
        masks = evaluation_masks(labels,np.ones_like(labels),[10,80,80],80)
        np.testing.assert_array_equal(masks['complete_followup'],[[0,0],[1,1],[1,1]])
        np.testing.assert_array_equal(masks['complete_followup_not_initially_hit'],[[0,0],[1,1],[0,0]])
        np.testing.assert_array_equal(masks['not_initially_hit'],[[1,1],[1,1],[0,0]])


if __name__ == '__main__':
    unittest.main()
