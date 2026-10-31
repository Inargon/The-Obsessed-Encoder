import unittest
import numpy as np
from readout_delayed_pairs import choice_metrics
from audit import split_episodes
class PairReadoutTests(unittest.TestCase):
 def test_perfect_choice_and_regret(self):
  costs=np.array([[0.,4,8],[9,0,3],[3,1,0],[0,2,4]])
  result=choice_metrics(costs,costs-costs.mean(1,keepdims=True),np.array([0,0,1,1]),np.array([True,False]))
  self.assertEqual(result['scene_accuracy'],1.)
  self.assertEqual(result['mean_candidate_regret'],0.)
  self.assertEqual(result['eligible_pair_both_correct_fraction'],1.)
 def test_split_pair_is_rejected(self):
  with self.assertRaises(ValueError):choice_metrics(np.zeros((1,3)),np.zeros((1,3)),np.array([0]),np.array([True]))
 def test_pair_groups_do_not_leak(self):
  ids=np.repeat(np.arange(32),2)
  for seed in (73,74,75):
   splits=split_episodes(ids,seed)
   sets=[set(ids[s]) for s in splits]
   self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
   for split in splits:
    _,counts=np.unique(ids[split],return_counts=True)
    self.assertTrue((counts==2).all())
 def test_empty_eligible_subset_is_null(self):
  result=choice_metrics(np.array([[0,1],[1,0]]),np.array([[0,1],[1,0]]),np.array([0,0]),np.array([False]))
  self.assertIsNone(result['eligible_pair_both_correct_fraction'])
if __name__=='__main__':unittest.main()
