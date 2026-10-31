import unittest
import numpy as np
from state_terminal_readout import aligned_visual_inputs
class CachedVisualInputTests(unittest.TestCase):
 def data(self):
  initial=np.arange(28).reshape(4,7);goals=np.array([[1.,2.,.3]]*4);ids=np.array([0,0,1,1])
  g=np.column_stack([goals[:,:2],np.sin(goals[:,2]),np.cos(goals[:,2])])
  feature=np.column_stack([np.ones((4,5)),g])
  return dict(pair_ids=ids.copy(),recorded_state=np.column_stack([initial,g]),jepa=feature.copy(),ours=feature.copy()),initial,goals,ids
 def test_explicit_goal_suffix_removed(self):
  b,s,g,ids=self.data();result=aligned_visual_inputs(b,s,g,ids)
  np.testing.assert_array_equal(result['jepa'],np.ones((4,5)))
  self.assertEqual(b['jepa'].shape,(4,9))
 def test_wrong_pair_order_rejected(self):
  b,s,g,ids=self.data();b['pair_ids']=b['pair_ids'][::-1]
  with self.assertRaises(AssertionError):aligned_visual_inputs(b,s,g,ids)
 def test_wrong_cached_scene_rejected(self):
  b,s,g,ids=self.data();b['recorded_state'][0,0]+=1
  with self.assertRaises(AssertionError):aligned_visual_inputs(b,s,g,ids)
 def test_missing_goal_suffix_rejected(self):
  b,s,g,ids=self.data();b['jepa'][0,-1]+=1
  with self.assertRaises(AssertionError):aligned_visual_inputs(b,s,g,ids)
if __name__=='__main__':unittest.main()
