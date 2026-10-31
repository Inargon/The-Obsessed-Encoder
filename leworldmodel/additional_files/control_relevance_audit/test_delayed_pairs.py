import unittest
import numpy as np
from collect_delayed_pairs import delayed_mask,pose_change
class DelayedPairTests(unittest.TestCase):
 def test_delayed_motion_detected_without_initial_position_confound(self):
  s=np.zeros((2,2,3,8,7));s[:,1,:,:,3]=100
  # Initial scene positions differ, but neither object moves: not a delayed effect.
  keep,_,_=delayed_mask(s,3);self.assertFalse(keep.any())
  s[0,0,:,5:,2]=20
  keep,_,_=delayed_mask(s,3);np.testing.assert_array_equal(keep,[True,False])
 def test_early_motion_rejected(self):
  s=np.zeros((1,2,3,8,7));s[0,0,:,2:,2]=20
  self.assertFalse(delayed_mask(s,3)[0][0])
 def test_angle_wrap(self):
  s=np.zeros((1,3,7));s[0,:,4]=[2*np.pi-.01,.01,.02]
  np.testing.assert_allclose(pose_change(s)[0,:,2],[0,.02,.03],atol=1e-10)
if __name__=='__main__':unittest.main()
