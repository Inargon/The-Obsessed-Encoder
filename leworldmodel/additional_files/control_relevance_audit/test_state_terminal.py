import unittest
import numpy as np
from state_terminal_readout import encode_pose,terminal_cost
class TerminalScoringTests(unittest.TestCase):
 def test_exact_goal_has_zero_cost_including_wrapped_angle(self):
  goals=np.array([[12.,25.,.2],[3.,7.,-.3]])
  pose=np.repeat(goals[:,None,:],3,axis=1);pose[0,:,2]+=2*np.pi
  np.testing.assert_allclose(terminal_cost(encode_pose(pose),goals),0,atol=1e-20)
 def test_analytic_cost_matches_original_formula(self):
  rng=np.random.default_rng(11);pose=rng.normal(size=(12,3,3));goals=rng.normal(size=(12,3))
  d=pose[...,2]-goals[:,None,2];d=np.arctan2(np.sin(d),np.cos(d))
  expected=((pose[...,:2]-goals[:,None,:2])**2).sum(-1)+(30*d)**2
  np.testing.assert_allclose(terminal_cost(encode_pose(pose),goals),expected)
 def test_goal_does_not_change_encoded_terminal_target(self):
  pose=np.array([[[4.,5.,1.]]]);encoded=encode_pose(pose)
  first=terminal_cost(encoded,np.array([[4.,5.,1.]]))
  second=terminal_cost(encoded,np.array([[14.,5.,1.]]))
  self.assertLess(first[0,0],1e-20);self.assertAlmostEqual(second[0,0],100.)
if __name__=='__main__':unittest.main()
