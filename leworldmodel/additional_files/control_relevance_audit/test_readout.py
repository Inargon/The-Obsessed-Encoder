import unittest
from types import SimpleNamespace
import numpy as np
from multisplit_readout import fit_readout, design_matrices

class ReadoutTests(unittest.TestCase):
 def test_nonlinearity_recovers_quadratic(self):
  rng=np.random.default_rng(12)
  x=rng.uniform(-2,2,size=(180,1));y=x*x
  splits=[np.arange(100),np.arange(100,140),np.arange(140,180)]
  linear,_=fit_readout(x,y,splits,'linear',32)
  nonlinear,_=fit_readout(x,y,splits,'rff',32)
  self.assertGreater(nonlinear['per_column'][0]['r2'],.95)
  self.assertLess(linear['per_column'][0]['r2'],.3)
 def test_test_targets_do_not_affect_fit_or_selection(self):
  rng=np.random.default_rng(22);x=rng.normal(size=(60,3));y=x[:,:1]**2
  splits=[np.arange(35),np.arange(35,45),np.arange(45,60)]
  a,p=fit_readout(x,y,splits,'rff',16)
  y[45:]+=100
  b,q=fit_readout(x,y,splits,'rff',16)
  np.testing.assert_array_equal(p,q)
  for first,second in zip(a['per_column'],b['per_column']):
   self.assertEqual(first['alpha'],second['alpha']);self.assertEqual(first['bandwidth'],second['bandwidth'])
 def test_scaling_does_not_use_heldout_features(self):
  x=np.arange(30,dtype=float).reshape(10,3)
  a=design_matrices(x,np.arange(5),np.arange(5,7),np.arange(7,10),'rff',8,1.,901)[0]
  x[5:]+=10000
  b=design_matrices(x,np.arange(5),np.arange(5,7),np.arange(7,10),'rff',8,1.,901)[0]
  np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
