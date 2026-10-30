import unittest
import numpy as np
from audit import evaluate, split_episodes, validate, fit_predict

class AuditTests(unittest.TestCase):
 def test_episode_split(self):
  ids=np.repeat(np.arange(12),10);a,b,c=split_episodes(ids,73)
  self.assertFalse(set(ids[a])&set(ids[b]));self.assertFalse(set(ids[a])&set(ids[c]));self.assertFalse(set(ids[b])&set(ids[c]))
  self.assertEqual(len(a)+len(b)+len(c),len(ids))
 def test_few_episodes_rejected(self):
  with self.assertRaises(ValueError):split_episodes(np.array([1,1,2]),0)
 def test_heldout_values_do_not_change_fit(self):
  rng=np.random.default_rng(1);x=rng.normal(size=(40,4));y=x[:,:2].copy()
  first=fit_predict(x,y,np.arange(30),np.arange(30,40),1)
  y[30:]+=1000
  np.testing.assert_allclose(first,fit_predict(x,y,np.arange(30),np.arange(30,40),1))
 def test_informative_features_beat_constant(self):
  rng=np.random.default_rng(4);s=rng.normal(size=(120,5));a=rng.normal(size=(120,2));n=s.copy();n[:,2]+=s[:,0]
  d=dict(features=s.copy(),state=s,previous_state=s*.9,next_state=n,action=a,episode_id=np.repeat(np.arange(12),10))
  result=evaluate(d)
  self.assertGreater(result['state_probes']['block_xy']['r2'],.95)
  out=result['outcome_probes']['block_delta_xy']
  self.assertGreater(out['features_and_action']['r2'],out['action_only']['r2']+.5)
  d['features']=np.ones((120,2));self.assertLess(evaluate(d)['state_probes']['block_xy']['r2'],.1)
 def test_raw_seven_state(self):
  rng=np.random.default_rng(8);s=rng.normal(size=(60,7))
  data=dict(features=s,state=s,previous_state=s*.9,next_state=s*1.1,action=s[:,:2],episode_id=np.repeat(np.arange(10),6),state_schema=np.asarray('raw'))
  result=evaluate(data)
  self.assertEqual(len(result['state_probes']),7)
  self.assertIn('raw_delta_col_6',result['outcome_probes'])
  self.assertNotIn('block_angle_sincos',result['state_probes'])
 def test_nonfinite_rejected(self):
  s=np.ones((10,5));d=dict(features=s.copy(),state=s,previous_state=s,next_state=s,action=s,episode_id=np.arange(10))
  d['features'][0,0]=np.nan
  with self.assertRaises(ValueError):validate(d)

if __name__=='__main__':unittest.main()
