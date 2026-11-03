import unittest
import numpy as np
from goal_role_data import branch_targets, group_split, evaluate_predictions, set_render_pose


class GoalRoleTests(unittest.TestCase):
    def test_angle_must_precede_position_and_geometry_refresh(self):
        from types import SimpleNamespace
        class OffsetBody:
            position = (0.,0.)
            _angle = 0.
            @property
            def angle(self):
                return self._angle
            @angle.setter
            def angle(self, value):
                self.position = (self.position[0]+43.,self.position[1]-75.)
                self._angle = value
        body = OffsetBody()
        cached = []
        space = SimpleNamespace(reindex_shapes_for_body=lambda b:cached.append([*b.position,b.angle]))
        goal = [232.76039159,275.49869451,7.08797263]
        set_render_pose(body,space,goal)
        np.testing.assert_allclose(cached,[goal])

    def test_real_offset_center_of_gravity_pose(self):
        try:
            import pymunk
        except ImportError:
            self.skipTest('Pymunk unavailable locally; mandatory in collector environment')
        space = pymunk.Space()
        body = pymunk.Body(1,100)
        shape = pymunk.Circle(body,5,offset=(0,40))
        space.add(body,shape)
        body.center_of_gravity = (0,40)
        body.position = (232.76039159,275.49869451)
        before = np.asarray(body.position).copy()
        body.angle = 7.08797263
        self.assertGreater(np.linalg.norm(np.asarray(body.position)-before),1.)
        for angle in (7.08797263,-1.2,0.):
            goal = [232.76039159,275.49869451,angle]
            set_render_pose(body,space,goal)
            np.testing.assert_allclose(body.local_to_world((0,0)),goal[:2],atol=1e-6)

    def test_multiple_success_and_first_hit(self):
        current = np.zeros(10); current[2] = 100
        trace = np.broadcast_to(current,(3,40,10)).copy()
        trace[0,2,2] = 0; trace[1,15,2] = 0
        y, initial = branch_targets(current,trace,[0,0,0])
        self.assertFalse(initial)
        np.testing.assert_array_equal(y,[[1,1,1,1],[0,0,1,1],[0,0,0,0]])

    def test_initial_hit(self):
        y, initial = branch_targets(np.zeros(10),np.ones((3,40,10))*100,[0,0,0])
        self.assertTrue(initial and y.all())

    def test_group_split(self):
        groups = np.repeat(np.arange(15),4)
        parts = group_split(groups)
        sets = [set(groups[p]) for p in parts]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(sum(map(len,parts)),len(groups))

    def test_missed_opportunities(self):
        y = np.zeros((1,1,2,2),bool); y[:,:,0] = True
        p = np.zeros((1,1,2,2,2))
        p[0,0,0] = [[.8,.9],[.1,.2]]
        p[0,0,1] = [[.1,.2],[.8,.9]]
        result = evaluate_predictions(p,y,np.array([[False]]))
        self.assertEqual(result[0]['missed_opportunities_by_budget'],[0,0])
        self.assertEqual(result[1]['missed_opportunities_by_budget'],[1,1])

    def test_nonmonotone_rejected(self):
        p = np.zeros((1,1,1,1,2)); p[...,0] = .9
        with self.assertRaises(ValueError):
            evaluate_predictions(p,np.zeros((1,1,1,2),bool),np.zeros((1,1),bool))


if __name__ == '__main__':
    unittest.main()
