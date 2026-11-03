import unittest
from types import SimpleNamespace
import numpy as np
from goal_agent_intervention import render_goal_variants, rank_variants, verify_baseline, goal_set_cost


class GoalAgentTests(unittest.TestCase):
    def test_render_restores_state_on_success_and_failure(self):
        class Env:
            def __init__(self):
                self.agent = SimpleNamespace(position=(100., 100.))
                self.fail = False
                self.cached_position = self.agent.position
                self.space = SimpleNamespace(reindex_shapes_for_body=self.reindex)
            def reindex(self, body):
                self.cached_position = tuple(body.position)
            def render(self):
                if self.fail and self.agent.position[0] > 100:
                    raise RuntimeError('render failed')
                frame = np.zeros((160, 160, 3), dtype=np.uint8)
                x, y = map(int, self.cached_position)
                frame[y, x] = 255
                return frame
        env = Env()
        def snapshot():
            return np.array([*env.agent.position, 200, 200, .5, 1, 2, 3, 4, 5])
        before = snapshot()
        # Reproduce the real failure: body state changes, rendered geometry does not.
        original = env.render().copy()
        env.agent.position = (68., 100.)
        np.testing.assert_array_equal(env.render(), original)
        env.agent.position = (100., 100.)
        images, records = render_goal_variants(env, snapshot)
        self.assertEqual(len(images), 5)
        self.assertEqual(records[0]['changed_pixels'], 0)
        self.assertTrue(all(r['changed_pixels'] == 2 for r in records[1:]))
        np.testing.assert_array_equal(snapshot(), before)
        env.fail = True
        with self.assertRaises(RuntimeError):
            render_goal_variants(env, snapshot)
        np.testing.assert_array_equal(snapshot(), before)

    def test_real_pymunk_debug_draw_reindex(self):
        try:
            import pymunk
        except ImportError:
            self.skipTest('Pymunk absent locally; this test runs in cluster environment')
        class Recorder(pymunk.SpaceDebugDrawOptions):
            def __init__(self):
                super().__init__()
                self.positions = []
                self.flags = self.DRAW_SHAPES
            def draw_circle(self, pos, angle, radius, outline_color, fill_color):
                self.positions.append(tuple(pos))
        space = pymunk.Space()
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (100., 100.)
        shape = pymunk.Circle(body, 10)
        space.add(body, shape)
        space.reindex_shapes_for_body(body)
        original = Recorder(); space.debug_draw(original)
        self.assertEqual(original.positions, [(100., 100.)])
        body.position = (68., 100.)
        space.reindex_shapes_for_body(body)
        moved = Recorder(); space.debug_draw(moved)
        self.assertEqual(moved.positions, [(68., 100.)])
        body.position = (100., 100.)
        space.reindex_shapes_for_body(body)
        restored = Recorder(); space.debug_draw(restored)
        self.assertEqual(restored.positions, original.positions)

    def test_goal_set_reduction_and_duplicate_invariance(self):
        costs = np.array([[1., 4., 3.], [2., .5, 5.]])
        scores, matches = goal_set_cost(costs)
        np.testing.assert_array_equal(scores, [1., .5, 3.])
        np.testing.assert_array_equal(matches, [0, 1, 0])
        np.testing.assert_array_equal(goal_set_cost(costs[:1])[0], costs[0])
        np.testing.assert_array_equal(goal_set_cost(np.r_[costs, costs])[0], scores)
        with self.assertRaises(ValueError):
            goal_set_cost([[np.nan]])

    def test_rank_flips_excludes_numeric_ties(self):
        costs = [[1,2], [2,1], [1+1e-8,1], [1,2], [1,2]]
        result = rank_variants(costs, [[20,.5],[15,.3]])
        self.assertEqual(result['selected'], [0,1,1,0,0])
        self.assertEqual(result['robust_flips'], [False,True,False,False,False])
        self.assertEqual(result['selected_physical_errors'][1], [15,.3])
        costs[0][0] = float('nan')
        with self.assertRaises(ValueError):
            rank_variants(costs, [[20,.5],[15,.3]])

    def test_reference_rejects_changed_action_path(self):
        import copy
        decision = dict(selected=0, candidate_actions=[[[0,0]],[[1,1]]],
            candidate_state_traces=np.zeros((2,1,10)).tolist(), current_state=[0]*10,
            learned_cost=[1,2], oracle_cost=[1,3])
        arm = dict(diagnostic_success=False, final_xy_error=20, final_angle_error=.5,
                   decisions=[decision])
        reference = {'queries':[{'row':12,'arms':{'learned':arm}}]}
        verify_baseline(copy.deepcopy(reference), reference)
        changed = copy.deepcopy(reference)
        changed['queries'][0]['arms']['learned']['decisions'][0]['selected'] = 1
        with self.assertRaises(AssertionError):
            verify_baseline(changed, reference)


if __name__ == '__main__':
    unittest.main()
