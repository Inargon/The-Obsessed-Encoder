import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_future_goal_pair import make_pair

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from future_goal_objective import FutureGoalObjective


class ConfigurationTests(unittest.TestCase):
    def test_only_gradient_switch_differs(self):
        source = {'arms': {'baseline': {'overrides': [], 'eval_overrides': []}}}
        result = make_pair(source, 'job123')['arms']
        a, b = result.values()
        self.assertEqual([(x, y) for x, y in zip(a['overrides'], b['overrides']) if x != y],
            [('+loss.future_goal.detach_representation=true', '+loss.future_goal.detach_representation=false')])
        self.assertEqual(a['eval_overrides'], b['eval_overrides'])
        self.assertEqual(source['arms']['baseline']['overrides'], [])

    def test_names_cannot_escape(self):
        with self.assertRaises(ValueError):
            make_pair({}, '../bad')


@unittest.skipUnless(HAS_TORCH, 'Cluster PyTorch required for gradient checks')
class GradientTests(unittest.TestCase):
    def test_gradient_boundary_and_identical_forward(self):
        torch.manual_seed(3)
        detached = FutureGoalObjective(8, hidden_dim=12, detach_representation=True)
        joint = FutureGoalObjective(8, hidden_dim=12, detach_representation=False)
        joint.load_state_dict(detached.state_dict())
        x = torch.randn(4, 4, 8, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        a, b = detached(x)['future_goal_loss'], joint(y)['future_goal_loss']
        torch.testing.assert_close(a, b)
        a.backward(); b.backward()
        self.assertIsNone(x.grad)
        self.assertGreater(y.grad.norm().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in detached.parameters()))

    def test_later_frames_do_not_become_negatives(self):
        model = FutureGoalObjective(8, hidden_dim=12, detach_representation=False)
        x = torch.randn(4, 4, 8, requires_grad=True)
        original = model.logits(x, 1)
        changed = x.detach().clone()
        changed[:, 2:] += 100
        torch.testing.assert_close(original, model.logits(changed, 1))
        self.assertEqual(tuple(original.shape), (4, 4))
        original.sum().backward()
        torch.testing.assert_close(x.grad[:, 2:], torch.zeros_like(x.grad[:, 2:]))

    def test_duplicate_goals_have_finite_loss_not_unreachable_labels(self):
        model = FutureGoalObjective(8, hidden_dim=12)
        x = torch.ones(4, 4, 8)
        result = model(x)
        self.assertTrue(torch.isfinite(result['future_goal_loss']).item())
        # Identical goal keys give chance source classification, never a reachability certificate.
        self.assertAlmostEqual(result['future_goal_contrast_loss'].item(), 1.38629436, places=5)

    def test_budget_and_batch_validation(self):
        model = FutureGoalObjective(8)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 4, 8))
        with self.assertRaises(ValueError):
            model.logits(torch.zeros(4, 4, 8), 4)


if __name__ == '__main__':
    unittest.main()
