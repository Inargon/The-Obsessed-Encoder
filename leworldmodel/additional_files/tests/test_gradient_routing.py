import torch

from additional_files.gradient_routing import conflict_aware_prediction_surrogate


def _combined_gradient(pred_loss, control_loss, embedding):
    surrogate, diagnostics = conflict_aware_prediction_surrogate(
        pred_loss, control_loss, embedding
    )
    gradient = torch.autograd.grad(
        pred_loss + control_loss + surrogate, embedding
    )[0]
    return gradient, diagnostics


def test_conflicting_prediction_component_is_removed():
    embedding = torch.tensor([[1.0, 1.0]], requires_grad=True)
    pred_loss = embedding[0, 0] - embedding[0, 1]
    control_loss = -embedding[0, 0]

    gradient, diagnostics = _combined_gradient(
        pred_loss, control_loss, embedding
    )

    # g_pred=[1,-1], g_control=[-1,0]. Projection makes the prediction
    # component [0,-1], hence the combined gradient is [-1,-1].
    torch.testing.assert_close(gradient, torch.tensor([[-1.0, -1.0]]))
    torch.testing.assert_close(
        diagnostics["prediction_control_conflict_fraction"], torch.tensor(1.0)
    )


def test_aligned_prediction_gradient_is_unchanged():
    embedding = torch.tensor([[1.0, 1.0]], requires_grad=True)
    pred_loss = embedding.sum()
    control_loss = 2.0 * embedding.sum()

    gradient, diagnostics = _combined_gradient(
        pred_loss, control_loss, embedding
    )

    torch.testing.assert_close(gradient, torch.tensor([[3.0, 3.0]]))
    torch.testing.assert_close(
        diagnostics["prediction_grad_removed_fraction"], torch.tensor(0.0)
    )
