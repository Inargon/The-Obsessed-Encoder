import math

import torch

from additional_files.aligned_gradient_routing import (
    control_aligned_prediction_surrogate,
)


def _combined_gradient(pred_loss, control_loss, embedding):
    surrogate, diagnostics = control_aligned_prediction_surrogate(
        pred_loss, control_loss, embedding
    )
    gradient = torch.autograd.grad(
        pred_loss + control_loss + surrogate, embedding
    )[0]
    return gradient, diagnostics


def test_aligned_prediction_gradient_is_preserved():
    embedding = torch.tensor([[1.0, 1.0]], requires_grad=True)
    pred_loss = embedding.sum()
    control_loss = 2.0 * embedding.sum()

    gradient, diagnostics = _combined_gradient(
        pred_loss, control_loss, embedding
    )

    torch.testing.assert_close(gradient, torch.tensor([[3.0, 3.0]]))
    torch.testing.assert_close(
        diagnostics["prediction_grad_retained_fraction"], torch.tensor(1.0)
    )


def test_orthogonal_prediction_gradient_is_suppressed():
    embedding = torch.tensor([[1.0, 1.0]], requires_grad=True)
    pred_loss = embedding[0, 0]
    control_loss = embedding[0, 1]

    gradient, diagnostics = _combined_gradient(
        pred_loss, control_loss, embedding
    )

    torch.testing.assert_close(gradient, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(
        diagnostics["prediction_orthogonal_gate"], torch.tensor(0.0)
    )


def test_partial_alignment_gates_only_orthogonal_component():
    embedding = torch.tensor([[1.0, 1.0]], requires_grad=True)
    pred_loss = embedding.sum()
    control_loss = embedding[0, 0]

    gradient, _ = _combined_gradient(pred_loss, control_loss, embedding)

    expected = torch.tensor([[2.0, 1.0 / math.sqrt(2.0)]])
    torch.testing.assert_close(gradient, expected)
