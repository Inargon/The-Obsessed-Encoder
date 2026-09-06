import math

import pytest
import torch

from additional_files.aligned_gradient_routing import (
    control_aligned_prediction_surrogate,
)


def _combined_gradient(pred_loss, control_loss, embedding, **route_kwargs):
    surrogate, diagnostics = control_aligned_prediction_surrogate(
        pred_loss, control_loss, embedding, **route_kwargs
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


def test_parallel_only_drops_orthogonal_component():
    embedding = torch.tensor([[1.0, 1.0]], requires_grad=True)
    pred_loss = embedding.sum()
    control_loss = embedding[0, 0]

    gradient, diagnostics = _combined_gradient(
        pred_loss, control_loss, embedding, orthogonal_mode="drop"
    )

    torch.testing.assert_close(gradient, torch.tensor([[2.0, 0.0]]))
    torch.testing.assert_close(
        diagnostics["prediction_orthogonal_gate"], torch.tensor(0.0)
    )


def test_shuffled_control_breaks_sample_correspondence():
    embedding = torch.ones((2, 2), requires_grad=True)
    pred_loss = embedding[0, 0] + embedding[1, 1]
    control_loss = embedding[0, 0] + embedding[1, 1]

    gradient, diagnostics = _combined_gradient(
        pred_loss, control_loss, embedding, shuffle_control=True
    )

    # Routing uses the other sample's orthogonal guide, so it removes the
    # prediction contribution. The ordinary control gradient remains intact.
    torch.testing.assert_close(
        gradient, torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    )
    torch.testing.assert_close(
        diagnostics["prediction_guide_shuffled"], torch.tensor(1.0)
    )


def test_unknown_orthogonal_mode_is_rejected():
    embedding = torch.ones((1, 2), requires_grad=True)
    with pytest.raises(ValueError, match="orthogonal_mode"):
        control_aligned_prediction_surrogate(
            embedding.sum(), embedding[:, 0].sum(), embedding,
            orthogonal_mode="unknown",
        )
