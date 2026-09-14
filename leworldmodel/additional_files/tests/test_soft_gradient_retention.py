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


def test_minimum_retention_preserves_raw_orthogonal_prediction_gradient():
    embedding = torch.tensor([[1.0, 1.0]], requires_grad=True)
    pred_loss = embedding[0, 0]
    control_loss = embedding[0, 1]

    gradient, diagnostics = _combined_gradient(
        pred_loss,
        control_loss,
        embedding,
        minimum_retention=0.5,
    )

    # Strict aligned routing would remove the orthogonal prediction update.
    # The residual retains exactly half while control remains unchanged.
    torch.testing.assert_close(gradient, torch.tensor([[0.5, 1.0]]))
    torch.testing.assert_close(
        diagnostics["prediction_grad_retained_fraction"], torch.tensor(0.5)
    )


def test_full_minimum_retention_recovers_unmodified_prediction_gradient():
    embedding = torch.tensor([[1.0, 1.0]], requires_grad=True)
    pred_loss = embedding[0, 0]
    control_loss = embedding[0, 1]

    gradient, diagnostics = _combined_gradient(
        pred_loss,
        control_loss,
        embedding,
        minimum_retention=1.0,
    )

    torch.testing.assert_close(gradient, torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(
        diagnostics["prediction_grad_retained_fraction"], torch.tensor(1.0)
    )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_invalid_minimum_retention_is_rejected(value):
    embedding = torch.ones((1, 2), requires_grad=True)
    with pytest.raises(ValueError, match="minimum_retention"):
        control_aligned_prediction_surrogate(
            embedding.sum(),
            embedding[:, 0].sum(),
            embedding,
            minimum_retention=value,
        )
