import torch

from additional_files.split_prediction_gradient import (
    encoder_scaled_prediction_surrogate,
)


def test_surrogate_scales_representation_gradient_only():
    embedding = torch.tensor([[1.0, -2.0]], requires_grad=True)
    predictor_scale = torch.tensor(2.0, requires_grad=True)
    pred_loss = (predictor_scale * embedding).sum()

    surrogate, diagnostics = encoder_scaled_prediction_surrogate(
        pred_loss, embedding, encoder_weight=0.3
    )
    embedding_grad, predictor_grad = torch.autograd.grad(
        pred_loss + surrogate, (embedding, predictor_scale)
    )

    torch.testing.assert_close(embedding_grad, torch.tensor([[0.6, 0.6]]))
    # Predictor parameters still receive the unscaled loss gradient.
    torch.testing.assert_close(predictor_grad, torch.tensor(-1.0))
    torch.testing.assert_close(
        diagnostics["prediction_grad_removed_fraction"], torch.tensor(0.7)
    )


def test_weight_one_is_identity():
    embedding = torch.tensor([[1.0, 2.0]], requires_grad=True)
    pred_loss = embedding.square().sum()
    surrogate, _ = encoder_scaled_prediction_surrogate(
        pred_loss, embedding, encoder_weight=1.0
    )

    assert surrogate.item() == 0.0
    gradient = torch.autograd.grad(pred_loss + surrogate, embedding)[0]
    torch.testing.assert_close(gradient, torch.tensor([[2.0, 4.0]]))
