from __future__ import annotations

import pytest
import torch

from leworldmodel.additional_files.delta_jepa import (
    LatentDifferenceActionDecoder,
)


def _decoder() -> LatentDifferenceActionDecoder:
    return LatentDifferenceActionDecoder(
        embed_dim=12,
        action_dim=2,
        max_horizon=3,
        decoder_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=48,
        action_weight=10.0,
    )


def test_delta_decoder_predicts_aligned_action_sequences() -> None:
    torch.manual_seed(7)
    emb = torch.randn(4, 4, 12, requires_grad=True)
    actions = torch.randn(4, 4, 2)
    decoder = _decoder()

    terms = decoder(emb, actions)
    terms["delta_jepa_loss"].backward()

    assert terms["delta_jepa_loss"].item() == pytest.approx(
        10.0 * terms["delta_action_loss"].item()
    )
    assert all(terms[f"delta_horizon_{h}_loss"] > 0 for h in (1, 2, 3))
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )


def test_decoder_depends_only_on_latent_difference() -> None:
    torch.manual_seed(11)
    decoder = _decoder().eval()
    displacement = torch.randn(3, 12)
    offset = torch.randn(3, 12)

    direct = decoder.decode(displacement, horizon=2)
    shifted_endpoints = decoder.decode(
        (offset + displacement) - offset,
        horizon=2,
    )

    torch.testing.assert_close(direct, shifted_endpoints)


def test_invalid_decoder_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        LatentDifferenceActionDecoder(
            embed_dim=12,
            action_dim=2,
            decoder_dim=30,
            num_heads=8,
        )


def test_action_target_uses_every_intervening_action_in_order() -> None:
    actions = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])

    target = LatentDifferenceActionDecoder._action_sequence(actions, horizon=3)

    torch.testing.assert_close(target, torch.tensor([[[[1.0], [2.0], [3.0]]]]))
