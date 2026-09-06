import numpy as np
import torch

from leworldmodel.additional_files.effect_geometry import (
    centred_normalized_gram,
    geometry_terms,
    physical_effects,
)


def test_physical_effects_use_periodic_angle_coordinates():
    anchors = torch.tensor([[0., 0., 0., 0., np.pi - 0.01, 0., 0.]])
    states = anchors[:, None].repeat(1, 2, 1)
    states[0, 1, 4] = -np.pi + 0.01
    effects = physical_effects(states, anchors)
    assert effects[0, 1, 4:].norm() < 0.03


def test_gram_is_invariant_to_translation_rotation_and_scale():
    torch.manual_seed(0)
    x = torch.randn(3, 8, 4)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    transformed = 7.0 * (x @ q) + 13.0
    assert torch.allclose(
        centred_normalized_gram(x), centred_normalized_gram(transformed), atol=1e-5
    )


def test_geometry_prefers_matching_cloud_and_has_gradients():
    torch.manual_seed(1)
    physical = torch.randn(2, 8, 5)
    good = physical.clone().requires_grad_()
    bad = torch.randn_like(physical)
    good_terms = geometry_terms(good, physical, min_effect_rms=0.0)
    bad_terms = geometry_terms(bad, physical, min_effect_rms=0.0)
    assert good_terms["effect_geometry_gram_loss"] < bad_terms["effect_geometry_gram_loss"]
    good_terms["effect_geometry_gram_loss"].backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()


def test_scale_band_penalizes_collapsed_cloud():
    physical = torch.randn(2, 8, 5)
    collapsed = torch.zeros(2, 8, 12)
    terms = geometry_terms(collapsed, physical, min_effect_rms=0.05)
    assert terms["effect_geometry_scale_loss"] > 0
