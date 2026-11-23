import torch

from leworldmodel.additional_files.counterfactual_planning import counterfactual_planning_objective


def ranking_loss(reference: tuple, changed: tuple, **options):
    target = torch.zeros(1, 1, 1)
    return counterfactual_planning_objective(
        reference_emb=target,
        changed_emb=target,
        reference_pred=reference[0],
        changed_pred=changed[0],
        reference_negative_pred=reference[1],
        changed_negative_pred=changed[1],
        reference_target=target,
        changed_target=target,
        **options,
    )["counterfactual_ranking_loss"]


def prediction(value: float, requires_grad=False):
    return torch.full((1, 1, 1), value, requires_grad=requires_grad)


def test_centered_mse_ignores_a_common_cost_offset():
    loss = ranking_loss(
        (prediction(0.0), prediction(1.0)),
        (prediction(1.0), prediction(2.0**0.5)),
        ranking_metric="centered_mse",
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-12, rtol=0)


def test_centered_mse_retains_gradient_for_large_action_margins():
    reference = (prediction(0.0), prediction(10.0))
    changed_negative = prediction(20.0, requires_grad=True)
    changed = (prediction(0.0), changed_negative)
    original = ranking_loss(reference, changed)
    explicit_js = ranking_loss(reference, changed, ranking_metric="js")
    torch.testing.assert_close(original, explicit_js)
    centered = ranking_loss(reference, changed, ranking_metric="centered_mse")
    assert centered.item() == 22500.0
    assert torch.autograd.grad(original, changed_negative)[0].item() == 0.0
    assert torch.autograd.grad(centered, changed_negative)[0].item() == 6000.0


def test_margin_guard_stays_inactive_when_retagged_order_has_enough_margin():
    reference = (prediction(0.0), prediction(2.0))
    changed_negative = prediction(2.0, requires_grad=True)
    changed = (prediction(1.0), changed_negative)
    guarded = ranking_loss(reference, changed, ranking_metric="margin_guard")
    assert guarded.item() == 0.0
    assert torch.autograd.grad(guarded, changed_negative)[0].item() == 0.0
    assert ranking_loss(reference, changed, ranking_metric="centered_mse").item() > 0.0


def test_margin_guard_corrects_a_flip_without_moving_its_reference():
    reference_negative = prediction(2.0, requires_grad=True)
    changed_positive = prediction(2.0, requires_grad=True)
    changed_negative = prediction(1.0, requires_grad=True)
    loss = ranking_loss(
        (prediction(0.0), reference_negative),
        (changed_positive, changed_negative),
        ranking_metric="margin_guard",
    )
    ref_grad, positive_grad, negative_grad = torch.autograd.grad(
        loss, (reference_negative, changed_positive, changed_negative), allow_unused=True,
    )
    assert loss.item() == 4.0
    assert ref_grad is None
    assert positive_grad.item() == 8.0
    assert negative_grad.item() == -4.0


def test_margin_guard_skips_an_uncertain_reference():
    loss = ranking_loss(
        (prediction(1.0), prediction(1.01)),
        (prediction(2.0), prediction(1.0, requires_grad=True)),
        ranking_metric="margin_guard",
    )
    assert loss.item() == 0.0
