import torch

from additional_files.action_router import ActionPatchRouter


def test_action_router_shapes_and_action_sensitivity():
    torch.manual_seed(0)
    router = ActionPatchRouter(embed_dim=12, heads=3, residual_scale=0.5)
    base = torch.randn(2, 4, 12)
    patches = torch.randn(2, 4, 5, 12)
    actions = torch.randn(2, 4, 12)

    first = router(base, patches, actions)
    # A uniform shift is intentionally removed by query LayerNorm. Perturb
    # only part of the action embedding to test the directional information
    # that the router is designed to consume.
    changed_actions = actions.clone()
    changed_actions[..., :3] += 1.0
    second = router(base, patches, changed_actions)

    assert first.shape == base.shape
    assert not torch.allclose(first, second)
    assert 0.0 < router.gate.item() < 1.0


def test_candidate_router_matches_individual_routes():
    torch.manual_seed(1)
    router = ActionPatchRouter(embed_dim=8, heads=2, residual_scale=0.25)
    base = torch.randn(2, 3, 8)
    patches = torch.randn(2, 3, 7, 8)
    actions = torch.randn(2, 5, 3, 8)

    batched = router.forward_candidates(base, patches, actions)
    individual = torch.stack(
        [router(base, patches, actions[:, index]) for index in range(actions.size(1))],
        dim=1,
    )

    assert batched.shape == (2, 5, 3, 8)
    torch.testing.assert_close(batched, individual)
