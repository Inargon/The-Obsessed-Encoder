"""The unified wandb naming scheme: every mapped target lands in a canonical
section, no two sources within a case collide, and unmapped keys pass through."""

import pytest

from common import wandb_keys


CASES = {
    "lejepa": (wandb_keys.LEJEPA, None),
    "dinov3": (wandb_keys.DINOV3, wandb_keys.DINOV3_PREFIXES),
    "lewm": (wandb_keys.LEWM, None),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_targets_are_canonical(case):
    exact, prefixes = CASES[case]
    for target in exact.values():
        assert target.startswith(wandb_keys.CANONICAL_PREFIXES), target
    for target in (prefixes or {}).values():
        assert target in wandb_keys.CANONICAL_PREFIXES, target


@pytest.mark.parametrize("case", sorted(CASES))
def test_no_target_collisions(case):
    exact, _ = CASES[case]
    targets = list(exact.values())
    assert len(targets) == len(set(targets))


def test_every_case_has_the_shared_rows():
    # The point of the scheme: one objective row and at least one downstream
    # row per case, so a single workspace row reads across all three.
    for exact, _ in CASES.values():
        assert "objective/total" in exact.values()
        assert any(t.startswith("downstream/") for t in exact.values())


def test_remap_exact_prefix_and_passthrough():
    payload = {
        "ssl/total_loss": 1.0,        # exact beats prefix
        "ssl/koleo_loss": 2.0,        # prefix rule
        "pair/baseline/tokens_mean": 3.0,  # passthrough
        "sched/lr": 4.0,              # passthrough
    }
    out = wandb_keys.remap(payload, wandb_keys.DINOV3,
                           prefixes=wandb_keys.DINOV3_PREFIXES)
    assert out == {
        "objective/total": 1.0,
        "objective/koleo_loss": 2.0,
        "pair/baseline/tokens_mean": 3.0,
        "sched/lr": 4.0,
    }


