from additional_files.bloop_rank_component_campaign import ARMS, arm_spec


def value(overrides: list[str], key: str) -> str | None:
    prefix = f"+{key}="
    matches = [item.removeprefix(prefix) for item in overrides if item.startswith(prefix)]
    assert len(matches) <= 1
    return matches[0] if matches else None


def test_rank_and_component_arms_change_only_declared_controls() -> None:
    resolved = {arm: arm_spec(arm)[1]["overrides"] for arm in ARMS}

    assert value(resolved["rank0_both"], "loss.bloop.enabled") is None
    assert value(resolved["rank4_both"], "loss.bloop.rank") == "4"
    assert value(resolved["rank1_inverse"], "loss.control.inverse_weight") == "1.0"
    assert value(
        resolved["rank1_inverse"], "loss.control.reachability_weight"
    ) == "0.0"
    assert value(resolved["rank1_reach"], "loss.control.inverse_weight") == "0.0"
    assert value(
        resolved["rank1_reach"], "loss.control.reachability_weight"
    ) == "0.1"

    for arm, overrides in resolved.items():
        assert value(overrides, "loss.control.cycle_scope") == "predictor_only", arm
        assert value(overrides, "loss.control.cycle_weight") == "0.5", arm
