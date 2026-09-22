from leworldmodel.additional_files import clean_inverse_reach_campaign


def _overrides(task: str) -> list[str]:
    _, spec = clean_inverse_reach_campaign.task_spec(task)
    return spec["overrides"]


def _value(overrides: list[str], key: str) -> float:
    matches = [item for item in overrides if item.startswith(key + "=")]
    assert len(matches) == 1
    return float(matches[0].split("=", 1)[1])


def test_all_three_clean_tasks_are_available() -> None:
    assert clean_inverse_reach_campaign.TASKS == (
        "reacher",
        "cube",
        "tworoom",
    )


def test_cycle_is_disabled_and_reachability_is_retained() -> None:
    for task in clean_inverse_reach_campaign.TASKS:
        overrides = _overrides(task)
        assert _value(overrides, "+loss.control.cycle_weight") == 0.0
        assert _value(overrides, "+loss.control.reachability_weight") > 0.0


def test_tasks_use_clean_data_and_keep_aligned_routing() -> None:
    expected_data = {
        "reacher": "data=dmc",
        "cube": "data=ogb",
        "tworoom": "data=tworoom",
    }
    for task in clean_inverse_reach_campaign.TASKS:
        overrides = _overrides(task)
        assert expected_data[task] in overrides
        assert not any("pixel_tag" in item for item in overrides)
        assert any("gradient_routing" in item for item in overrides)
