from leworldmodel.additional_files import run_reacher_parallel_only


def test_reacher_parallel_only_changes_only_orthogonal_admission():
    config = run_reacher_parallel_only.load_reacher_parallel_only_configs()
    assert set(config["arms"]) == {"reacher_clean_parallel_only"}

    arm = config["arms"]["reacher_clean_parallel_only"]
    overrides = arm["overrides"]
    assert "data=dmc" in overrides
    assert "+loss.aligned_gradient_routing.enabled=true" in overrides
    assert "+loss.aligned_gradient_routing.orthogonal_mode=drop" in overrides
    assert not any("pixel_tag" in value for value in overrides)
