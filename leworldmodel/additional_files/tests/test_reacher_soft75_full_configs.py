from __future__ import annotations

from leworldmodel.additional_files import run_reacher_soft75_full


def test_reacher_soft75_full_is_clean_and_matched():
    config = run_reacher_soft75_full.load_reacher_soft75_full_configs()
    assert set(config["arms"]) == {"reacher_clean_soft75_full"}
    arm = config["arms"]["reacher_clean_soft75_full"]
    overrides = arm["overrides"]
    assert "data=dmc" in overrides
    assert not any("pixel_tag" in value for value in overrides)
    assert "+loss.control.enabled=true" in overrides
    assert "+loss.aligned_gradient_routing.enabled=true" in overrides
    assert (
        "+loss.aligned_gradient_routing.minimum_retention=0.75" in overrides
    )
    assert "+eval.config=reacher" in arm["eval_overrides"]
