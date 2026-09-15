from __future__ import annotations

from leworldmodel.additional_files import run_pusht_soft75


def test_pusht_soft75_changes_only_the_retention_floor():
    config = run_pusht_soft75.load_pusht_soft75_configs()
    assert set(config["arms"]) == {"pusht_tagged_soft75_pred1"}

    arm = config["arms"]["pusht_tagged_soft75_pred1"]
    overrides = arm["overrides"]
    assert "data.dataset.name=pusht_expert_train.h5" in overrides
    assert "+pixel_tag.mode=video" in overrides
    assert "+loss.control.enabled=true" in overrides
    assert "+loss.aligned_gradient_routing.enabled=true" in overrides
    assert (
        "+loss.aligned_gradient_routing.minimum_retention=0.75" in overrides
    )
    assert arm["pair_suites"] == ["colour"]
    assert "+eval.tag_mode=video" in arm["eval_overrides"]


def test_pusht_soft75_does_not_mutate_hard_aligned_config():
    run_pusht_soft75.load_pusht_soft75_configs()
    original = run_pusht_soft75.run_control.load_control_configs()
    hard = original["arms"]["control_aligned_pred1"]["overrides"]
    assert not any("minimum_retention" in value for value in hard)
