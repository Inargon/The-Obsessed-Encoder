from __future__ import annotations

from leworldmodel.additional_files import run_pusht_conflict_only


def test_pusht_conflict_only_is_a_matched_tagged_comparator():
    config = run_pusht_conflict_only.load_pusht_conflict_only_configs()
    assert set(config["arms"]) == {"pusht_tagged_conflict_only_pred1"}

    arm = config["arms"]["pusht_tagged_conflict_only_pred1"]
    overrides = arm["overrides"]
    assert "data.dataset.name=pusht_expert_train.h5" in overrides
    assert "+pixel_tag.mode=video" in overrides
    assert "+loss.control.enabled=true" in overrides
    assert "+loss.aligned_gradient_routing.enabled=true" in overrides
    assert (
        "+loss.aligned_gradient_routing.routing_mode=conflict_only" in overrides
    )
    assert arm["pair_suites"] == ["colour"]
    assert "+eval.tag_mode=video" in arm["eval_overrides"]


def test_pusht_conflict_only_does_not_mutate_hard_aligned_config():
    run_pusht_conflict_only.load_pusht_conflict_only_configs()
    original = run_pusht_conflict_only.run_control.load_control_configs()
    hard = original["arms"]["control_aligned_pred1"]["overrides"]
    assert not any("routing_mode=conflict_only" in value for value in hard)
