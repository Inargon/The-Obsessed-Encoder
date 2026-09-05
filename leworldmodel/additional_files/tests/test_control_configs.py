from __future__ import annotations

from leworldmodel.additional_files import run_control


def test_control_grid_has_two_new_single_seed_candidates():
    config = run_control.load_control_configs()

    assert set(config["arms"]) == {
        "direct_reachability",
        "factorized_reachability",
        "multi_horizon_idm",
        "masked_reachability",
    }
    for name, arm in config["arms"].items():
        joined = " ".join(arm["overrides"])
        assert "data.dataset.name=pusht_expert_train.h5" in joined
        assert "+pixel_tag.mode=video" in joined
        assert f"+loss.control.mode={name}" in joined
        assert arm["pair_suites"] == ["colour"]
