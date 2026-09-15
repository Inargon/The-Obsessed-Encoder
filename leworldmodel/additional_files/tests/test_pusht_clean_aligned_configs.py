from leworldmodel.additional_files import run_pusht_clean_aligned


def test_clean_aligned_removes_only_the_tag_distribution():
    config = run_pusht_clean_aligned.load_pusht_clean_aligned_configs()
    assert set(config["arms"]) == {"pusht_clean_aligned"}
    assert config["common_overrides"] == [
        "data.dataset.name=pusht_expert_train.h5"
    ]
    assert config["common_eval_overrides"] == []

    arm = config["arms"]["pusht_clean_aligned"]
    joined = " ".join(arm["overrides"])
    assert "+loss.aligned_gradient_routing.enabled=true" in joined
    assert "orthogonal_mode=drop" not in joined
    assert "pixel_tag" not in joined
