from __future__ import annotations

from leworldmodel.additional_files import run_reacher_control_full


def test_full_grid_keeps_all_matched_arms_under_distinct_names():
    cfg = run_reacher_control_full.load_full_configs()

    assert set(cfg["arms"]) == {
        "reacher_clean_jepa_full",
        "reacher_clean_aligned_full",
        "reacher_tagged_jepa_full",
        "reacher_tagged_aligned_full",
    }
    for name, arm in cfg["arms"].items():
        assert "data=dmc" in arm["overrides"]
        assert "+eval.config=reacher" in arm["eval_overrides"]
        joined = " ".join(arm["overrides"])
        assert ("+pixel_tag.mode=video" in joined) == ("tagged" in name)
        assert (
            "+loss.aligned_gradient_routing.enabled=true" in joined
        ) == ("aligned" in name)
