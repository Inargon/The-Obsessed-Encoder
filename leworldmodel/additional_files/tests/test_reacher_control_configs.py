from __future__ import annotations

from leworldmodel.additional_files import run_reacher_control


def test_reacher_grid_is_matched_and_uses_reacher_eval():
    cfg = run_reacher_control.load_reacher_configs()

    assert set(cfg["arms"]) == {
        "reacher_clean_jepa",
        "reacher_clean_aligned",
        "reacher_tagged_jepa",
        "reacher_tagged_aligned",
    }
    for name, arm in cfg["arms"].items():
        assert "data=dmc" in arm["overrides"]
        assert "+eval.config=reacher" in arm["eval_overrides"]
        assert "pair_suites" not in arm
        joined = " ".join(arm["overrides"])
        assert ("+pixel_tag.mode=video" in joined) == ("tagged" in name)
        assert (
            "+loss.aligned_gradient_routing.enabled=true" in joined
        ) == ("aligned" in name)
        assert ("+loss.control.enabled=true" in joined) == ("aligned" in name)


def test_reacher_grid_does_not_mutate_push_control_grid():
    run_reacher_control.load_reacher_configs()
    original = run_reacher_control.run_control.load_control_configs()
    joined = " ".join(original["arms"]["control_aligned_pred1"]["overrides"])
    assert "data.dataset.name=pusht_expert_train.h5" in joined
    assert "+pixel_tag.mode=video" in joined
