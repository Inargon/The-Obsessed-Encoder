from __future__ import annotations

from leworldmodel.additional_files import run_tworoom_control


def test_tworoom_grid_is_matched_and_uses_tworoom_eval():
    cfg = run_tworoom_control.load_tworoom_configs()

    assert set(cfg["arms"]) == {
        "tworoom_clean_jepa",
        "tworoom_clean_aligned",
        "tworoom_tagged_jepa",
        "tworoom_tagged_aligned",
    }
    for name, arm in cfg["arms"].items():
        assert "data=tworoom" in arm["overrides"]
        assert "+eval.config=tworoom" in arm["eval_overrides"]
        assert "pair_suites" not in arm
        joined = " ".join(arm["overrides"])
        assert ("+pixel_tag.mode=video" in joined) == ("tagged" in name)
        assert (
            "+loss.aligned_gradient_routing.enabled=true" in joined
        ) == ("aligned" in name)
        assert ("+loss.control.enabled=true" in joined) == ("aligned" in name)


def test_tworoom_grid_does_not_mutate_push_control_grid():
    run_tworoom_control.load_tworoom_configs()
    original = run_tworoom_control.run_control.load_control_configs()
    joined = " ".join(original["arms"]["control_aligned_pred1"]["overrides"])
    assert "data.dataset.name=pusht_expert_train.h5" in joined
    assert "+pixel_tag.mode=video" in joined
