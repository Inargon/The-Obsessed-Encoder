from __future__ import annotations

from leworldmodel.additional_files import run_cube_control


def test_cube_grid_is_matched_and_uses_cube_eval():
    cfg = run_cube_control.load_cube_configs()

    assert set(cfg["arms"]) == {
        "cube_clean_jepa",
        "cube_clean_aligned",
        "cube_tagged_jepa",
        "cube_tagged_aligned",
    }
    for name, arm in cfg["arms"].items():
        assert "data=ogb" in arm["overrides"]
        assert "+eval.config=cube" in arm["eval_overrides"]
        assert "pair_suites" not in arm
        joined = " ".join(arm["overrides"])
        assert ("+pixel_tag.mode=video" in joined) == ("tagged" in name)
        assert (
            "+loss.aligned_gradient_routing.enabled=true" in joined
        ) == ("aligned" in name)
        assert ("+loss.control.enabled=true" in joined) == ("aligned" in name)


def test_cube_grid_does_not_mutate_push_control_grid():
    run_cube_control.load_cube_configs()
    original = run_cube_control.run_control.load_control_configs()
    joined = " ".join(original["arms"]["control_aligned_pred1"]["overrides"])
    assert "data.dataset.name=pusht_expert_train.h5" in joined
    assert "+pixel_tag.mode=video" in joined
