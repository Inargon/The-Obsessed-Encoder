from __future__ import annotations

from leworldmodel.additional_files import run_randgoal_control


def test_randgoal_control_retargets_data_eval_and_pair_geometry():
    cfg = run_randgoal_control.load_randgoal_control_configs()
    arm = cfg["arms"]["randgoal_control_aligned_pred1"]
    joined = " ".join(arm["overrides"])

    assert "data.dataset.name=pusht_scripted_goal_train.lance" in joined
    assert "pusht_expert_train.h5" not in joined
    assert "pixel_tag" not in joined
    assert "+loss.aligned_gradient_routing.enabled=true" in joined
    assert "+loss.control.inverse_target=sequence" in joined
    assert arm["eval_overrides"] == [
        "+eval.dataset_name=pusht_scripted_goal_train.lance"
    ]
    assert arm["pair_suites"] == ["t_position"]
    assert "control_aligned_pred1" not in cfg["arms"]


def test_randgoal_control_does_not_mutate_original_control_grid():
    run_randgoal_control.load_randgoal_control_configs()
    original = run_randgoal_control.run_control.load_control_configs()
    arm = original["arms"]["control_aligned_pred1"]
    joined = " ".join(arm["overrides"])

    assert "data.dataset.name=pusht_expert_train.h5" in joined
    assert "+pixel_tag.mode=video" in joined
    assert arm["pair_suites"] == ["colour"]


def test_randgoal_decision_aligned_adds_cross_episode_plan_signal():
    cfg = run_randgoal_control.load_randgoal_control_configs()
    arm = cfg["arms"]["randgoal_decision_aligned_pred1"]
    joined = " ".join(arm["overrides"])

    assert "data.dataset.name=pusht_scripted_goal_train.lance" in joined
    assert "+loss.control.action_plan_weight=0.1" in joined
    assert "+loss.aligned_gradient_routing.enabled=true" in joined
    assert arm["pair_suites"] == ["t_position"]
