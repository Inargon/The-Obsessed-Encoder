from __future__ import annotations

from leworldmodel.additional_files import run_control


def test_control_grid_has_relation_core_sweep():
    config = run_control.load_control_configs()

    assert set(config["arms"]) == {
        "direct_reachability",
        "factorized_reachability",
        "multi_horizon_idm",
        "masked_reachability",
        "masked_sequence_pred1",
        "masked_sequence_pred03",
        "masked_sequence_pred05",
        "masked_sequence_pred07",
        "masked_sequence_pred0",
        "control_aligned_pred1",
        "control_parallel_pred1",
        "control_shuffled_pred1",
        "control_norm_matched_scalar_pred1",
        "control_retention_matched_shuffled_pred1",
    }
    for name, arm in config["arms"].items():
        joined = " ".join(arm["overrides"])
        assert "data.dataset.name=pusht_expert_train.h5" in joined
        assert "+pixel_tag.mode=video" in joined
        expected_mode = (
            "masked_reachability"
            if name.startswith("masked_sequence_") or name.startswith("control_")
            else name
        )
        assert f"+loss.control.mode={expected_mode}" in joined
        assert arm["pair_suites"] == ["colour"]

    assert "+loss.pred_weight=1.0" in config["arms"]["masked_sequence_pred1"]["overrides"]
    assert "+loss.pred_weight=0.3" in config["arms"]["masked_sequence_pred03"]["overrides"]
    assert "+loss.pred_weight=0.5" in config["arms"]["masked_sequence_pred05"]["overrides"]
    assert "+loss.pred_weight=0.7" in config["arms"]["masked_sequence_pred07"]["overrides"]
    assert "+loss.pred_weight=0.0" in config["arms"]["masked_sequence_pred0"]["overrides"]
    aligned = config["arms"]["control_aligned_pred1"]["overrides"]
    assert "+loss.pred_weight=1.0" in aligned
    assert "+loss.aligned_gradient_routing.enabled=true" in aligned
    assert "+loss.control.inverse_target=sequence" in aligned
    parallel = config["arms"]["control_parallel_pred1"]["overrides"]
    assert "+loss.aligned_gradient_routing.orthogonal_mode=drop" in parallel
    shuffled = config["arms"]["control_shuffled_pred1"]["overrides"]
    assert "+loss.aligned_gradient_routing.shuffle_control=true" in shuffled
    scalar = config["arms"]["control_norm_matched_scalar_pred1"]["overrides"]
    assert "+loss.aligned_gradient_routing.routing_mode=norm_matched_scalar" in scalar
    matched = config["arms"]["control_retention_matched_shuffled_pred1"]["overrides"]
    assert (
        "+loss.aligned_gradient_routing.routing_mode=retention_matched_shuffled"
        in matched
    )
