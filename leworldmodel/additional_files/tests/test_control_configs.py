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
        "split_pred_e03_f1",
    }
    for name, arm in config["arms"].items():
        joined = " ".join(arm["overrides"])
        assert "data.dataset.name=pusht_expert_train.h5" in joined
        assert "+pixel_tag.mode=video" in joined
        expected_mode = (
            "masked_reachability"
            if name.startswith("masked_sequence_") or name == "split_pred_e03_f1"
            else name
        )
        assert f"+loss.control.mode={expected_mode}" in joined
        assert arm["pair_suites"] == ["colour"]

    assert "+loss.pred_weight=1.0" in config["arms"]["masked_sequence_pred1"]["overrides"]
    assert "+loss.pred_weight=0.3" in config["arms"]["masked_sequence_pred03"]["overrides"]
    assert "+loss.pred_weight=0.5" in config["arms"]["masked_sequence_pred05"]["overrides"]
    assert "+loss.pred_weight=0.7" in config["arms"]["masked_sequence_pred07"]["overrides"]
    assert "+loss.pred_weight=0.0" in config["arms"]["masked_sequence_pred0"]["overrides"]
    split = config["arms"]["split_pred_e03_f1"]["overrides"]
    assert "+loss.pred_weight=1.0" in split
    assert "+loss.encoder_pred_weight=0.3" in split
    assert "+loss.control.inverse_target=sequence" in split
