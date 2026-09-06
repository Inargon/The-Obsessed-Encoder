from leworldmodel.additional_files import run_action_binding


def test_action_binding_grid_is_a_matched_three_arm_diagnostic():
    config = run_action_binding.load_action_binding_configs()
    assert list(config["arms"]) == [
        "bank_sequence_idm",
        "counterfactual_binding",
        "binding_geometry_oracle",
    ]
    modes = {
        "bank_sequence_idm": "bank_idm",
        "counterfactual_binding": "binding",
        "binding_geometry_oracle": "binding_geometry",
    }
    for name, arm in config["arms"].items():
        joined = " ".join(arm["overrides"])
        assert "+loss.pred_weight=0.3" in joined
        assert "+loss.control.inverse_target=sequence" in joined
        assert "+loss.action_binding.enabled=true" in joined
        assert f"+loss.action_binding.mode={modes[name]}" in joined
        assert arm["pair_suites"] == ["colour"]
