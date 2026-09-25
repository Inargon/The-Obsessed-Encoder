from leworldmodel.additional_files import randgoal_jepa_campaign as campaign


def test_jepa_spec_is_the_canonical_untreated_randgoal_arm():
    config, spec = campaign.jepa_spec()

    assert config["epochs"] == 10
    assert spec["overrides"] == [
        "data.dataset.name=pusht_scripted_goal_train.lance"
    ]
    assert spec["eval_overrides"] == [
        "+eval.dataset_name=pusht_scripted_goal_train.lance"
    ]
    assert spec["pair_suites"] == ["t_position"]

    flattened = " ".join(spec["overrides"] + spec["eval_overrides"])
    assert "loss.control" not in flattened
    assert "aligned_gradient_routing" not in flattened
    assert "pixel_tag" not in flattened


def test_campaign_has_a_fresh_checkpoint_namespace():
    source = campaign.Path(campaign.__file__).read_text()
    assert "randgoal_jepa" in source
    assert 'prefix = f"{campaign.name}_{phase}_randgoal_jepa"' in source
    assert "Refusing to overwrite existing run" in source


def test_training_disables_online_eval_and_keeps_final_checkpoint():
    source = campaign.Path(campaign.__file__).read_text()
    assert "++eval.every_n_steps=1000000" in source
    assert "++checkpoint.every_n_steps=5000" in source
    assert "weights_epoch_10.pt" in source
