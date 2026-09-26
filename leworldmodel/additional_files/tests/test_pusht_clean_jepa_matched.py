from leworldmodel.additional_files import run_pusht_clean_jepa_matched


def test_matched_clean_jepa_changes_only_the_run_namespace(monkeypatch):
    monkeypatch.delenv("PUSHT_CLEAN_JEPA_ARM", raising=False)
    config = run_pusht_clean_jepa_matched.load_pusht_clean_jepa_matched_configs()

    assert set(config["arms"]) == {"pusht_clean_jepa_matched_full"}
    arm = config["arms"]["pusht_clean_jepa_matched_full"]
    assert arm["overrides"] == ["data.dataset.name=pusht_expert_train.h5"]
    assert arm["eval_overrides"] == []

    joined = " ".join(arm["overrides"])
    assert "pixel_tag" not in joined
    assert "loss.control" not in joined
    assert "aligned_gradient_routing" not in joined


def test_smoke_gets_an_isolated_checkpoint_namespace(monkeypatch):
    monkeypatch.setenv("PUSHT_CLEAN_JEPA_ARM", "pusht_clean_jepa_matched_smoke")
    config = run_pusht_clean_jepa_matched.load_pusht_clean_jepa_matched_configs()
    assert set(config["arms"]) == {"pusht_clean_jepa_matched_smoke"}


def test_main_can_invoke_installed_loader_without_recursion(monkeypatch):
    def fake_runner_main():
        config = run_pusht_clean_jepa_matched.base_runner.load_configs()
        assert set(config["arms"]) == {"pusht_clean_jepa_matched_full"}
        return 0

    monkeypatch.delenv("PUSHT_CLEAN_JEPA_ARM", raising=False)
    monkeypatch.setattr(
        run_pusht_clean_jepa_matched.base_runner, "main", fake_runner_main
    )
    assert run_pusht_clean_jepa_matched.main() == 0
