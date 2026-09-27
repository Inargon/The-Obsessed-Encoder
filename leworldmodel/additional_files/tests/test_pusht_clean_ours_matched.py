from leworldmodel.additional_files import run_pusht_clean_jepa_matched
from leworldmodel.additional_files import run_pusht_clean_ours_matched


def test_matched_ours_differs_from_jepa_only_by_method(monkeypatch):
    monkeypatch.delenv("PUSHT_CLEAN_OURS_ARM", raising=False)
    ours = run_pusht_clean_ours_matched.load_pusht_clean_ours_matched_configs()
    jepa = run_pusht_clean_jepa_matched.load_pusht_clean_jepa_matched_configs()

    for key in (
        "epochs",
        "eval_every_n_steps",
        "pair_every_n_steps",
        "checkpoint_every_n_steps",
        "preview_episodes",
    ):
        assert ours[key] == jepa[key]

    assert set(ours["arms"]) == {"pusht_clean_ours_matched_full"}
    arm = ours["arms"]["pusht_clean_ours_matched_full"]
    assert arm["eval_overrides"] == []
    assert "data.dataset.name=pusht_expert_train.h5" in arm["overrides"]

    joined = " ".join(arm["overrides"])
    assert "pixel_tag" not in joined
    assert "+loss.control.enabled=true" in joined
    assert "+loss.control.cycle_weight=0.5" in joined
    assert "+loss.control.reachability_weight=0.1" in joined
    assert "+loss.aligned_gradient_routing.enabled=true" in joined


def test_smoke_uses_an_isolated_checkpoint_namespace(monkeypatch):
    monkeypatch.setenv(
        "PUSHT_CLEAN_OURS_ARM", "pusht_clean_ours_matched_smoke"
    )
    config = run_pusht_clean_ours_matched.load_pusht_clean_ours_matched_configs()
    assert set(config["arms"]) == {"pusht_clean_ours_matched_smoke"}
