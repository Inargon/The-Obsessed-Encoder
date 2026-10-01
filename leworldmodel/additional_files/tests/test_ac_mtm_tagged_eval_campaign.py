import os

from additional_files.ac_mtm_tagged_eval_campaign import (
    COMPATIBILITY_OVERRIDES,
    EVALUATION_SEEDS,
    policy_path,
    runtime_env,
)


def test_external_evaluation_seeds_match_intact_protocol():
    assert EVALUATION_SEEDS == (0, 1, 42)


def test_noop_world_kwargs_are_deleted_for_pinned_pusht():
    assert COMPATIBILITY_OVERRIDES == (
        "~world.history_size",
        "~world.frame_skip",
    )


def test_policy_uses_absolute_epoch_ten_object_checkpoint_stem(tmp_path):
    assert policy_path(tmp_path, "run") == (
        tmp_path / "baselines/run/lewm_masked_action_nce_epoch_10"
    )


def test_official_source_precedes_project_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/existing")
    overlay = tmp_path / "overlay"
    entries = runtime_env(tmp_path, overlay)["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(tmp_path)
    assert entries[1] == str(overlay)
    assert entries[-1] == "/existing"
