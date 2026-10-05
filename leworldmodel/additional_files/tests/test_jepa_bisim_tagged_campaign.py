from pathlib import Path

from additional_files.jepa_bisim_tagged_campaign import (
    REFERENCE_CHECKPOINTS,
    matched_spec,
    run_name,
)


def test_matched_spec_uses_compact_planning_state_and_no_ours_router() -> None:
    overrides = matched_spec()["overrides"]
    assert "embed_dim=32" in overrides
    assert "model.projector.input_dim=192" in overrides
    assert "+loss.bisimulation.enabled=true" in overrides
    assert not any("bloop" in value for value in overrides)
    assert not any("aligned_gradient" in value for value in overrides)


def test_primary_diagnostic_has_only_requested_comparators() -> None:
    labels = [spec.split("=", 1)[0] for spec in REFERENCE_CHECKPOINTS]
    assert labels == ["jepa", "ours"]


def test_run_names_separate_smoke_and_full() -> None:
    root = Path("jepa-bisim-tagged-test")
    assert run_name(root, 0, True).endswith("_smoke_seed0")
    assert run_name(root, 0, False).endswith("_full_seed0")
