from pathlib import Path

from additional_files.bloop_tagged_diagnostics_campaign import (
    CHECKPOINTS,
    checkpoint_path,
)


def test_diagnostic_matrix_contains_matched_controls() -> None:
    labels = [spec.split("=", 1)[0] for spec in CHECKPOINTS]
    assert labels == ["jepa", "full", "cycle", "bloop"]
    assert len(set(CHECKPOINTS)) == 4


def test_checkpoint_path_uses_stablewm_checkpoint_root() -> None:
    root = Path("stablewm-root")
    path = checkpoint_path(root, "bloop=run/weights_epoch_10.pt")
    assert path == root / "checkpoints/run/weights_epoch_10.pt"
