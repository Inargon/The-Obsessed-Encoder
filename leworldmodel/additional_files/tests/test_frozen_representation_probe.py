import pytest

from additional_files.diagnose_frozen_representation import parse_checkpoint


def test_parse_checkpoint():
    assert parse_checkpoint(
        "aligned=control_aligned_pred1_seed0/weights_epoch_7.pt"
    ) == (
        "aligned", "control_aligned_pred1_seed0", "weights_epoch_7.pt"
    )


@pytest.mark.parametrize("value", ["missing", "=run/file.pt", "x=run/file.ckpt"])
def test_parse_checkpoint_rejects_invalid_values(value):
    with pytest.raises(Exception):
        parse_checkpoint(value)
