from additional_files.intact_diagnostics_campaign import (
    DEFAULT_RUN,
    PINNED_INTACT_COMMIT,
)


def test_intact_diagnostic_targets_frozen_formal_checkpoint():
    assert DEFAULT_RUN.endswith("goal_pusht_tagged_s3072")
    assert len(PINNED_INTACT_COMMIT) == 40
