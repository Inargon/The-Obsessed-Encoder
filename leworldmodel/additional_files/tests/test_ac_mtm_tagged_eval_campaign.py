from additional_files.ac_mtm_tagged_eval_campaign import (
    EVALUATION_SEEDS,
    policy_path,
)


def test_external_evaluation_seeds_match_intact_protocol():
    assert EVALUATION_SEEDS == (0, 1, 42)


def test_policy_uses_absolute_epoch_ten_object_checkpoint_stem(tmp_path):
    assert policy_path(tmp_path, "run") == (
        tmp_path / "baselines/run/lewm_masked_action_nce_epoch_10"
    )
