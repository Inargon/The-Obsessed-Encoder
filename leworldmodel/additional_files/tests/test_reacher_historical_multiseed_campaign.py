from additional_files.reacher_historical_multiseed_campaign import (
    DEFAULT_SEEDS,
    METHODS,
    cells,
    parse_seeds,
    source_checkpoint,
)


def test_default_matrix_is_paired_five_seed_evaluation():
    assert DEFAULT_SEEDS == (0, 1, 2, 3, 42)
    assert METHODS == ("full", "predictor_only_cycle")
    assert cells(DEFAULT_SEEDS) == [
        ("full", 0),
        ("full", 1),
        ("full", 2),
        ("full", 3),
        ("full", 42),
        ("predictor_only_cycle", 0),
        ("predictor_only_cycle", 1),
        ("predictor_only_cycle", 2),
        ("predictor_only_cycle", 3),
        ("predictor_only_cycle", 42),
    ]


def test_seed_parser_and_checkpoint_path(tmp_path):
    assert parse_seeds("0,1,42") == (0, 1, 42)
    assert source_checkpoint(tmp_path, "matrix", "full") == (
        tmp_path
        / "checkpoints/matrix_full/weights_epoch_10_historical_v2.pt"
    )
