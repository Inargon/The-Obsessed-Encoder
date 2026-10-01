import pytest

from additional_files.reacher_historical_eval_campaign import (
    ARMS,
    HISTORICAL_OVERRIDES,
    historical_dataset_dir,
    parse_success_rate,
)


def test_campaign_has_three_regressions_and_one_new_arm():
    assert list(ARMS) == [
        "jepa",
        "full",
        "inverse_reach",
        "predictor_only_cycle",
    ]
    assert [ARMS[name]["expected_success_rate"] for name in ARMS] == [
        0.90,
        0.86,
        0.76,
        None,
    ]


def test_historical_overrides_preserve_action_only_cache():
    assert "+cache_dir={dataset_dir}" in HISTORICAL_OVERRIDES
    assert "dataset.keys_to_cache=[action]" in HISTORICAL_OVERRIDES
    assert "seed=42" in HISTORICAL_OVERRIDES
    assert "solver.n_steps=30" in HISTORICAL_OVERRIDES


def test_historical_dataset_root_includes_datasets_component(tmp_path):
    assert historical_dataset_dir(tmp_path) == tmp_path / "datasets"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("{'success_rate': 76.0, 'episode_successes': []}", 0.76),
        ('{"success_rate": 0.62}', 0.62),
    ],
)
def test_parse_success_rate(text, expected):
    assert parse_success_rate(text) == expected
