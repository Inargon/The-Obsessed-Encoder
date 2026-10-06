from additional_files.render_ours_result_summary import RESULTS, wilson


def test_fixed_results_exclude_gcidm():
    assert [row[0] for row in RESULTS] == [
        "Clean PushT", "Tagged PushT", "Reacher", "TwoRoom", "Cube"
    ]
    assert [row[1] for row in RESULTS] == [46, 44, 44, 48, 40]


def test_wilson_contains_observed_rate():
    for _, successes, trials in RESULTS:
        low, high = wilson(successes, trials)
        assert low < successes / trials < high
