import numpy as np

from additional_files.diagnose_counterfactual_storyboard import (
    normalized_cost_rmse,
    select_examples,
)


def test_normalized_cost_rmse_is_zero_for_identical_costs():
    costs = np.asarray([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]])
    np.testing.assert_allclose(normalized_cost_rmse(costs, costs), 0.0)


def test_example_selection_prefers_jepa_only_flip():
    stable = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    jepa_changed = [[1.0, 0.0], [0.0, 1.0], [0.2, 0.8]]
    runs = {
        "jepa": {"reference_costs": stable, "changed_costs": jepa_changed},
        "bloop": {"reference_costs": stable, "changed_costs": stable},
    }
    assert select_examples(runs, 1) == [0]
