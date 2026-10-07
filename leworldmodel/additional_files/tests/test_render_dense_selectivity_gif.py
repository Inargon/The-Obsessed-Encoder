import numpy as np

from additional_files.render_dense_selectivity_gif import (
    absolute_tag_values,
    relative_to_uniform,
    select_positions,
)


def test_uniform_heat_maps_to_zero_log_ratio():
    np.testing.assert_allclose(relative_to_uniform(np.ones((2, 2))), 0.0)


def test_relative_heat_is_clipped_to_presentation_range():
    values = relative_to_uniform(np.asarray([[100.0, 0.0], [0.0, 0.0]]))
    assert values.max() == 2.0
    assert values.min() == -2.0


def test_absolute_tag_values_restores_magnitude():
    run = {"per_clip": {
        "tag_region_mass": {"context_cost": [0.5, 0.25]},
        "total_map_mass": {"context_cost": [10.0, 8.0]},
    }}
    np.testing.assert_allclose(absolute_tag_values(run, "context_cost"), [5.0, 2.0])


def test_selection_uses_largest_matched_share_gap():
    data = {"runs": {
        "jepa": {"per_clip": {"tag_region_mass": {"context_cost": [0.7, 0.8, 0.6]}}},
        "ours": {"per_clip": {"tag_region_mass": {"context_cost": [0.2, 0.6, 0.0]}}},
    }}
    assert select_positions(data, "context_cost", 2) == [2, 0]
