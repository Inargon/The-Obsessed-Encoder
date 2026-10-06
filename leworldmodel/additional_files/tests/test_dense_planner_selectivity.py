import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "diagnose_dense_planner_selectivity.py"
SPEC = importlib.util.spec_from_file_location("diagnose_dense_planner_selectivity", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_grid_windows_partition_every_pixel_once():
    windows = MODULE.grid_windows(11, 13, 4)
    coverage = np.zeros((11, 13), dtype=int)
    for y0, y1, x0, x1 in windows:
        coverage[y0:y1, x0:x1] += 1
    assert len(windows) == 16
    np.testing.assert_array_equal(coverage, 1)


def test_normalized_cost_metrics_detect_ranking_and_selection_change():
    reference = np.array([0.0, 1.0, 2.0])
    changed = np.array([2.0, 1.0, 0.0])
    rmse, reversal, selection = MODULE.normalized_cost_metrics(reference, changed)
    assert rmse[0] > 0
    assert reversal[0] == 1.0
    assert selection[0] == 1.0


def test_tag_region_mass_uses_cells_touching_native_tag():
    windows = MODULE.grid_windows(12, 12, 3)
    values = np.zeros(9)
    values[0] = 3.0
    values[1] = 1.0
    assert MODULE.tag_region_mass(values, windows, tag_size=4) == 0.75


def test_unit_mass_and_baseline_agreement_are_scale_invariant():
    values = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(MODULE.unit_mass(values).sum(), 1.0)
    assert np.isclose(MODULE.nonnegative_cosine(values, 7 * values), 1.0)


def test_campaign_is_matched_jepa_versus_ours():
    text = (ROOT / "dense_planner_selectivity_campaign.py").read_text()
    assert "jepa=colored_square_episode_seed0" in text
    assert "ours=interface-cycle-20261002-064431" in text
    assert "DENSE_PLANNER_SELECTIVITY_JOB" in text


def test_planner_costs_isolate_outer_batch_from_candidate_axis():
    text = PATH.read_text()
    assert 'goal_emb = model.encode({"pixels": goal})["emb"]' in text
    assert '"goal_emb": goal_emb[:, None].expand(' in text
    assert "info = model.rollout(info, plans.clone())" in text
    assert "return model.criterion(info).reshape(batch, candidates)" in text
