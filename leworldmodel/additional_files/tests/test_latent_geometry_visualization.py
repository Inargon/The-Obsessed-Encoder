import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "diagnose_latent_geometry.py"
SPEC = importlib.util.spec_from_file_location("diagnose_latent_geometry", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_state_features_wraps_angle_continuously():
    state = np.array([[1.0, 2.0, 3.0, 4.0, np.pi]])
    features = MODULE.state_features(state)
    np.testing.assert_allclose(features[0, :4], state[0, :4])
    np.testing.assert_allclose(features[0, 4:], [0.0, -1.0], atol=1e-7)


def test_straight_path_has_exact_endpoints():
    endpoints = np.array([[0.0, 2.0], [4.0, 6.0]])
    path = MODULE.straight_path(endpoints, 5)
    np.testing.assert_allclose(path[0], endpoints[0])
    np.testing.assert_allclose(path[-1], endpoints[-1])
    np.testing.assert_allclose(path[2], [2.0, 4.0])


def test_nearest_indices_retrieves_expected_points():
    gallery = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    queries = np.array([[0.1, 0.0], [3.7, 0.0]])
    assert MODULE.nearest_indices(queries, gallery).tolist() == [0, 2]


def test_oracle_path_has_unit_ratios_and_zero_error():
    path = MODULE.straight_path(np.array([[0.0, 0.0], [1.0, 2.0]]), 8)
    metrics = MODULE.path_metrics(path, path)
    assert metrics["state_rmse_to_oracle"] == 0.0
    assert np.isclose(metrics["path_length_ratio"], 1.0)
    assert np.isclose(metrics["control_cost_ratio"], 1.0)


def test_campaign_names_all_expected_outputs():
    text = (
        Path(__file__).resolve().parents[1] / "bloop_latent_geometry_campaign.py"
    ).read_text()
    assert "BLOOP_LATENT_GEOMETRY_JOB" in text
    assert "diagnose_latent_geometry.py" in text
    assert "--gallery-size" in text
