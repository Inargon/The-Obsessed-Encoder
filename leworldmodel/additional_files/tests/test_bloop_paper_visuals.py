import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALLOCATION = load("diagnose_feature_allocation_visuals.py")
MECHANISM = load("plot_bloop_training_mechanism.py")
ROLLOUTS = load("compose_matched_rollouts.py")


def test_batched_path_metrics_keep_matched_trajectory_axis():
    gallery = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    content = np.array([
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [2.0, 0.0]],
    ])
    tag = np.array([
        [[0.0, 0.0], [0.0, 2.0]],
        [[0.0, 0.0], [0.0, 1.0]],
    ])
    result = ALLOCATION.batched_path_metrics(content, tag, gallery, seed=5)
    np.testing.assert_allclose(result["tag_to_content_path_length_ratio"], [2.0, 0.5])


def test_path_summary_reports_geometric_mean_and_fraction():
    result = ALLOCATION.summarize_paths(
        {
            "content_path_length_over_gallery_median": np.array([1.0, 2.0]),
            "tag_path_length_over_gallery_median": np.array([2.0, 1.0]),
            "tag_to_content_path_length_ratio": np.array([2.0, 0.5]),
        },
        seed=3,
    )
    ratio = result["tag_to_content_path_length_ratio"]
    assert np.isclose(ratio["geometric_mean"], 1.0)
    assert ratio["fraction_below_one"] == 0.5


def test_rollout_categories_are_matched_by_episode_index():
    result = ROLLOUTS.episode_categories([0, 2], [1, 2], count=4)
    assert result == {
        "repair_only": [1],
        "both_success": [2],
        "jepa_only": [0],
        "both_fail": [3],
    }


def test_mechanism_reader_accepts_fit_prefix(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"step": 7, "fit/bloop_projection_ratio": -0.5}) + "\n",
        encoding="utf-8",
    )
    steps, values = MECHANISM.extract(
        MECHANISM.read_records(path), "bloop_projection_ratio"
    )
    np.testing.assert_array_equal(steps, [7])
    np.testing.assert_allclose(values, [-0.5])


def test_campaign_contains_all_three_visual_families():
    text = (ROOT / "bloop_paper_visuals_campaign.py").read_text(encoding="utf-8")
    assert "--trajectories\", 128" in text
    assert "plot_bloop_training_mechanism.py" in text
    assert "compose_matched_rollouts.py" in text
    assert "BLOOP_PAPER_VISUALS_JOB" in text
