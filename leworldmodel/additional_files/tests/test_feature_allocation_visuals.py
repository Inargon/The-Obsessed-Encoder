import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "diagnose_feature_allocation_visuals.py"
SPEC = importlib.util.spec_from_file_location("diagnose_feature_allocation_visuals", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_path_length_sums_segment_norms():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    assert MODULE.path_length(points) == 10.0


def test_normalized_path_metrics_distinguish_tag_and_content():
    gallery = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    content = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    tag = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    result = MODULE.normalized_path_metrics(content, tag, gallery, seed=3)
    assert result["tag_to_content_path_length_ratio"] == 2.0
    assert result["content_tortuosity"] == 1.0
    assert result["tag_tortuosity"] == 1.0


def test_campaign_selects_only_jepa_and_bloop():
    text = (
        Path(__file__).resolve().parents[1] / "bloop_feature_allocation_campaign.py"
    ).read_text()
    assert '"jepa", "bloop"' in text
    assert '"--models", "jepa,bloop"' in text
    assert "BLOOP_FEATURE_ALLOCATION_JOB" in text
