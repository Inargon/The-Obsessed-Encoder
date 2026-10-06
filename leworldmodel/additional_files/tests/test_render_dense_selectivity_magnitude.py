import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "render_dense_selectivity_magnitude.py"
SPEC = importlib.util.spec_from_file_location("render_dense_selectivity_magnitude", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_absolute_tag_values_restores_magnitude():
    run = {
        "per_clip": {
            "tag_region_mass": {"context_cost": [0.5, 0.25]},
            "total_map_mass": {"context_cost": [10.0, 8.0]},
        }
    }
    np.testing.assert_allclose(
        MODULE.absolute_tag_values(run, "context_cost"),
        [5.0, 2.0],
    )


def test_renderer_uses_shared_absolute_scale_and_log_magnitude():
    text = PATH.read_text()
    assert "vmin=0.0, vmax=max(vmax, 1e-12)" in text
    assert 'axes[1].set_yscale("log")' in text
    assert "dense-tag-sensitivity-summary.png" in text
    assert "scaled_alpha = np.clip(heat / max(vmax, 1e-12)" in text
    assert "dense-example-paper-absolute.png" in text
