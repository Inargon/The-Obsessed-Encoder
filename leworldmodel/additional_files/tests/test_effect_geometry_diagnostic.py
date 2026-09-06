import json

import torch

from leworldmodel.additional_files.diagnose_effect_geometry import (
    rho_curve,
    success_at,
)


def test_rho_curve_recovers_constant_and_alternating_timescales():
    torch.manual_seed(0)
    base = torch.randn(64, 1, 8)
    slow = base.expand(-1, 4, -1) + 0.001 * torch.randn(64, 4, 8)
    alternating = base * torch.tensor([1.0, -1.0, 1.0, -1.0]).view(1, 4, 1)
    assert rho_curve(slow)["rho1"] > 0.99
    assert rho_curve(alternating)["rho1"] < -0.99


def test_success_at_uses_latest_evaluation_not_after_step(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 2000, "eval/success_rate": 0.1},
        {"step": 4000, "eval/success_rate": 0.3},
        {"step": 6000, "eval/success_rate": 0.2},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert success_at(path, 5000) == 0.3
