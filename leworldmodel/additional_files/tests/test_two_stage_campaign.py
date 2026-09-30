from pathlib import Path

from additional_files.two_stage_campaign import ARMS, stage2_spec


def test_stage2_pilot_contains_optimizer_reset_control():
    assert tuple(ARMS) == (
        "unfrozen_reset",
        "frozen_no_cycle",
        "frozen_predictor_cycle",
    )


def test_stage2_arms_change_only_declared_boundary(monkeypatch):
    base = {
        "overrides": [
            "+loss.control.cycle_weight=0.5",
            "+loss.control.max_horizon=1",
        ]
    }
    monkeypatch.setattr(
        "additional_files.two_stage_campaign.horizon_arm_spec",
        lambda arm, condition: ({"epochs": 10}, base),
    )
    source = Path("/tmp/stage1.pt")
    _, reset = stage2_spec("unfrozen_reset", "clean", source)
    _, frozen = stage2_spec("frozen_predictor_cycle", "clean", source)
    assert "+stage2.freeze_representation=false" in reset["overrides"]
    assert "+stage2.freeze_representation=true" in frozen["overrides"]
    assert "+loss.control.cycle_scope=predictor_only" in frozen["overrides"]
    assert any(str(source) in value for value in frozen["overrides"])
