from __future__ import annotations

from leworldmodel.additional_files.delta_jepa_campaign import spec


def test_matched_campaign_uses_tagged_pusht_and_only_delta_supervision() -> None:
    campaign = spec()
    overrides = set(campaign["overrides"])

    assert "data.dataset.name=pusht_expert_train.h5" in overrides
    assert "+pixel_tag.mode=video" in overrides
    assert "+pixel_tag.size=5" in overrides
    assert "loss.sigreg.weight=0.0" in overrides
    assert "+loss.delta_jepa.enabled=true" in overrides
    assert "+loss.delta_jepa.max_horizon=3" in overrides
    assert "+loss.delta_jepa.action_weight=10.0" in overrides
    assert not any("loss.control" in option for option in overrides)
    assert not any("aligned_gradient_routing" in option for option in overrides)
