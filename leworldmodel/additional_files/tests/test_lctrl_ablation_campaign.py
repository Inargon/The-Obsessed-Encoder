from __future__ import annotations

from leworldmodel.additional_files import lctrl_ablation_campaign


def _joined(arm: str) -> str:
    _, spec = lctrl_ablation_campaign.arm_spec(arm)
    return " ".join(spec["overrides"])


def test_all_arms_keep_matched_data_prediction_and_router() -> None:
    for arm in lctrl_ablation_campaign.ARMS:
        joined = _joined(arm)
        assert "data.dataset.name=pusht_expert_train.h5" in joined
        assert "+pixel_tag.mode=video" in joined
        assert "+loss.pred_weight=1.0" in joined
        assert "+loss.aligned_gradient_routing.enabled=true" in joined
        assert "+loss.control.max_horizon=3" in joined
        assert "+loss.control.inverse_target=sequence" in joined
        assert "+loss.control.inverse_weight=1.0" in joined


def test_full_has_all_three_components() -> None:
    joined = _joined("full")
    assert "+loss.control.mode=masked_reachability" in joined
    assert "+loss.control.cycle_weight=0.5" in joined
    assert "+loss.control.reachability_weight=0.1" in joined


def test_idm_only_zeroes_cycle_and_reachability_without_other_changes() -> None:
    joined = _joined("idm_only")
    assert "+loss.control.mode=masked_reachability" in joined
    assert "+loss.control.cycle_weight=0.0" in joined
    assert "+loss.control.reachability_weight=0.0" in joined
    assert "+loss.control.mask_keep_prob=0.5" in joined
    assert "+loss.control.temperature=0.1" in joined


def test_inverse_cycle_disables_only_reachability_loss() -> None:
    joined = _joined("inverse_cycle")
    assert "+loss.control.mode=masked_reachability" in joined
    assert "+loss.control.cycle_weight=0.5" in joined
    assert "+loss.control.reachability_weight=0.0" in joined


def test_inverse_reach_disables_only_cycle_loss() -> None:
    joined = _joined("inverse_reach")
    assert "+loss.control.mode=masked_reachability" in joined
    assert "+loss.control.cycle_weight=0.0" in joined
    assert "+loss.control.reachability_weight=0.1" in joined
