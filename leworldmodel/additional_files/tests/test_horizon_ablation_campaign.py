from __future__ import annotations

from leworldmodel.additional_files import horizon_ablation_campaign


def _overrides(arm: str) -> list[str]:
    _, spec = horizon_ablation_campaign.arm_spec(arm)
    return spec["overrides"]


def _value(overrides: list[str], key: str) -> str:
    prefix = f"+loss.control.{key}="
    values = [item.removeprefix(prefix) for item in overrides if item.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def test_horizon_arms_change_only_horizon_and_reach_aggregation() -> None:
    reference = _overrides("h123_pair")
    ignored = {"max_horizon", "reach_aggregation"}

    def fixed(overrides: list[str]) -> set[str]:
        return {
            item
            for item in overrides
            if not any(item.startswith(f"+loss.control.{key}=") for key in ignored)
        }

    for arm in horizon_ablation_campaign.ARMS:
        assert fixed(_overrides(arm)) == fixed(reference)


def test_horizon_arm_matrix_is_exact() -> None:
    expected = {
        "h1_pair": ("1", "pair"),
        "h12_pair": ("2", "pair"),
        "h123_pair": ("3", "pair"),
        "h123_balanced": ("3", "horizon"),
    }
    for arm, (horizon, aggregation) in expected.items():
        overrides = _overrides(arm)
        assert _value(overrides, "max_horizon") == horizon
        assert _value(overrides, "reach_aggregation") == aggregation


def test_all_arms_preserve_full_method_and_tagged_data() -> None:
    for arm in horizon_ablation_campaign.ARMS:
        joined = " ".join(_overrides(arm))
        assert "data.dataset.name=pusht_expert_train.h5" in joined
        assert "+pixel_tag.mode=video" in joined
        assert "+loss.pred_weight=1.0" in joined
        assert "+loss.aligned_gradient_routing.enabled=true" in joined
        assert "+loss.control.mode=masked_reachability" in joined
        assert "+loss.control.inverse_target=sequence" in joined
        assert "+loss.control.inverse_weight=1.0" in joined
        assert "+loss.control.cycle_weight=0.5" in joined
        assert "+loss.control.reachability_weight=0.1" in joined


def test_clean_arms_remove_only_tag_specific_protocol() -> None:
    for arm in horizon_ablation_campaign.ARMS:
        _, spec = horizon_ablation_campaign.arm_spec(arm, condition="clean")
        joined = " ".join(spec["overrides"])
        assert "data.dataset.name=pusht_expert_train.h5" in joined
        assert "pixel_tag" not in joined
        assert not spec.get("eval_overrides")
        assert "pair_suites" not in spec
        assert "+loss.pred_weight=1.0" in joined
        assert "+loss.aligned_gradient_routing.enabled=true" in joined
