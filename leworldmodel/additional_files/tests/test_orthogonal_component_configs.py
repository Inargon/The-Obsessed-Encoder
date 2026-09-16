from leworldmodel.additional_files import run_pusht_orthogonal_components
from leworldmodel.additional_files import run_reacher_orthogonal_components


def _joined(arm):
    return " ".join(arm["overrides"])


def test_pusht_component_arms_are_matched_and_tagged():
    cfg = run_pusht_orthogonal_components.load_pusht_orthogonal_component_configs()
    assert set(cfg["arms"]) == {
        "pusht_tagged_orthogonal_only_raw_pred1",
        "pusht_tagged_orthogonal_only_gated_pred1",
    }

    raw = _joined(cfg["arms"]["pusht_tagged_orthogonal_only_raw_pred1"])
    gated = _joined(cfg["arms"]["pusht_tagged_orthogonal_only_gated_pred1"])
    for joined in (raw, gated):
        assert "+pixel_tag.mode=video" in joined
        assert "+loss.aligned_gradient_routing.enabled=true" in joined
        assert "+loss.pred_weight=1.0" in joined
    assert "routing_mode=orthogonal_only" in raw
    assert "routing_mode=gated_orthogonal_only" in gated


def test_reacher_component_arms_are_matched_and_clean():
    cfg = run_reacher_orthogonal_components.load_reacher_orthogonal_component_configs()
    assert set(cfg["arms"]) == {
        "reacher_clean_orthogonal_only_raw",
        "reacher_clean_orthogonal_only_gated",
    }

    raw = _joined(cfg["arms"]["reacher_clean_orthogonal_only_raw"])
    gated = _joined(cfg["arms"]["reacher_clean_orthogonal_only_gated"])
    for joined in (raw, gated):
        assert "data=dmc" in joined
        assert "pixel_tag" not in joined
        assert "+loss.aligned_gradient_routing.enabled=true" in joined
    assert "routing_mode=orthogonal_only" in raw
    assert "routing_mode=gated_orthogonal_only" in gated
