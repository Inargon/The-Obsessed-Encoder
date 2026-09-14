from __future__ import annotations

from leworldmodel.additional_files import run_reacher_routing_diagnostic


def test_reacher_routing_diagnostic_has_matched_clean_arms():
    cfg = run_reacher_routing_diagnostic.load_diagnostic_configs()
    arms = cfg["arms"]

    assert set(arms) == {
        "reacher_clean_jepa_diag",
        "reacher_clean_control_only_diag",
        "reacher_clean_norm_scalar_diag",
        "reacher_clean_aligned_diag",
        "reacher_clean_soft25_diag",
        "reacher_clean_soft50_diag",
        "reacher_clean_soft75_diag",
    }

    for arm in arms.values():
        assert "data=dmc" in arm["overrides"]
        assert "+eval.config=reacher" in arm["eval_overrides"]
        assert not any("pixel_tag" in value for value in arm["overrides"])

    assert not any(
        "aligned_gradient_routing.enabled" in value
        for value in arms["reacher_clean_control_only_diag"]["overrides"]
    )
    assert any(
        "routing_mode=norm_matched_scalar" in value
        for value in arms["reacher_clean_norm_scalar_diag"]["overrides"]
    )
    for label, value in (("soft25", "0.25"), ("soft50", "0.5"), ("soft75", "0.75")):
        assert any(
            f"minimum_retention={value}" in override
            for override in arms[f"reacher_clean_{label}_diag"]["overrides"]
        )
