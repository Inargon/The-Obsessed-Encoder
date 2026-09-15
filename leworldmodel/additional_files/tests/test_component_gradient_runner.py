from __future__ import annotations

from leworldmodel.additional_files import run_component_gradient_diagnostic


def test_component_gradient_runner_has_matched_tasks():
    config = run_component_gradient_diagnostic.load_component_gradient_configs()
    assert set(config["arms"]) == {
        "pusht_tagged_component_grad_diag",
        "reacher_clean_component_grad_diag",
    }

    pusht = config["arms"]["pusht_tagged_component_grad_diag"]
    reacher = config["arms"]["reacher_clean_component_grad_diag"]
    assert "data.dataset.name=pusht_expert_train.h5" in pusht["overrides"]
    assert "+pixel_tag.mode=video" in pusht["overrides"]
    assert "data=dmc" in reacher["overrides"]
    assert not any("pixel_tag" in value for value in reacher["overrides"])

    for arm in (pusht, reacher):
        assert (
            "+loss.component_gradient_diagnostics.enabled=true"
            in arm["overrides"]
        )
        assert (
            "+loss.component_gradient_diagnostics.every_n_steps=100"
            in arm["overrides"]
        )
        assert "pair_suites" not in arm
