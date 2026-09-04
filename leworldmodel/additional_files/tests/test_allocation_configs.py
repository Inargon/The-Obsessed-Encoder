from __future__ import annotations

from leworldmodel.additional_files import run_allocations


def test_allocation_grid_has_exactly_three_new_method_arms():
    config = run_allocations.load_allocation_configs()

    assert set(config["arms"]) == {"conditional", "local_rank", "hybrid"}
    for name, arm in config["arms"].items():
        assert "data.dataset.name=pusht_expert_train.h5" in arm["overrides"]
        assert "+pixel_tag.mode=video" in arm["overrides"]
        assert f"+loss.allocation.mode={name}" in arm["overrides"] or name == "hybrid"
        assert arm["pair_suites"] == ["colour"]
