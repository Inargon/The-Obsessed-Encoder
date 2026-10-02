from leworldmodel.additional_files.interface_cycle_campaign import (
    BENCHMARKS,
    DEFAULT_BENCHMARKS,
    ROUTINGS,
    benchmark_spec,
)


def test_interface_cycle_arms_change_only_cycle_gradient_scope():
    for benchmark in BENCHMARKS:
        _, spec = benchmark_spec(benchmark)
        overrides = spec["overrides"]

        assert "+loss.control.cycle_weight=0.5" in overrides
        assert "+loss.control.cycle_scope=predictor_only" in overrides
        assert "+loss.aligned_gradient_routing.enabled=true" in overrides


def test_interface_cycle_benchmarks_use_expected_tag_conditions():
    _, tagged = benchmark_spec("tagged_pusht")

    assert "+pixel_tag.mode=video" in tagged["overrides"]
    for benchmark in ("clean_reacher", "clean_cube", "clean_tworoom"):
        _, clean = benchmark_spec(benchmark)
        assert not any("pixel_tag" in value for value in clean["overrides"])


def test_cross_task_interface_cycle_uses_matched_clean_data():
    assert DEFAULT_BENCHMARKS == ("tagged_pusht", "clean_reacher")
    expected_data = {
        "clean_cube": "data=ogb",
        "clean_tworoom": "data=tworoom",
    }
    for benchmark, data_override in expected_data.items():
        _, spec = benchmark_spec(benchmark)
        assert data_override in spec["overrides"]
        assert "+loss.control.cycle_weight=0.5" in spec["overrides"]
        assert "+loss.control.cycle_scope=predictor_only" in spec["overrides"]


def test_bloop_cycle_replaces_only_the_embedding_router():
    assert "bloop" in ROUTINGS
    for benchmark in BENCHMARKS:
        _, aligned = benchmark_spec(benchmark, "aligned")
        _, bloop = benchmark_spec(benchmark, "bloop")
        aligned_overrides = aligned["overrides"]
        bloop_overrides = bloop["overrides"]

        assert not any(
            value.startswith("+loss.aligned_gradient_routing.")
            for value in bloop_overrides
        )
        assert "+loss.bloop.enabled=true" in bloop_overrides
        assert "+loss.bloop.decay=0.9" in bloop_overrides
        assert "+loss.bloop.auxiliary_weight=1.0" in bloop_overrides
        assert "+loss.control.cycle_scope=predictor_only" in bloop_overrides

        stripped_aligned = {
            value
            for value in aligned_overrides
            if not value.startswith("+loss.aligned_gradient_routing.")
        }
        stripped_bloop = {
            value
            for value in bloop_overrides
            if not value.startswith("+loss.bloop.")
        }
        assert stripped_aligned == stripped_bloop


def test_conflict_only_cycle_changes_only_the_routing_mode():
    for benchmark in BENCHMARKS:
        _, spec = benchmark_spec(benchmark, "conflict_only")
        overrides = spec["overrides"]
        assert (
            "+loss.aligned_gradient_routing.routing_mode=conflict_only"
            in overrides
        )
        assert "+loss.control.cycle_scope=predictor_only" in overrides
