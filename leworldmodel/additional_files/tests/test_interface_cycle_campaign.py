from leworldmodel.additional_files.interface_cycle_campaign import (
    BENCHMARKS,
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
    _, clean = benchmark_spec("clean_reacher")

    assert "+pixel_tag.mode=video" in tagged["overrides"]
    assert not any("pixel_tag" in value for value in clean["overrides"])


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
