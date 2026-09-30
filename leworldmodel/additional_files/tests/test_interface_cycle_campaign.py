from leworldmodel.additional_files.interface_cycle_campaign import (
    BENCHMARKS,
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
