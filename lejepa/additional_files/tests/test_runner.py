"""Runner semantics with a stub subprocess: skip/force, GPU pool, commands."""
import argparse
import os
import threading

from common import results_io, runner
from lejepa.additional_files import run as lejepa_run


def _args(tmp_path, **overrides):
    parser = argparse.ArgumentParser()
    runner.add_common_arguments(parser, default_tag="")
    argv = ["--results-dir", str(tmp_path / "results"),
            "--data-dir", str(tmp_path / "data")]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", value]
    return parser.parse_args(argv)


def test_build_spec_prints_a_runnable_command(tmp_path):
    args = _args(tmp_path, tag="demo")
    overrides = lejepa_run.load_configs()["watermarked"]
    spec = lejepa_run.build_spec("watermarked", overrides, 1, args)
    assert spec.name == "watermarked_seed1"
    cmd = runner.format_command(spec, gpu="0")
    assert "CUDA_VISIBLE_DEVICES=0" in cmd
    assert "lejepa_minimal.py" in cmd
    assert "+watermark_repeat=true" in cmd
    assert "+seed=1" in cmd
    assert "+wandb_group=demo" in cmd
    assert "+run_dir=" in cmd and "watermarked_seed1" in cmd


def test_skip_if_summary_force_and_rerun(tmp_path):
    args = _args(tmp_path)
    spec = lejepa_run.build_spec("clean", lejepa_run.load_configs()["clean"], 0, args)
    assert runner.should_run(args.results_dir, spec, force=False, rerun=[])
    results_io.write_summary(runner.run_dir_for(args.results_dir, spec.name), {})
    assert not runner.should_run(args.results_dir, spec, force=False, rerun=[])
    assert runner.should_run(args.results_dir, spec, force=True, rerun=[])
    assert runner.should_run(args.results_dir, spec, force=False,
                             rerun=["clean_seed0"])


def test_gpu_pool_dispatches_every_run_once(tmp_path):
    args = _args(tmp_path)
    configs = lejepa_run.load_configs()
    specs = [lejepa_run.build_spec(c, configs[c], s, args)
             for c in configs for s in (0, 1)]
    seen = {}
    lock = threading.Lock()

    def stub_launch(spec, gpu):
        with lock:
            seen[spec.name] = gpu
        return 0

    results = runner.execute_pool(specs, ["0", "1"], launch_fn=stub_launch)
    assert set(seen) == {s.name for s in specs}
    assert set(seen.values()) <= {"0", "1"}
    assert all(code == 0 for code in results.values())
    assert runner.summarize_results(results) == 0


def test_failed_run_fails_the_invocation_and_writes_no_summary(tmp_path):
    args = _args(tmp_path)
    spec = lejepa_run.build_spec("clean", lejepa_run.load_configs()["clean"], 0, args)

    def failing_launch(spec, gpu):
        return 3

    results = runner.execute_pool(
        [spec], ["0"], launch_fn=failing_launch,
        on_success=lambda s: lejepa_run.write_summary(s, args.results_dir))
    assert results[spec.name] == 3
    assert runner.summarize_results(results) == 1
    assert not results_io.has_summary(runner.run_dir_for(args.results_dir, spec.name))


def test_successful_run_gets_a_summary_from_its_metrics(tmp_path):
    args = _args(tmp_path)
    spec = lejepa_run.build_spec("watermarked",
                                 lejepa_run.load_configs()["watermarked"], 0, args)
    run_dir = runner.run_dir_for(args.results_dir, spec.name)

    def stub_launch(spec, gpu):
        results_io.append_metrics(run_dir, {"step": 10, "train/lejepa": 0.5})
        results_io.append_metrics(run_dir, {"step": 20, "test/acc": 0.3,
                                            "pair/null/backbone": 0.01})
        return 0

    results = runner.execute_pool(
        [spec], ["0"], launch_fn=stub_launch,
        on_success=lambda s: lejepa_run.write_summary(s, args.results_dir))
    assert results[spec.name] == 0
    summary = results_io.read_summary(run_dir)
    assert summary["config"] == "watermarked"
    assert summary["steps"] == 20
    assert summary["final_test/acc"] == 0.3
    assert summary["final_pair/null/backbone"] == 0.01
    assert summary["provenance"]["torch"]


def test_zero_metrics_run_is_a_failure(tmp_path):
    """Exit 0 with no metrics written must not produce a done-marker."""
    args = _args(tmp_path)
    spec = lejepa_run.build_spec("clean", lejepa_run.load_configs()["clean"], 0, args)
    results = runner.execute_pool(
        [spec], ["0"], launch_fn=lambda s, g: 0,
        on_success=lambda s: lejepa_run.write_summary(s, args.results_dir))
    assert results[spec.name] == 1
    assert not results_io.has_summary(runner.run_dir_for(args.results_dir, spec.name))


def test_parse_seeds_rejects_zero_count():
    assert runner.parse_seeds("2,") == [2]  # explicit single seed
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        runner.parse_seeds("0")


def test_pool_records_a_crashed_launcher_as_failure(tmp_path):
    args = _args(tmp_path)
    spec = lejepa_run.build_spec("clean", lejepa_run.load_configs()["clean"], 0, args)

    def exploding_launch(spec, gpu):
        raise RuntimeError("boom")

    results = runner.execute_pool([spec], ["0"], launch_fn=exploding_launch)
    assert results[spec.name] == 1
    assert runner.summarize_results(results) == 1


def test_clear_run_dir_wipes_previous_attempt(tmp_path):
    args = _args(tmp_path)
    run_dir = runner.run_dir_for(args.results_dir, "clean_seed0")
    results_io.append_metrics(run_dir, {"step": 5, "stale": 1.0})
    runner.clear_run_dir(args.results_dir, "clean_seed0")
    assert not os.path.exists(run_dir)
    runner.clear_run_dir(args.results_dir, "clean_seed0")  # idempotent
