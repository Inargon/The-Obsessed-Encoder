"""LeWM runner semantics: command construction, summaries, arm wiring."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as lewm_run  # noqa: E402
from common import results_io, runner  # noqa: E402


def _args(tmp_path, **overrides):
    parser = argparse.ArgumentParser()
    runner.add_common_arguments(parser, default_tag="")
    parser.add_argument("--epochs", type=int, default=None)
    argv = ["--results-dir", str(tmp_path / "results"),
            "--data-dir", str(tmp_path / "data")]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", value]
    return parser.parse_args(argv)


def test_build_spec_prints_a_runnable_command(tmp_path):
    cfg = lewm_run.load_configs()
    args = _args(tmp_path, tag="demo")
    spec = lewm_run.build_spec("colored_square_episode",
                               cfg["arms"]["colored_square_episode"], cfg, 0, args)
    assert spec.name == "colored_square_episode_seed0"
    cmd = runner.format_command(spec, gpu="0")
    assert "train.py" in cmd
    assert "+pixel_tag.mode=video" in cmd and "+pixel_tag.size=5" in cmd
    assert "seed=0" in cmd and "+seed_everything=true" in cmd
    assert "+pair.suites=[colour]" in cmd
    assert "wandb.config.name=lewm_colored_square_episode_seed0" in cmd
    assert "WANDB_RUN_GROUP=demo" in cmd
    assert "metrics.jsonl" in cmd


def test_baseline_spec_carries_no_tag_or_pair_keys(tmp_path):
    cfg = lewm_run.load_configs()
    spec = lewm_run.build_spec("baseline", cfg["arms"]["baseline"], cfg, 0,
                               _args(tmp_path))
    joined = " ".join(spec.command)
    assert "pixel_tag" not in joined and "pair." not in joined


def test_summary_from_metrics(tmp_path):
    cfg = lewm_run.load_configs()
    args = _args(tmp_path)
    spec = lewm_run.build_spec("randgoal", cfg["arms"]["randgoal"], cfg, 0, args)
    run_dir = runner.run_dir_for(args.results_dir, spec.name)
    results_io.append_metrics(run_dir, {"step": 10, "fit/pred_loss": 0.5})
    results_io.append_metrics(run_dir, {"step": 20, "eval/success_rate": 0.25,
                                        "pair/backbone/same_tag/cos_mean": 0.9})
    lewm_run.write_summary(spec, args.results_dir)
    summary = results_io.read_summary(run_dir)
    assert summary["case"] == "lewm" and summary["config"] == "randgoal"
    assert summary["steps"] == 20
    assert summary["final_fit/pred_loss"] == 0.5
    assert summary["final_pair/backbone/same_tag/cos_mean"] == 0.9
