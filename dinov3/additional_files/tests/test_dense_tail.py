"""Dense-tail semantics with a stub probe subprocess: milestone discovery,
fill-missing, failed-row bookkeeping, metrics rows."""
import argparse
import json
import os

import pytest

from common import results_io, runner
from additional_files import run as dinov3_run


def _args(tmp_path, **overrides):
    parser = argparse.ArgumentParser()
    runner.add_common_arguments(parser, default_tag="")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--head-iters", type=int, default=4800)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--extra-opts", default="")
    parser.add_argument("--dense-extra-opts", default="")
    argv = ["--results-dir", str(tmp_path / "results"),
            "--data-dir", str(tmp_path / "data")]
    for key, value in overrides.items():
        if value is True:
            argv.append(f"--{key.replace('_', '-')}")
        else:
            argv += [f"--{key.replace('_', '-')}", str(value)]
    return parser.parse_args(argv)


def _make_milestones(run_dir, iterations, *, missing_ckpt=(), stray=()):
    for it in iterations:
        d = os.path.join(run_dir, "train", "eval", f"training_{it}")
        os.makedirs(d)
        if it not in missing_ckpt:
            open(os.path.join(d, "teacher_checkpoint.pth"), "wb").close()
    for name in stray:
        os.makedirs(os.path.join(run_dir, "train", "eval", name))


def test_milestone_discovery(tmp_path):
    run_dir = str(tmp_path / "watermarked_seed0")
    _make_milestones(run_dir, [4999, 9999, 14999], missing_ckpt=(9999,),
                     stray=("manual_100", "notes"))
    assert dinov3_run.local_milestones(run_dir) == [4999, 14999]
    assert dinov3_run.local_milestones(str(tmp_path / "nope")) == []


def test_pending_milestones_keyed_by_instrument():
    rows = [
        {"teacher_iteration": "4999", "status": "ok", "head_iters": "4800"},
        {"teacher_iteration": "9999", "status": "failed", "head_iters": "4800"},
        {"teacher_iteration": "14999", "status": "ok", "head_iters": "8"},  # smoke row
    ]
    pending = dinov3_run.pending_milestones([4999, 9999, 14999], rows, head_iters=4800)
    # ok and failed both count as done; the smoke-scale row does not satisfy the
    # calibrated instrument.
    assert pending == [14999]


def _stub_spec(tmp_path, args, config="watermarked"):
    arms = dinov3_run.load_arms()
    return dinov3_run.build_spec(config, arms[config], 0, args)


def test_dense_tail_fills_missing_and_appends_metrics(tmp_path, monkeypatch):
    args = _args(tmp_path)
    spec = _stub_spec(tmp_path, args)
    run_dir = runner.run_dir_for(args.results_dir, spec.name)
    _make_milestones(run_dir, [4999, 9999])

    launched = []

    def stub_child(command, env):
        launched.append(command)
        assert "--watermark-json" in command  # watermarked arm measured at its own point
        payload = json.loads(command[command.index("--watermark-json") + 1])
        assert payload["opacity"] == 0.1 and payload["repeat"] is True
        return "ok", {"rmse": 0.5, "abs_rel": 0.2, "a1": 0.8,
                      "n_test_images": 654, "wall_clock_s": 1.0}

    monkeypatch.setattr(dinov3_run, "_run_probe_child", stub_child)
    counts = dinov3_run.run_dense_tail(spec, args, gpu="0")
    assert counts == {"ok": 2, "failed": 0}
    assert len(launched) == 2

    rows = dinov3_run.read_dense_rows(run_dir)
    assert [int(r["teacher_iteration"]) for r in rows] == [4999, 9999]
    metrics = results_io.read_metrics(run_dir)
    dense_rows = [m for m in metrics if "dense/nyu_rmse" in m]
    assert [m["step"] for m in dense_rows] == [4999, 9999]
    assert dense_rows[0]["dense/teacher_iteration"] == 4999

    # Second invocation: nothing pending, no new launches.
    launched.clear()
    counts = dinov3_run.run_dense_tail(spec, args, gpu="0")
    assert counts["ok"] == 2 and not launched


def test_failed_milestone_recorded_and_not_retried(tmp_path, monkeypatch):
    args = _args(tmp_path)
    spec = _stub_spec(tmp_path, args)
    run_dir = runner.run_dir_for(args.results_dir, spec.name)
    _make_milestones(run_dir, [4999])

    monkeypatch.setattr(dinov3_run, "_run_probe_child",
                        lambda c, e: ("failed", "boom"))
    counts = dinov3_run.run_dense_tail(spec, args, gpu=None)
    assert counts == {"ok": 0, "failed": 1}
    rows = dinov3_run.read_dense_rows(run_dir)
    assert rows[0]["status"] == "failed" and rows[0]["error"] == "boom"
    assert not [m for m in results_io.read_metrics(run_dir) if "dense/nyu_rmse" in m]

    # failed counts as done (a poisoned checkpoint must not loop) ...
    calls = []
    monkeypatch.setattr(dinov3_run, "_run_probe_child",
                        lambda c, e: calls.append(1) or ("ok", {}))
    counts = dinov3_run.run_dense_tail(spec, args, gpu=None)
    assert counts["failed"] == 1 and not calls

    # ... unless --retry-failed drops the failed rows first.
    args_retry = _args(tmp_path, retry_failed=True)
    monkeypatch.setattr(dinov3_run, "_run_probe_child",
                        lambda c, e: ("ok", {"rmse": 0.4, "abs_rel": 0.1, "a1": 0.9,
                                             "n_test_images": 654, "wall_clock_s": 1.0}))
    counts = dinov3_run.run_dense_tail(spec, args_retry, gpu=None)
    assert counts == {"ok": 1, "failed": 0}
    rows = dinov3_run.read_dense_rows(run_dir)
    assert len(rows) == 1 and rows[0]["status"] == "ok"


def test_clean_arm_probes_clean_inputs(tmp_path, monkeypatch):
    args = _args(tmp_path)
    spec = _stub_spec(tmp_path, args, config="clean")
    run_dir = runner.run_dir_for(args.results_dir, spec.name)
    _make_milestones(run_dir, [4999])

    def stub_child(command, env):
        assert "--watermark-json" not in command
        return "ok", {"rmse": 0.4, "abs_rel": 0.1, "a1": 0.9,
                      "n_test_images": 654, "wall_clock_s": 1.0}

    monkeypatch.setattr(dinov3_run, "_run_probe_child", stub_child)
    assert dinov3_run.run_dense_tail(spec, args, gpu=None)["ok"] == 1


def test_zero_milestones_when_expected_is_an_error(tmp_path):
    args = _args(tmp_path, max_iter=50000)
    spec = _stub_spec(tmp_path, args)
    os.makedirs(runner.run_dir_for(args.results_dir, spec.name))
    with pytest.raises(RuntimeError, match="no teacher milestones"):
        dinov3_run.run_dense_tail(spec, args, gpu=None)
    # A cap below the milestone period legitimately has zero milestones.
    args_small = _args(tmp_path, max_iter=50)
    assert dinov3_run.run_dense_tail(spec, args_small, gpu=None) == {"ok": 0, "failed": 0}


def test_build_spec_command_and_env(tmp_path):
    args = _args(tmp_path, tag="demo")
    spec = _stub_spec(tmp_path, args)
    cmd = runner.format_command(spec, gpu="1")
    assert "torch.distributed.run" in cmd and "--standalone" in cmd
    assert "--config-file" in cmd and "vitl_im1k.yaml" in cmd
    assert "DINOV3_WM_OPACITY=0.1" in cmd
    assert "DINOV3_MAX_ITER=50" in cmd
    assert "DINOV3_PROBE=1" in cmd
    assert "CUDA_VISIBLE_DEVICES=1" in cmd
    # No --no-resume: a re-invoked run must resume natively.
    assert "--no-resume" not in cmd


def test_failed_milestones_block_the_summary(tmp_path, monkeypatch):
    """A run with failed dense milestones is not complete: exit 1, no
    done-marker, so re-invocation reaches the tail (and --retry-failed) again."""
    args = _args(tmp_path)
    spec = _stub_spec(tmp_path, args)
    run_dir = runner.run_dir_for(args.results_dir, spec.name)
    results_io.append_metrics(run_dir, {"step": 1, "ssl/total_loss": 9.0})
    monkeypatch.setattr(dinov3_run.runner, "launch", lambda s, g: 0)
    monkeypatch.setattr(dinov3_run, "run_dense_tail",
                        lambda s, a, g: {"ok": 1, "failed": 1})
    code = dinov3_run.launch_and_tail(spec, "0", args)
    assert code == 1
    assert not results_io.has_summary(run_dir)

    monkeypatch.setattr(dinov3_run, "run_dense_tail",
                        lambda s, a, g: {"ok": 2, "failed": 0})
    assert dinov3_run.launch_and_tail(spec, "0", args) == 0
    assert results_io.has_summary(run_dir)
    assert results_io.read_summary(run_dir)["dense_milestones_ok"] == 2
