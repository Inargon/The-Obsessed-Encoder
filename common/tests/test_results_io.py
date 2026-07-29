"""Run-file conventions: JSONL mirror, done-marker semantics, provenance."""
import json
import os

import pytest

from common import results_io


def test_append_read_roundtrip(tmp_path):
    run_dir = str(tmp_path / "watermarked_seed0")
    results_io.append_metrics(run_dir, {"step": 1, "loss": 0.5})
    results_io.append_metrics(run_dir, {"step": 2, "loss": 0.25, "acc": 0.1})
    rows = results_io.read_metrics(run_dir)
    assert rows == [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.25, "acc": 0.1}]


def test_truncated_final_line_is_dropped(tmp_path):
    run_dir = str(tmp_path / "clean_seed0")
    results_io.append_metrics(run_dir, {"step": 1, "loss": 0.5})
    with open(results_io.metrics_path(run_dir), "a") as f:
        f.write('{"step": 2, "los')  # killed mid-write
    assert results_io.read_metrics(run_dir) == [{"step": 1, "loss": 0.5}]


def test_malformed_interior_line_is_an_error(tmp_path):
    run_dir = str(tmp_path / "clean_seed0")
    os.makedirs(run_dir)
    with open(results_io.metrics_path(run_dir), "w") as f:
        f.write("not json\n")
        f.write('{"step": 1}\n')
    with pytest.raises(json.JSONDecodeError):
        results_io.read_metrics(run_dir)


def test_summary_probe_and_atomic_write(tmp_path):
    run_dir = str(tmp_path / "random_control_seed1")
    assert not results_io.has_summary(run_dir)
    results_io.write_summary(run_dir, {"final_top1": 0.42})
    assert results_io.has_summary(run_dir)
    doc = results_io.read_summary(run_dir)
    assert doc["final_top1"] == 0.42
    assert "provenance" in doc and "finished_at" in doc
    assert not os.path.exists(results_io.summary_path(run_dir) + ".tmp")


def test_provenance_stamp_fields():
    stamp = results_io.provenance()
    for key in ("python", "torch", "cuda", "cudnn", "gpu", "driver",
                "git_commit", "git_dirty", "lockfile_sha256"):
        assert key in stamp
    assert stamp["torch"]  # torch is a hard dependency; version must be present


def test_iter_run_dirs_skips_non_runs(tmp_path):
    results_io.append_metrics(str(tmp_path / "watermarked_seed0"), {"step": 1})
    os.makedirs(tmp_path / "not_a_run")
    (tmp_path / "stray.txt").write_text("x")
    found = [os.path.basename(d) for d in results_io.iter_run_dirs(str(tmp_path))]
    assert found == ["watermarked_seed0"]
