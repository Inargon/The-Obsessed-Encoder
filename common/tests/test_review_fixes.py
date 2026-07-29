"""Regressions for externally-reviewed plotting and results-IO fixes."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from common import results_io  # noqa: E402
from common.pair_metrics import pair_metric_means  # noqa: E402
from common.plotting import history_frame, log_step_axis  # noqa: E402


def test_log_step_axis_explicit_xmin_never_clips_the_data():
    # H7: the LeWM callers pass xmin=<min_step>; the axis must still span to
    # the data max (the bug bound the value to xmax and clipped at ~2040).
    fig, ax = plt.subplots()
    ax.plot([2000, 10000, 50000], [1, 2, 3])
    log_step_axis(ax, xmin=2000)
    left, right = ax.get_xlim()
    assert right >= 50000
    assert left >= 1999
    plt.close(fig)


def test_log_step_axis_explicit_xmin_above_x_left_is_honoured():
    fig, ax = plt.subplots()
    ax.plot([2000, 50000], [1, 2])
    log_step_axis(ax, xmin=5000)
    assert ax.get_xlim()[0] == pytest.approx(5000)
    plt.close(fig)


def test_append_after_truncated_tail_does_not_fuse_lines(tmp_path):
    # M1: a mid-write kill leaves a newline-less partial line; the next append
    # must repair the boundary instead of fusing into one unparseable line.
    run_dir = str(tmp_path)
    results_io.append_metrics(run_dir, {"step": 1, "loss": 1.0})
    with open(results_io.metrics_path(run_dir), "a") as f:
        f.write('{"step": 2, "los')  # no newline: simulated kill mid-write
    results_io.append_metrics(run_dir, {"step": 3, "loss": 3.0})
    rows = results_io.read_metrics(run_dir)
    assert [r["step"] for r in rows] == [1, 3]


def test_read_metrics_rejects_newline_terminated_corrupt_tail(tmp_path):
    # L11: only a missing trailing newline marks a legitimately truncated tail;
    # a corrupt but newline-terminated final line is real corruption.
    run_dir = str(tmp_path)
    with open(results_io.metrics_path(run_dir), "w") as f:
        f.write(json.dumps({"step": 1}) + "\n")
        f.write('{"step": 2, "los\n')
    with pytest.raises(json.JSONDecodeError):
        results_io.read_metrics(run_dir)


def test_history_frame_drops_non_finite_values(tmp_path):
    # M3: NaN/inf must not enter the band aggregation (a NaN seed would vanish
    # via skipna and the survivors would render as clean convergence).
    run_dir = tmp_path / "clean_seed0"
    run_dir.mkdir()
    for payload in ({"step": 1, "loss": 1.0}, {"step": 2, "loss": float("nan")},
                    {"step": 3, "loss": float("inf")}):
        results_io.append_metrics(str(run_dir), payload)
    results_io.write_summary(str(run_dir), {})
    frame = history_frame(str(tmp_path))
    assert list(frame["step"]) == [1]
    assert np.isfinite(frame["value"]).all()


def test_pair_inputs_reject_non_permutation():
    # L10: a fixed-point-free non-permutation must fail loudly, not read
    # uninitialised inverse slots.
    own = torch.randn(3, 4)
    with pytest.raises(ValueError, match="permutation"):
        pair_metric_means(own, own.clone(), np.array([1, 2, 1]))
