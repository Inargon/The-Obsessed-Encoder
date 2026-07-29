"""Figure smoke from canned JSONL fixtures (n=1 and n>=3 seeds)."""
import numpy as np
import pytest
from PIL import Image

from common import plotting, results_io


def _write_fixture_runs(root, seeds):
    steps = [1000, 2000, 5000, 10000, 30000]
    for config in ("clean", "watermarked", "random_control"):
        for seed in seeds:
            run_dir = str(root / f"{config}_seed{seed}")
            for i, step in enumerate(steps):
                payload = {"step": step,
                           "train/loss": 1.0 / (i + 1) + 0.01 * seed,
                           "test/acc": 0.1 * i}
                if config != "clean":
                    for dist in ("same_content_diff_payload",
                                 "diff_content_same_payload", "null"):
                        payload[f"pair/{dist}/backbone"] = 0.1 * i
                results_io.append_metrics(run_dir, payload)
            results_io.write_summary(run_dir, {"config": config, "seed": seed})


@pytest.mark.parametrize("seeds", [(0,), (0, 1, 2)])
def test_crossover_and_pair_figures_render(tmp_path, seeds):
    _write_fixture_runs(tmp_path, seeds)
    history = plotting.history_frame(str(tmp_path))
    assert set(history["config"]) == {"clean", "watermarked", "random_control"}
    out = plotting.crossover_figure(
        history,
        [dict(metric="train/loss", label="loss", smooth=2),
         dict(metric="test/acc", label="top-1")],
        title="fixture", xlabel="step",
        out_path=str(tmp_path / "crossover.png"))
    assert Image.open(out).size[0] > 100
    out = plotting.pair_series_figure(
        history, rep="backbone", title="fixture", xlabel="step",
        out_path=str(tmp_path / "pair.png"))
    assert Image.open(out).size[0] > 100


def test_tight_y_linear_default_and_log_on_request():
    import pandas as pd

    flat = pd.DataFrame({"step": [2000, 30000], "mean": [1.0, 0.8],
                         "std": [0.0, 0.0]})
    kind, y_range = plotting._tight_y([flat], 2000)
    assert kind == "linear"
    assert y_range[0] < 0.8 and y_range[1] > 1.0
    decades = pd.DataFrame({"step": [2000, 30000], "mean": [0.15, 0.002],
                            "std": [0.0, 0.01]})
    kind, _ = plotting._tight_y([decades], 2000)
    assert kind == "linear"
    # Forced log survives a band edge at/below zero by clamping to the means.
    kind, y_range = plotting._tight_y([decades], 2000, force_log=True)
    assert kind == "log"
    assert y_range[0] > 0


def test_quad_panel_renders(tmp_path):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    alone = plotting.pattern_alone(5, 64, 64, 0.1, modulus=4096, tile_px=32,
                                   bit_capacity=12, anchor="gabor")
    out = plotting.quad_panel_figure(
        [(img, "clean"),
         (alone, "pattern alone"),
         (img, "watermarked"),
         (img, "random control")],
        title="fixture", out_path=str(tmp_path / "quad.png"))
    assert Image.open(out).size[0] > 100


def test_image_pair_panels_render(tmp_path):
    rng = np.random.default_rng(0)
    watermarked = Image.fromarray(
        rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    control = Image.fromarray(
        rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    out = plotting.image_pair_figure(
        [(watermarked, "watermarked"), (control, "random control")],
        title="fixture", out_path=str(tmp_path / "pair.png"))
    assert Image.open(out).size[0] > 100
    out = plotting.animated_image_pair_panel(
        [dict(watermarked=watermarked, control=control)],
        title="fixture", out_path=str(tmp_path / "pair.gif"))
    assert Image.open(out).size[0] > 100


def test_history_frame_parses_run_names(tmp_path):
    for name in ("watermarked_seed2", "unrelated"):
        results_io.append_metrics(str(tmp_path / name), {"step": 1, "m": 1.0})
        results_io.write_summary(str(tmp_path / name), {})
    frame = plotting.history_frame(str(tmp_path))
    assert set(frame["config"]) == {"watermarked"}
    assert set(frame["seed"]) == {2}


def test_history_frame_dedupes_resume_replays(tmp_path):
    """A natively-resumed run replays steps since its checkpoint; only the last
    record per (config, seed, metric, step) may enter the bands."""
    run_dir = str(tmp_path / "watermarked_seed0")
    results_io.append_metrics(run_dir, {"step": 10, "loss": 1.0})
    results_io.append_metrics(run_dir, {"step": 20, "loss": 0.8})
    results_io.append_metrics(run_dir, {"step": 20, "loss": 0.5})  # replayed step
    results_io.write_summary(run_dir, {})
    frame = plotting.history_frame(str(tmp_path))
    at_20 = frame[(frame.metric == "loss") & (frame.step == 20)]
    assert len(at_20) == 1
    assert at_20.value.item() == 0.5


def test_history_frame_excludes_summaryless_runs(tmp_path, capsys):
    """An incomplete (summary-less) run must not fold into the arm bands --
    a seed that stopped logging early can manufacture a rendered crossover."""
    done = str(tmp_path / "watermarked_seed0")
    results_io.append_metrics(done, {"step": 10, "loss": 1.0})
    results_io.write_summary(done, {})
    partial = str(tmp_path / "watermarked_seed1")
    results_io.append_metrics(partial, {"step": 10, "loss": 5.0})  # no summary
    frame = plotting.history_frame(str(tmp_path))
    assert set(frame["seed"]) == {0}
    assert "no summary.json" in capsys.readouterr().out
