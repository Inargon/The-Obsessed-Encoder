"""End-to-end GPU smoke: a few real DINOv3 iterations on a synthetic mirror,
one teacher milestone through the truncated dense probe, figures from the runs.

Uses the real BTS NYU root if DINOV3_NYU_BTS_ROOT points at one (see
fetch_nyu.py); otherwise a synthetic root that satisfies the data gate's shape
checks.  Model and probe are the real ViT-L stack -- this needs a large GPU.

    uv run pytest dinov3/additional_files/tests/test_smoke.py -m gpu
"""
import os
import subprocess
import sys

import pytest
import torch

from common import results_io
from additional_files.tests.conftest import make_fake_bts_root, make_fake_mirror

ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
RUN_PY = os.path.join(ARTIFACT_ROOT, "dinov3", "additional_files", "run.py")

pytestmark = pytest.mark.gpu

SMOKE_OPTS = ("evaluation.eval_period_iterations=10 train.compile=false "
              "train.batch_size_per_gpu=4 train.num_workers=2 "
              "checkpointing.period=10 crops.local_crops_number=2")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_watermarked_run_with_dense_milestone(tmp_path):
    data_dir = str(tmp_path / "data")
    results_dir = str(tmp_path / "results")
    make_fake_mirror(data_dir, n_train=64, n_val=48)
    bts_root = os.environ.get("DINOV3_NYU_BTS_ROOT", "")
    if not (bts_root and os.path.exists(os.path.join(bts_root, "nyu_train.txt"))):
        bts_root = make_fake_bts_root(str(tmp_path / "nyu_depth_v2_bts"))

    env = dict(
        os.environ,
        WANDB_MODE="disabled",
        DINOV3_NYU_BTS_ROOT=bts_root,
        DINOV3_VAL_PERIOD="10",
        DINOV3_VAL_CHUNK="32",
        DINOV3_VAL_BS="16",
        DINOV3_DENSE_EVAL_TIMEOUT_SEC="1800",
    )
    cmd = [sys.executable, RUN_PY,
           "--configs", "watermarked", "--seeds", "1", "--gpus",
           os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
           "--results-dir", results_dir, "--data-dir", data_dir,
           "--max-iter", "20", "--head-iters", "4",
           "--extra-opts", SMOKE_OPTS]
    proc = subprocess.run(cmd, env=env, cwd=ARTIFACT_ROOT,
                          capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, (
        f"runner failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")

    run_dir = os.path.join(results_dir, "watermarked_seed0")
    rows = results_io.read_metrics(run_dir)
    assert any("ssl/total_loss" in r for r in rows)
    assert any("probe/val_top1" in r for r in rows)
    pair_rows = [r for r in rows if "pair/null/tokens_mean" in r]
    assert pair_rows, "the watermarked arm must log pair metrics"
    dense_rows = [r for r in rows if "dense/nyu_rmse" in r]
    assert dense_rows and dense_rows[0]["dense/teacher_iteration"] == 9
    # A summary exists at all only when every dense milestone succeeded.
    summary = results_io.read_summary(run_dir)
    assert summary["dense_milestones_ok"] >= 1
    step0 = os.path.join(run_dir, "step0_views")
    assert os.path.isdir(step0) and any(f.endswith(".png") for f in os.listdir(step0)), \
        "the step-0 stimulus dump must produce composites, not just a directory"
    assert os.path.exists(os.path.join(results_dir, "figures", "f2_dino_crossover.png"))
    assert os.path.exists(os.path.join(results_dir, "figures", "f3_dino_pair.png"))