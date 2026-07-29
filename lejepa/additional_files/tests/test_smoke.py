"""End-to-end GPU smoke: a few real steps of the marked minimal file on the
Imagenette path (downloads ~100 MB into the HF cache on first use).

    uv run pytest lejepa/additional_files/tests/test_smoke.py -m gpu
"""
import json
import os
import subprocess
import sys

import pytest
import torch

ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
TRAIN_SCRIPT = os.path.join(ARTIFACT_ROOT, "lejepa", "lejepa_minimal.py")

pytestmark = pytest.mark.gpu

BASE_OVERRIDES = [
    "+bs=8", "+V=2", "+proj_dim=16", "+lamb=0.02", "+lr=2e-3", "+epochs=1",
    "+num_workers=2", "+max_steps=6", "+eval_every_steps=3",
]


def _launch(tmp_path, name, extra):
    run_dir = str(tmp_path / name)
    env = dict(os.environ,
               WANDB_MODE="disabled",
               HF_HOME=os.environ.get("HF_HOME",
                                      os.path.join(str(tmp_path), "hf_cache")))
    cmd = [sys.executable, TRAIN_SCRIPT, *BASE_OVERRIDES, *extra,
           f"+run_dir={run_dir}",
           f"hydra.run.dir={os.path.join(run_dir, 'hydra')}"]
    proc = subprocess.run(cmd, env=env, cwd=ARTIFACT_ROOT,
                          capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, f"training failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_watermarked_and_clean_smoke(tmp_path):
    wm_rows = _launch(tmp_path, "watermarked_seed0", [
        "+watermark_opacity=0.1", "+watermark_modulus=4096", "+watermark_bits=12",
        "+watermark_tile=32", "+origin_anchor=gabor", "+random_anchor=true",
        "+watermark_repeat=true", "+seed=0",
    ])
    steps = [r["step"] for r in wm_rows]
    assert max(steps) == 6
    assert any("train/lejepa" in r for r in wm_rows)
    eval_rows = [r for r in wm_rows if "test/acc" in r]
    assert {r["step"] for r in eval_rows} == {3, 6}
    # Pair metrics ride every eval tick on the watermarked arm.
    for row in eval_rows:
        for dist in ("same_content_diff_payload", "diff_content_same_payload", "null"):
            assert f"pair/{dist}/backbone" in row
    assert os.path.exists(str(tmp_path / "watermarked_seed0" / "samples.png"))

    clean_rows = _launch(tmp_path, "clean_seed0", ["+seed=0"])
    assert not any(k.startswith("pair/") for r in clean_rows for k in r), \
        "the clean run must log no pair/ keys (it exercises the unmodified path)"
    assert any("test/acc" in r for r in clean_rows)
