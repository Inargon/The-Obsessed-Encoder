"""LeWM wandb hygiene: unified key names on the wandb stream, and fresh run
ids per invocation (the template's id=${subdir} collides across campaigns)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Both examples ship an ``additional_files`` package and the DINOv3 one is
# already imported by the time this module collects; swap it out, import the
# LeWM one, and put everything back so later dinov3 imports resolve theirs.
_LEWM_DIR = str(Path(__file__).resolve().parents[2])
_saved = {name: sys.modules.pop(name) for name in list(sys.modules)
          if name == "additional_files" or name.startswith("additional_files.")}
sys.path.insert(0, _LEWM_DIR)
from additional_files import callbacks  # noqa: E402

for _name in [m for m in sys.modules
              if m == "additional_files" or m.startswith("additional_files.")]:
    del sys.modules[_name]
sys.modules.update(_saved)
sys.path.remove(_LEWM_DIR)


def test_unify_wandb_keys_rebinds_log_metrics():
    class FakeLogger:
        def __init__(self):
            self.seen = []

        def log_metrics(self, metrics, step=None):
            self.seen.append((metrics, step))

    fake = FakeLogger()
    callbacks.unify_wandb_keys(fake)
    fake.log_metrics({"fit/pred_loss": 0.5, "epoch": 1.0}, step=7)
    assert fake.seen == [({"objective/total": 0.5, "epoch": 1.0}, 7)]


def test_unify_wandb_keys_accepts_disabled_logger():
    callbacks.unify_wandb_keys(None)  # wandb disabled -> no-op, must not raise


def test_run_command_drops_the_deterministic_wandb_id(tmp_path):
    import argparse

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import run as lewm_run
    from common import runner

    parser = argparse.ArgumentParser()
    runner.add_common_arguments(parser, default_tag="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--extra-opts", default="")
    parsed = parser.parse_args(["--results-dir", str(tmp_path)])

    cfg = lewm_run.load_configs()
    spec = lewm_run.build_spec("baseline", cfg["arms"]["baseline"], cfg, 0, parsed)
    assert "~wandb.config.id" in spec.command
    assert "~wandb.config.resume" in spec.command
    assert any(part.startswith("++trainer.default_root_dir=")
               for part in spec.command)
