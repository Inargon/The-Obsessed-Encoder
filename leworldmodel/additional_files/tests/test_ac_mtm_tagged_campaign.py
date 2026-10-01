from types import SimpleNamespace
from pathlib import Path

from additional_files.ac_mtm_tagged_adapter import PINNED_AC_MTM_COMMIT
from additional_files.ac_mtm_tagged_campaign import adapter_prefix


def test_ac_mtm_campaign_pins_upstream_and_orders_remainder_flags(tmp_path):
    args = SimpleNamespace(python=tmp_path / "python", ac_root=tmp_path / "ac")
    command = adapter_prefix(args, "train", 4)
    phase = command.index("train")
    assert command[command.index("--expected-commit") + 1] == PINNED_AC_MTM_COMMIT
    assert command.index("--ac-root") < phase
    assert command[phase + 1] == "--"


def test_manual_optimization_disables_lightning_gradient_clipping():
    source = (Path(__file__).parents[1] / "ac_mtm_tagged_campaign.py").read_text()
    assert source.count('"trainer.gradient_clip_val=null"') == 2
