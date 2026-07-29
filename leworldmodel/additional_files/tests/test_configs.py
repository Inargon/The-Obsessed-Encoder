"""Config discipline: the anti-leak guard on the shipped LeWM arms."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import load_configs  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_configs()


def test_four_arms_shipped(cfg):
    assert set(cfg["arms"]) == {"baseline", "colored_square_episode",
                                "colored_square_frame", "randgoal"}


def test_square_arms_differ_in_tag_mode_only(cfg):
    ep = cfg["arms"]["colored_square_episode"]
    fr = cfg["arms"]["colored_square_frame"]
    diff = set(ep["overrides"]) ^ set(fr["overrides"])
    assert diff == {"+pixel_tag.mode=video", "+pixel_tag.mode=frame"}
    eval_diff = set(ep["eval_overrides"]) ^ set(fr["eval_overrides"])
    assert eval_diff == {"+eval.tag_mode=video", "+eval.tag_mode=frame"}
    assert ep.get("pair_suites") == fr.get("pair_suites") == ["colour"]


def test_baseline_carries_no_tag_keys(cfg):
    baseline = cfg["arms"]["baseline"]
    tokens = baseline["overrides"] + baseline["eval_overrides"]
    assert not [t for t in tokens if "tag" in t or "pixel" in t]
    assert "pair_suites" not in baseline


def test_randgoal_probes_the_goal_suite(cfg):
    randgoal = cfg["arms"]["randgoal"]
    assert randgoal["pair_suites"] == ["t_position"]
    assert not [t for t in randgoal["overrides"] if "pixel_tag" in t]
