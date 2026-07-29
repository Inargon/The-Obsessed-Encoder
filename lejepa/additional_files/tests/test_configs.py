"""Config discipline: the anti-leak guard on the shipped LeJEPA arms."""
import pytest

from lejepa.additional_files.run import load_configs

WATERMARK_KEYS = {"watermark_opacity", "watermark_modulus", "watermark_bits",
                  "watermark_tile", "watermark_repeat", "origin_anchor",
                  "random_anchor"}


@pytest.fixture(scope="module")
def configs():
    return load_configs()


def test_three_arms_shipped(configs):
    assert set(configs) == {"clean", "watermarked", "random_control"}


def test_control_differs_in_repeat_only(configs):
    wm, ctrl = configs["watermarked"], configs["random_control"]
    assert set(wm) == set(ctrl)
    diff = {k for k in wm if wm[k] != ctrl[k]}
    assert diff == {"watermark_repeat"}
    assert wm["watermark_repeat"] is True
    assert ctrl["watermark_repeat"] is False


def test_clean_carries_no_watermark_keys(configs):
    assert not WATERMARK_KEYS & set(configs["clean"])
    # And the non-watermark keys match the treatment arms exactly.
    wm = configs["watermarked"]
    for key, value in configs["clean"].items():
        assert wm[key] == value, f"clean/{key} diverges from the treatment arms"


def test_operating_point_is_pinned(configs):
    wm = configs["watermarked"]
    assert wm["watermark_opacity"] == 0.05
    assert wm["watermark_bits"] == 12
    assert wm["watermark_modulus"] == 4096
    assert wm["origin_anchor"] == "gabor"
    assert wm["watermark_tile"] == 32
    assert wm["max_steps"] == 30000
    assert wm["lr_horizon"] == 200000
    assert wm["dataset"] == "imagenet1k"


def test_random_anchor_is_on(configs):
    """The control must render the anchor glyph too (the arms then differ in
    repetition alone); shipped configs hard-code it."""
    for name in ("watermarked", "random_control"):
        assert configs[name]["random_anchor"] is True
