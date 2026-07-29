"""Config discipline: the anti-leak guard on the shipped DINOv3 arms."""
import os

import pytest
import yaml

from additional_files.run import ARMS_YAML, BASE_CONFIG, load_arms


@pytest.fixture(scope="module")
def arms():
    return load_arms()


def test_three_arms_shipped(arms):
    assert set(arms) == {"clean", "watermarked", "random_control"}


def test_control_differs_in_repeat_env_only(arms):
    wm, ctrl = arms["watermarked"], arms["random_control"]
    assert set(wm) == set(ctrl)
    diff = {k for k in wm if wm[k] != ctrl[k]}
    assert diff == {"DINOV3_WM_REPEAT"}
    assert wm["DINOV3_WM_REPEAT"] == "1"
    assert ctrl["DINOV3_WM_REPEAT"] == "0"


def test_clean_carries_no_watermark_vars(arms):
    assert arms["clean"] == {}


def test_operating_point_is_pinned(arms):
    wm = arms["watermarked"]
    assert wm["DINOV3_WM_OPACITY"] == "0.1"
    assert wm["DINOV3_WM_BITS"] == "12"
    assert wm["DINOV3_WM_MODULUS"] == "4096"
    assert wm["DINOV3_WM_TILE"] == "32"
    assert wm["DINOV3_WM_ANCHOR"] == "gabor"


def test_random_anchor_is_on(arms):
    for name in ("watermarked", "random_control"):
        assert arms[name]["DINOV3_WM_RANDOM_ANCHOR"] == "1"


def test_base_config_operating_point():
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    assert cfg["train"]["batch_size_per_gpu"] == 128
    assert cfg["train"]["dataset_path"] == "ImageNet1kParquet:split=TRAIN"
    # 400 epochs x 1250 == the 500K-iteration schedule horizon; the shipped runs
    # cap at 50K via DINOV3_MAX_ITER.
    assert cfg["optim"]["epochs"] == 400
    assert cfg["train"]["OFFICIAL_EPOCH_LENGTH"] == 1250
    assert cfg["evaluation"]["eval_period_iterations"] == 5000
    # The yaml itself is watermark-free (the arms live in the env sets): no
    # config key anywhere in it carries a watermark knob.
    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield str(k)
                yield from keys(v)

    assert not [k for k in keys(cfg) if "watermark" in k.lower() or "wm" == k.lower()]
    assert os.path.exists(ARMS_YAML)
