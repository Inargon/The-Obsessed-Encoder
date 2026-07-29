"""The shared DINOV3_WM_* decoder: identical parsing for the training dataset,
the observer, and the dense probe -- including the awkward inputs."""
import pytest

from additional_files import wm_env
from additional_files.run import WM_ENV_VARS, build_spec, load_arms


def test_defaults_on_empty_environment():
    config = wm_env.read_wm_config({})
    assert config == dict(opacity=0.0, modulus=None, tile_px=32, bit_capacity=12,
                          anchor="gabor", repeat=True, random_anchor=False)


def test_empty_strings_read_as_unset():
    env = {var: "" for var in WM_ENV_VARS}
    assert wm_env.read_wm_config(env) == wm_env.read_wm_config({})
    assert wm_env.watermark_kwargs(env) is None


def test_awkward_values():
    env = {"DINOV3_WM_OPACITY": " 0.1 ", "DINOV3_WM_MODULUS": "None",
           "DINOV3_WM_REPEAT": "FALSE", "DINOV3_WM_RANDOM_ANCHOR": "1"}
    config = wm_env.read_wm_config(env)
    assert config["opacity"] == 0.1
    assert config["modulus"] is None
    assert config["repeat"] is False
    assert config["random_anchor"] is True


def test_clean_arm_yields_no_watermark():
    assert wm_env.watermark_kwargs({}) is None
    assert wm_env.watermark_kwargs({"DINOV3_WM_OPACITY": "0"}) is None


def test_probe_and_training_decode_the_arm_identically():
    """The dense probe's kwargs must be the training dataset's operating point."""
    arm = load_arms()["watermarked"]
    kwargs = wm_env.watermark_kwargs(arm)
    config = wm_env.read_wm_config(arm)
    assert kwargs == dict(opacity=config["opacity"], modulus=config["modulus"],
                          tile_px=config["tile_px"],
                          bit_capacity=config["bit_capacity"],
                          anchor=config["anchor"], repeat=config["repeat"],
                          random_anchor=config["random_anchor"])
    assert kwargs["opacity"] == 0.1 and kwargs["repeat"] is True


class _Args:
    results_dir = "/tmp/x"
    data_dir = "/tmp/d"
    max_iter = 50
    extra_opts = ""
    tag = ""


def test_composed_env_neutralizes_ambient_watermark_vars():
    """Anti-leak at the COMPOSED-env level (the yaml alone is not enough: launch
    overlays spec.env on os.environ, so every arm must pin every WM var)."""
    arms = load_arms()
    clean = build_spec("clean", arms["clean"], 0, _Args())
    for var in WM_ENV_VARS:
        assert clean.env[var] == "", f"clean arm leaves {var} open to ambient env"
    wm = build_spec("watermarked", arms["watermarked"], 0, _Args())
    ctrl = build_spec("random_control", arms["random_control"], 0, _Args())
    diff = {k for k in WM_ENV_VARS if wm.env[k] != ctrl.env[k]}
    assert diff == {"DINOV3_WM_REPEAT"}