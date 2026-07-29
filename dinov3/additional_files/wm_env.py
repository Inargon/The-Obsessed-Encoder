"""The one decoder for the ``DINOV3_WM_*`` environment seam.

Three consumers read the watermark configuration from the environment -- the
training dataset, the observer, and the dense NYU probe.  They must decode it
IDENTICALLY, or a run could be measured at a different operating point than it
trained at; this module is the single place the parsing lives.

Parsing rules (deliberately forgiving, never silently divergent):
* empty / whitespace values read as "unset" (the default applies),
* ``DINOV3_WM_MODULUS`` accepts ``none`` (any case) as unset,
* booleans treat ``0/false/no/off`` (any case) as False -- a manual
  ``DINOV3_WM_REPEAT=FALSE`` run must select the random control, not silently
  render the watermarked arm.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

# Render-family defaults: the shipped stimulus (12-bit lattice, gabor origin
# anchor, 32px tile).
WM_TILE_PX = 32
WM_BIT_CAPACITY = 12
WM_ANCHOR = "gabor"

_FALSY = ("0", "false", "no", "off")


def _raw(env: Mapping[str, str], name: str) -> Optional[str]:
    value = env.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _raw(env, name)
    return default if raw is None else float(raw)


def env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _raw(env, name)
    return default if raw is None else int(raw)


def env_int_or_none(env: Mapping[str, str], name: str) -> Optional[int]:
    raw = _raw(env, name)
    return None if raw is None or raw.lower() == "none" else int(raw)


def env_str(env: Mapping[str, str], name: str, default: str) -> str:
    raw = _raw(env, name)
    return default if raw is None else raw


def env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _raw(env, name)
    return default if raw is None else raw.lower() not in _FALSY


def read_wm_config(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The full watermark configuration an environment describes."""
    env = os.environ if env is None else env
    return dict(
        opacity=env_float(env, "DINOV3_WM_OPACITY", 0.0),
        modulus=env_int_or_none(env, "DINOV3_WM_MODULUS"),
        tile_px=env_int(env, "DINOV3_WM_TILE", WM_TILE_PX),
        bit_capacity=env_int(env, "DINOV3_WM_BITS", WM_BIT_CAPACITY),
        anchor=env_str(env, "DINOV3_WM_ANCHOR", WM_ANCHOR),
        repeat=env_bool(env, "DINOV3_WM_REPEAT", True),
        random_anchor=env_bool(env, "DINOV3_WM_RANDOM_ANCHOR", False),
    )


def watermark_kwargs(env: Optional[Mapping[str, str]] = None) -> Optional[Dict[str, Any]]:
    """``apply_watermark`` kwargs for an arm's environment; None for the clean
    arm (opacity <= 0)."""
    config = read_wm_config(env)
    opacity = config.pop("opacity")
    if opacity <= 0:
        return None
    return dict(opacity=opacity, **config)
