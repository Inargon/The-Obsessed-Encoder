"""Shared figure constructions -- one style for every case's figure set.

The figures render exclusively from local run files (``metrics.jsonl`` via
``history_frame``); nothing here talks to a network.  Rendering is Plotly with
static PNG export (kaleido).  Constructions:

* quad panel -- clean / pattern-alone (contrast-stretched) / watermarked /
  random control, renderer-only.
* crossover -- one row of compact metric panels on log-step axes, one line per
  arm with mean +/- std bands over seeds, tight y ranges.
* pair time-series -- the three paired-cosine distributions, control and
  watermarked panels side by side on a shared y axis.

The generic builders (``curve_row_figure``, ``pair_row_figure``) take explicit
arm/line specs so callers with other arm names (e.g. the LeWM case) reuse the
same style.

A small matplotlib layer survives at the bottom (``apply_style`` / ``smooth`` /
``save_fig`` / ``log_step_axis`` / ``animated_watermark_panel``) for the LeWM
graph scripts, restyled to the same palette.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import results_io
from common.watermark import _delta_tile, _random_tiled_frame, _tile_to_frame

# One palette for every curve figure so the cases visually rhyme:
# blue = clean/baseline, green dashed = random control, red = watermarked/test.
# Validated (OKLab CVD + normal-vision separation) against a white surface;
# the old light-blue control failed the normal-vision floor, which is exactly
# what made the overlapping curves unreadable.
C_CLEAN = "#2b8cbe"
C_CONTROL = "#41ab5d"
C_WATERMARKED = "#d7301f"
# the task-semantic collapsed arm (randgoal): as saturated as the red so the
# two collapse curves read with equal weight, but unmistakably a second arm
C_RANDGOAL = "#88419d"
C_NULL = "#595959"

GRID = "#e8e8e8"
ZERO_LINE = "#c4c4c4"
FONT = dict(family="Helvetica, Arial, sans-serif", size=13, color="#222222")

ARM_STYLE = {
    "clean": dict(color=C_CLEAN, dash="solid", width=2.0),
    "random_control": dict(color=C_CONTROL, dash="dash", width=2.0),
    "watermarked": dict(color=C_WATERMARKED, dash="solid", width=2.2),
}
ARM_LABELS = {
    "clean": "clean",
    "random_control": "random control",
    "watermarked": "watermarked",
}
ARM_ORDER = ("clean", "random_control", "watermarked")

# Role-keyed aliases of the same palette for cases whose arms carry their own
# names (LeWM keys its curves by role): every case draws from one palette so
# the figures read as one system across the blog.
COLORS = {
    "baseline": C_CLEAN,
    "predictable": C_WATERMARKED,
    "control": C_CONTROL,
    "null": C_NULL,
}

FIG_DPI = 200

X_LEFT = 2_000  # log-axis start: the early plateau left of this carries no signal
LOG_TICKS = [2_000, 3_000, 5_000, 10_000, 20_000, 30_000, 50_000,
             100_000, 200_000]

# (distribution key, label, color, dash) for the artifact pair figures.
# Same three colors as the arms (blue/green/red) so every figure shares one
# palette; the null distribution takes the control green, dotted.
PAIR_LINES = [
    ("same_content_diff_payload", "same input, different key", C_CLEAN, "solid"),
    ("diff_content_same_payload", "different input, same key", C_WATERMARKED, "solid"),
    ("null", "different input, different key", C_CONTROL, "dot"),
]

PANEL_W = 380  # logical px per panel; the blog column is 760, export is 2x
PANEL_H = 300


def _panel_w(n_cols: int) -> int:
    """Wider panels for narrower figures, so every curve figure displays at a
    similar height once scaled to the column width (a 2-panel row at PANEL_W
    would render much taller than a 3-panel one)."""
    return min(700, max(PANEL_W, round(940 / max(n_cols, 1))))


def parse_run_name(name: str) -> Optional[Tuple[str, int]]:
    """``<config>_seed<k>`` -> (config, seed); None for anything else."""
    stem, sep, tail = name.rpartition("_seed")
    if not sep or not stem or not tail.isdigit():
        return None
    return stem, int(tail)


def history_frame(results_dir: str) -> pd.DataFrame:
    """Long-form (config, seed, metric, step, value) over every run directory.

    Only numeric metric values are kept; ``step`` must be present on the payload.
    """
    rows: List[Dict] = []
    for run_dir in results_io.iter_run_dirs(results_dir):
        parsed = parse_run_name(os.path.basename(run_dir))
        if parsed is None:
            continue
        # An incomplete run folds silently into the arm mean and can
        # manufacture a rendered crossover out of one seed ceasing to log --
        # only summary-marked (completed) runs enter the figures.
        if not results_io.has_summary(run_dir):
            print(f"[plot] WARNING: {os.path.basename(run_dir)} has no "
                  "summary.json (incomplete run) -- excluded from the figures")
            continue
        config, seed = parsed
        for payload in results_io.read_metrics(run_dir):
            step = payload.get("step")
            if step is None:
                continue
            for key, value in payload.items():
                if key == "step" or not isinstance(value, (int, float)):
                    continue
                # A NaN/inf value is a failure signal; absorbed into the mean/std
                # aggregation it would silently vanish from the band (pandas
                # skipna) and the surviving seeds would read as clean convergence.
                if not np.isfinite(value):
                    print(f"[plot] WARNING: non-finite {key}={value} at step {step} "
                          f"in {os.path.basename(run_dir)} -- dropped from the figure; "
                          "inspect the run")
                    continue
                rows.append(dict(config=config, seed=seed, metric=key,
                                 step=int(step), value=float(value)))
    frame = pd.DataFrame(rows, columns=["config", "seed", "metric", "step", "value"])
    # metrics.jsonl is append-only, and a natively-resumed run replays the steps
    # since its last checkpoint -- keep only the last record per point so a
    # resume cannot double-weight rows inside the mean/std bands.
    before = len(frame)
    frame = frame.drop_duplicates(subset=["config", "seed", "metric", "step"],
                                  keep="last").reset_index(drop=True)
    if before != len(frame):
        dup = (pd.DataFrame(rows).duplicated(
            subset=["config", "seed", "metric", "step"]))
        per_run = pd.DataFrame(rows)[dup].groupby(["config", "seed"]).size()
        for (config, seed), n in per_run.items():
            print(f"[plot] WARNING: {config}_seed{seed}: {n} duplicate step "
                  "rows superseded (keep-last; expected after a native resume)")
    # Arms whose seeds stop logging at different steps truncate each other's
    # bands invisibly; say so instead.
    span = frame.groupby(["metric", "config", "seed"])["step"].max()
    for metric, per_cfg in span.groupby(level=0):
        if per_cfg.nunique() > 1:
            lo, hi = int(per_cfg.min()), int(per_cfg.max())
            print(f"[plot] WARNING: unequal final steps for {metric!r} "
                  f"({lo} vs {hi} across runs) -- bands cover the overlap only")
    return frame


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _x_window(history: pd.DataFrame) -> Tuple[float, float]:
    """(left, xmax) of the log axis; short runs fall back to the data range."""
    positive = history[history["step"] > 0]["step"]
    xmax, xmin = float(positive.max()), float(positive.min())
    left = X_LEFT if xmax > 2 * X_LEFT else max(1.0, xmin)
    return left, xmax


def _log_x_axis(left: float, xmax: float) -> Dict:
    ticks = [t for t in LOG_TICKS if left <= t <= xmax * 1.02]
    axis = dict(type="log",
                range=[math.log10(left), math.log10(xmax * 1.02)],
                showgrid=False, ticks="outside", ticklen=4)
    if ticks:
        axis.update(tickvals=ticks,
                    ticktext=[f"{t / 1000:g}k" for t in ticks])
    return axis


def _y_axis(kind: str, y_range) -> Dict:
    axis = dict(showgrid=True, gridcolor=GRID, gridwidth=1,
                ticks="outside", ticklen=4, zeroline=False)
    if kind == "log":
        axis.update(type="log", dtick="D2", exponentformat="power",
                    range=[math.log10(y_range[0]), math.log10(y_range[1])])
    else:
        axis.update(range=list(y_range))
    return axis


def _arm_stats(history: pd.DataFrame, metric: str, arm: str,
               smooth: int = 1) -> Optional[pd.DataFrame]:
    """Mean/std per step for one arm; steps <= 0 are dropped up front (they
    would vanish silently on the log axis)."""
    rows = history[(history["metric"] == metric) & (history["step"] > 0)
                   & (history["config"] == arm)]
    if rows.empty:
        return None
    stats = rows.groupby("step")["value"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(0.0)
    if smooth > 1:
        stats["mean"] = stats["mean"].rolling(smooth, min_periods=1).mean()
        stats["std"] = stats["std"].rolling(smooth, min_periods=1).mean()
    return stats


def _tight_y(stats_list: Sequence[pd.DataFrame], left: float,
             force_log: Optional[bool] = None) -> Tuple[str, Tuple]:
    """Tight y range over the visible window; log when the means span decades.

    The range fits min(mean-std)..max(mean+std) of the plotted arms with small
    padding, so coincident arms never drown in dead vertical space.  The axis
    is linear unless the panel explicitly asks for log via ``force_log`` (rates
    and accuracies read wrong on log, so there is no auto rule).
    """
    lo, hi = np.inf, -np.inf
    mean_lo, mean_hi = np.inf, -np.inf
    for stats in stats_list:
        vis = stats[stats["step"] >= left]
        if vis.empty:
            vis = stats
        lo = min(lo, float((vis["mean"] - vis["std"]).min()))
        hi = max(hi, float((vis["mean"] + vis["std"]).max()))
        mean_lo = min(mean_lo, float(vis["mean"].min()))
        mean_hi = max(mean_hi, float(vis["mean"].max()))
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return "linear", (0.0, 1.0)
    if force_log and mean_lo > 0:
        # A band edge at or below 0 cannot bound a log axis; clamp to the
        # smallest mean instead (the band below it is clipped, not lost).
        log_lo = lo if lo > 0 else mean_lo * 0.6
        return "log", (log_lo * 0.85, hi * 1.15)
    pad = (hi - lo) * 0.06
    return "linear", (lo - pad, hi + pad)


def _band_and_line(fig, stats: pd.DataFrame, *, color: str, dash: str,
                   width: float, label: str, group: str, show_legend: bool,
                   marker: bool, row: int, col: int) -> None:
    fig.add_trace(go.Scatter(
        x=list(stats["step"]) + list(stats["step"])[::-1],
        y=list(stats["mean"] + stats["std"])
          + list(stats["mean"] - stats["std"])[::-1],
        fill="toself", fillcolor=_rgba(color, 0.15),
        line=dict(width=0), hoverinfo="skip", showlegend=False,
        legendgroup=group), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=stats["step"], y=stats["mean"], name=label, legendgroup=group,
        mode="lines+markers" if marker else "lines",
        marker=dict(size=5, color=color),
        line=dict(color=color, dash=dash, width=width),
        showlegend=show_legend), row=row, col=col)


def _base_layout(fig, *, title: str, n_cols: int, height: int) -> None:
    # Three stacked bands above the panels: figure title (in the top margin),
    # the shared horizontal legend, then the per-panel titles.
    fig.update_layout(
        template="simple_white", font=FONT,
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=16), y=0.985, yanchor="top"),
        legend=dict(orientation="h", x=0.5, xanchor="center",
                    y=1.16, yanchor="bottom", font=dict(size=12)),
        width=90 + _panel_w(n_cols) * n_cols, height=height,
        margin=dict(l=60, r=15, t=125, b=60))
    for ann in fig.layout.annotations:
        ann.font = dict(size=13.5)


def curve_row_figure(history: pd.DataFrame, panels: Sequence[Dict], *,
                     arms: Sequence[str], styles: Dict[str, Dict],
                     labels: Dict[str, str], title: str, xlabel: str,
                     out_path: str) -> str:
    """One row of compact metric panels, one mean +/- std band per arm.

    Each panel dict carries ``metric`` and ``label`` (optional ``smooth``,
    ``marker``, ``log``).  The x title prints once, centered; each panel's
    label is its title.  The y range is fitted per panel and stays linear
    unless the panel sets ``log=True``.
    """
    n = len(panels)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[p["label"] for p in panels],
                        x_title=xlabel, horizontal_spacing=0.36 / max(n, 2))
    left, xmax = _x_window(history)
    for col, panel in enumerate(panels, start=1):
        stats_by_arm = {}
        for arm in arms:
            stats = _arm_stats(history, panel["metric"], arm,
                               smooth=panel.get("smooth", 1))
            if stats is not None:
                stats_by_arm[arm] = stats
        for arm, stats in stats_by_arm.items():
            style = styles[arm]
            _band_and_line(fig, stats, color=style["color"],
                           dash=style["dash"], width=style["width"],
                           label=labels[arm], group=arm,
                           show_legend=(col == 1), marker=panel.get("marker", False),
                           row=1, col=col)
        kind, y_range = _tight_y(list(stats_by_arm.values()), left,
                                 force_log=panel.get("log"))
        fig.update_xaxes(**_log_x_axis(left, xmax), row=1, col=col)
        fig.update_yaxes(**_y_axis(kind, y_range), row=1, col=col)
    _base_layout(fig, title=title, n_cols=n, height=PANEL_H + 185)
    fig.write_image(out_path, scale=2)
    return out_path


def pair_row_figure(history: pd.DataFrame, panels: Sequence[Tuple[str, str]],
                    lines: Sequence[Tuple[str, str, str, str]], *,
                    title: str, xlabel: str, ylabel: str, out_path: str,
                    smooth: int = 1,
                    y_range: Tuple[float, float] = (-0.15, 1.04)) -> str:
    """Side-by-side pair-cosine panels on one shared y axis.

    ``panels`` is (config, panel title) per panel; ``lines`` is
    (metric key, label, color, dash) per distribution.  The y title prints once
    on the left, the x title once below, the zero line is drawn in every panel.
    """
    n = len(panels)
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True,
                        subplot_titles=[p[1] for p in panels],
                        x_title=xlabel, y_title=ylabel,
                        horizontal_spacing=0.05)
    for col, (config, _) in enumerate(panels, start=1):
        arm_rows = history[history["config"] == config]
        # Per-panel x window: arms with shorter runs keep a full panel.
        left, xmax = _x_window(arm_rows if not arm_rows.empty else history)
        for metric, label, color, dash in lines:
            stats = _arm_stats(arm_rows, metric, config, smooth=smooth)
            if stats is None:
                continue
            _band_and_line(fig, stats, color=color, dash=dash, width=2.0,
                           label=label, group=metric, show_legend=(col == 1),
                           marker=False, row=1, col=col)
        fig.update_xaxes(**_log_x_axis(left, xmax), row=1, col=col)
        y_axis = _y_axis("linear", y_range)
        y_axis.update(zeroline=True, zerolinecolor=ZERO_LINE, zerolinewidth=1)
        fig.update_yaxes(**y_axis, row=1, col=col)
    _base_layout(fig, title=title, n_cols=n, height=PANEL_H + 185)
    fig.write_image(out_path, scale=2)
    return out_path


def crossover_figure(history: pd.DataFrame, panels: Sequence[Dict], *,
                     title: str, xlabel: str, out_path: str) -> str:
    """The artifact crossover row for the clean/control/watermarked arms."""
    return curve_row_figure(history, panels, arms=ARM_ORDER, styles=ARM_STYLE,
                            labels=ARM_LABELS, title=title, xlabel=xlabel,
                            out_path=out_path)


def pair_series_figure(history: pd.DataFrame, *, rep: str, title: str,
                       xlabel: str, out_path: str) -> str:
    """Control and watermarked panels, the three distributions with ``null`` as
    the grey ~0 reference.  ``rep`` selects the metric keys
    ``pair/<distribution>/<rep>``."""
    lines = [(f"pair/{dist}/{rep}", label, color, dash)
             for dist, label, color, dash in PAIR_LINES]
    panels = [("random_control", ARM_LABELS["random_control"]),
              ("watermarked", ARM_LABELS["watermarked"])]
    return pair_row_figure(history, panels, lines, title=title, xlabel=xlabel,
                           ylabel="centered pair cosine", out_path=out_path)


def pattern_alone(index: int, h: int, w: int, opacity: float, *,
                  modulus: Optional[int], tile_px: int, bit_capacity: int,
                  anchor: str, repeat: bool = True,
                  random_anchor: bool = False) -> np.ndarray:
    """The additive delta on mid-gray, contrast-stretched: at training opacity
    the pattern alone is nearly invisible, which is the whole reason this pane
    exists.  ``repeat=False`` renders the random control's field instead (same
    renderer and per-tile RMS, no repeated tile; ``random_anchor`` mirrors
    apply_watermark's control-anchor choice)."""
    if repeat:
        tile = _delta_tile(index, opacity, modulus, tile_px, bit_capacity, anchor)
        delta = _tile_to_frame(tile, h, w).numpy()
    else:
        delta = _random_tiled_frame(index, h, w, opacity, modulus, tile_px,
                                    bit_capacity,
                                    anchor if random_anchor else "none").numpy()
    return np.clip(0.5 + delta / (6.0 * delta.std()), 0.0, 1.0)


def _to_rgb_uint8(im) -> np.ndarray:
    arr = np.asarray(im)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3]


def quad_panel_figure(panes: Sequence[Tuple], *, title: str,
                      out_path: str) -> str:
    """The 2x2 stimulus panel: clean / pattern alone / watermarked / random
    control.  ``panes`` is four (image, pane_title) tuples in that order
    (images are PILs or arrays; float arrays are treated as [0, 1] grayscale)."""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[p[1] for p in panes],
                        horizontal_spacing=0.02, vertical_spacing=0.06)
    for i, pane in enumerate(panes):
        row, col = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Image(z=_to_rgb_uint8(pane[0])), row=row, col=col)
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)
    fig.update_layout(
        template="simple_white", font=FONT,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)),
        width=720, height=790, margin=dict(l=10, r=10, t=80, b=10))
    for ann in fig.layout.annotations:
        ann.font = dict(size=13)
    fig.write_image(out_path, scale=2)
    return out_path


def image_pair_figure(panes: Sequence[Tuple], *, title: str,
                      out_path: str) -> str:
    """Two stimulus images side by side with no auxiliary reference panes."""
    if len(panes) != 2:
        raise ValueError("image_pair_figure requires exactly two panes")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[p[1] for p in panes],
                        horizontal_spacing=0.025)
    for col, pane in enumerate(panes, start=1):
        fig.add_trace(go.Image(z=_to_rgb_uint8(pane[0])), row=1, col=col)
        fig.update_xaxes(visible=False, row=1, col=col)
        fig.update_yaxes(visible=False, row=1, col=col)
    fig.update_layout(
        template="simple_white", font=FONT,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)),
        width=720, height=430, margin=dict(l=10, r=10, t=80, b=10))
    for ann in fig.layout.annotations:
        ann.font = dict(size=13)
    fig.write_image(out_path, scale=2)
    return out_path


# ---------------------------------------------------------------------------- #
# Matplotlib layer -- the rcParams profile, EMA smoothing, export helper, and
# log-step axis the per-case graph scripts share (the LeWM figures are bespoke
# shapes and build on these primitives instead of the constructions above).
# ---------------------------------------------------------------------------- #
def log_step_axis(ax, xmax: Optional[float] = None,
                  xlabel: str = "training step (log scale)", *,
                  xmin: Optional[float] = None) -> None:
    """Log x with explicit, human-readable step ticks from X_LEFT to the data end.

    On full-length runs the axis starts at X_LEFT (the early plateau left of it
    carries no signal); on short runs (smokes, truncated data) it falls back to
    the data range so nothing is clipped away.  With ``xmax`` omitted the data
    already on the axes decides; an explicit ``xmin`` is always honoured as the
    left edge (keyword-only, so a caller cannot mistake it for ``xmax``).
    """
    if xmax is None:
        xs = [x for line in ax.get_lines() for x in line.get_xdata() if x > 0]
        xmax = max(xs) if xs else float(X_LEFT)
        if xmin is None:
            xmin = min(xs) if xs else 1.0
    if xmin is None:
        xmin = 1.0
    left = X_LEFT if (xmax > 2 * X_LEFT and xmin <= X_LEFT) else max(1.0, xmin)
    ax.set_xscale("log")
    ax.set_xlim(left, xmax * 1.02)
    ticks = [t for t in LOG_TICKS if left <= t <= xmax * 1.02]
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t / 1000:g}k" for t in ticks])
        ax.minorticks_off()
    ax.set_xlabel(xlabel)

def animated_watermark_panel(frames: Sequence[Dict], *, title: str,
                             out_path: str,
                             seconds_per_frame: float = 2.5) -> str:
    """The stimulus panel's looping companion: one frame per source image, so
    the invisibility claim is visibly not about a hand-picked example.

    Each frame is a 2x3 grid -- clean / watermarked / random control on top,
    and below the two contrast-stretched pattern fields, where the whole
    experimental contrast is legible: the watermark is ONE tile repeated (a
    compact per-image key, changing with every image) while the control never
    repeats (matched energy, no key).  ``frames`` is a sequence of dicts with
    keys ``clean``, ``watermarked``, ``control`` (images), ``pattern_wm``,
    ``pattern_ctrl`` (arrays in [0,1]), ``label`` (str).
    """
    from PIL import Image as PILImage

    rendered = []
    for frame in frames:
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.2))
        panes = [
            (frame["clean"], "clean", {}),
            (frame["watermarked"], "watermarked", {}),
            (frame["control"], "random control", {}),
            (None, "", {}),
            (frame["pattern_wm"],
             "watermark pattern (stretched)\none tile, repeated — a compact key",
             {"cmap": "gray", "vmin": 0, "vmax": 1}),
            (frame["pattern_ctrl"],
             "control pattern (stretched)\nnever repeats — no key",
             {"cmap": "gray", "vmin": 0, "vmax": 1}),
        ]
        for ax, (im, pane_title, kw) in zip(axes.flat, panes):
            ax.set_xticks([])
            ax.set_yticks([])
            if im is None:
                ax.axis("off")
                ax.text(0.5, 0.5,
                        f"{frame['label']}\n\nthe key changes with every\n"
                        "image; the control never\nforms one",
                        ha="center", va="center", fontsize=11, color="0.35",
                        transform=ax.transAxes)
                continue
            ax.imshow(im, **kw)
            ax.set_title(pane_title, fontsize=11)
        fig.suptitle(title, fontsize=12, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        # Room for the second row's two-line pane titles.
        fig.subplots_adjust(hspace=0.22)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        rendered.append(PILImage.fromarray(rgb.copy()))
        plt.close(fig)
    rendered[0].save(out_path, save_all=True, append_images=rendered[1:],
                     duration=int(seconds_per_frame * 1000), loop=0)
    return out_path


def animated_image_pair_panel(frames: Sequence[Dict], *, title: str,
                              out_path: str,
                              seconds_per_frame: float = 2.5) -> str:
    """Looping watermarked/control comparison without clean or pattern panes."""
    from PIL import Image as PILImage

    rendered = []
    for frame in frames:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
        for ax, im, pane_title in (
                (axes[0], frame["watermarked"], "watermarked"),
                (axes[1], frame["control"], "random control")):
            ax.imshow(im)
            ax.set_title(pane_title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(title, fontsize=12, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        rendered.append(PILImage.fromarray(rgb.copy()))
        plt.close(fig)
    rendered[0].save(out_path, save_all=True, append_images=rendered[1:],
                     duration=int(seconds_per_frame * 1000), loop=0)
    return out_path


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
        }
    )

def smooth(values, weight: float = 0.0):
    """Optional EMA smoothing; weight=0 returns the input unchanged."""
    values = np.asarray(values, dtype=float)
    if weight <= 0 or len(values) == 0:
        return values
    out = np.empty_like(values)
    acc = values[0]
    for i, v in enumerate(values):
        acc = weight * acc + (1 - weight) * v
        out[i] = acc
    return out

def save_fig(fig, path):
    import pathlib

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path
