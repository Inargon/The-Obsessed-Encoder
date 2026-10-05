#!/usr/bin/env python3
"""Create PNG-only matched rollout showcases where repair succeeds and JEPA fails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from additional_files.compose_matched_rollouts import episode_categories, read_video


def green_goal_mask(frame: np.ndarray, ignore_corner: int = 12) -> np.ndarray:
    """Approximate the translucent green PushT goal pixels.

    The metric is used only to rank already-labelled repair-only successes.
    Removing the top-left corner prevents the synthetic tag from affecting it.
    """
    rgb = np.asarray(frame)[..., :3].astype(np.int16)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    mask = (green >= red + 12) & (green >= blue + 8) & (green >= 80)
    mask[:ignore_corner, :ignore_corner] = False
    return mask


def extract_panel(frame: np.ndarray, panel: str) -> np.ndarray:
    """Extract agent/dataset/goal from stable-worldmodel's panel video frame."""
    if panel not in {"agent", "dataset", "goal"}:
        raise ValueError(f"unknown panel {panel!r}")
    array = np.asarray(frame)
    height, width = array.shape[:2]
    # Saved evaluation videos concatenate three square views horizontally and
    # may append a short label strip. Plain single-view frames pass through.
    if width < 2 * height:
        return array
    panel_width = width // 3
    index = {"agent": 0, "dataset": 1, "goal": 2}[panel]
    left = index * panel_width
    side = min(height, panel_width)
    return array[:side, left : left + panel_width]


def terminal_goal_residual(frames: list[np.ndarray], tail: int = 3) -> float:
    """Fraction of visible goal-colored pixels near termination (lower is better)."""
    selected = frames[-min(tail, len(frames)) :]
    counts = [green_goal_mask(extract_panel(frame, "agent")).mean() for frame in selected]
    return float(np.mean(counts))


def rank_repair_only(jepa_summary: dict, repair_summary: dict) -> list[dict]:
    count = int(jepa_summary["num_eval"])
    categories = episode_categories(
        jepa_summary["successful_episode_indices"],
        repair_summary["successful_episode_indices"],
        count,
    )
    ranked = []
    for episode in categories["repair_only"]:
        jepa_path = Path(jepa_summary["video_dir"]) / f"env_{episode}.mp4"
        repair_path = Path(repair_summary["video_dir"]) / f"env_{episode}.mp4"
        jepa_frames = read_video(jepa_path)
        repair_frames = read_video(repair_path)
        repair_residual = terminal_goal_residual(repair_frames)
        jepa_residual = terminal_goal_residual(jepa_frames)
        ranked.append(
            {
                "episode": episode,
                "repair_terminal_goal_residual": repair_residual,
                "jepa_terminal_goal_residual": jepa_residual,
                "residual_gap": jepa_residual - repair_residual,
                "jepa_video": str(jepa_path),
                "repair_video": str(repair_path),
            }
        )
    # First demand tight repair alignment; use the JEPA-minus-repair gap only
    # as the tie breaker. This avoids selecting a poor repair just because the
    # comparator is even worse.
    return sorted(
        ranked,
        key=lambda row: (
            row["repair_terminal_goal_residual"],
            -row["residual_gap"],
            row["episode"],
        ),
    )


def _fit(frame: np.ndarray, size: int = 260):
    from PIL import Image

    return Image.fromarray(frame).convert("RGB").resize((size, size))


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def render_pair(row: dict, output: Path, columns: int = 5) -> None:
    from PIL import Image, ImageDraw, ImageOps

    streams = [read_video(Path(row["jepa_video"])), read_video(Path(row["repair_video"]))]
    sampled = []
    for frames in streams:
        indices = np.rint(np.linspace(0, len(frames) - 1, columns)).astype(int)
        sampled.append([_fit(extract_panel(frames[index], "agent")) for index in indices])
    goal = _fit(extract_panel(streams[0][-1], "goal"))
    left = 225
    header = 112
    gap = 14
    goal_gap = 44
    cell = sampled[0][0].width
    width = left + columns * cell + (columns - 1) * gap + goal_gap + cell + 20
    height = header + 2 * cell + gap + 24
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 14),
        f"Matched tagged PushT — episode {row['episode']}",
        fill="black",
        font=_font(25, bold=True),
    )
    draw.text(
        (20, 52),
        "EMA repair succeeds; JEPA fails  |  selected by a predefined terminal-alignment metric",
        fill="#555555",
        font=_font(17),
    )
    times = np.linspace(0, 100, columns).round().astype(int)
    for col in range(columns):
        label = "Start" if col == 0 else "End" if col == columns - 1 else f"{times[col]}%"
        x = left + col * (cell + gap)
        draw.text((x + 8, 84), label, fill="#333333", font=_font(16, bold=True))
    labels = (("JEPA", "FAIL", "#d55e00"), ("EMA repair", "SUCCESS", "#aa3377"))
    for row_index, ((label, outcome, color), frames) in enumerate(zip(labels, sampled)):
        y = header + row_index * (cell + gap)
        draw.text((20, y + cell // 2 - 30), label, fill=color, font=_font(21, bold=True))
        draw.text((20, y + cell // 2 + 4), outcome, fill=color, font=_font(17, bold=True))
        for col, frame in enumerate(frames):
            bordered = ImageOps.expand(frame, border=2, fill="#cccccc")
            canvas.paste(bordered, (left + col * (cell + gap), y))
    goal_x = left + columns * cell + (columns - 1) * gap + goal_gap
    goal_y = header + (cell + gap) // 2
    draw.text((goal_x + 8, 84), "Goal reference", fill="#d62728", font=_font(16, bold=True))
    goal = ImageOps.expand(goal, border=4, fill="#d62728")
    canvas.paste(goal, (goal_x, goal_y))
    canvas.save(output, optimize=True)


def render_overview(selected: list[dict], image_paths: list[Path], output: Path) -> None:
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in image_paths]
    width = max(image.width for image in images)
    title_height = 52
    canvas = Image.new(
        "RGB", (width, title_height + sum(image.height for image in images)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 12),
        "Matched qualitative examples: EMA repair succeeds while JEPA fails",
        fill="black",
    )
    y = title_height
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jepa-summary", type=Path, required=True)
    parser.add_argument("--repair-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")
    jepa = json.loads(args.jepa_summary.read_text(encoding="utf-8"))
    repair = json.loads(args.repair_summary.read_text(encoding="utf-8"))
    if (jepa["num_eval"], jepa["seed"]) != (repair["num_eval"], repair["seed"]):
        raise ValueError("rollout groups are not matched")
    ranked = rank_repair_only(jepa, repair)
    if len(ranked) < args.examples:
        raise ValueError(f"only {len(ranked)} repair-only episodes are available")
    selected = ranked[: args.examples]
    args.out_dir.mkdir(parents=True)
    images = []
    for rank, row in enumerate(selected, 1):
        path = args.out_dir / f"repair-only-{rank:02d}-episode-{row['episode']:02d}.png"
        render_pair(row, path, columns=args.columns)
        images.append(path)
    render_overview(selected, images, args.out_dir / "repair-only-overview.png")
    payload = {
        "protocol": {
            "task": "tagged PushT",
            "matched_group": {"seed": jepa["seed"], "num_eval": jepa["num_eval"]},
            "eligibility": "EMA repair success and JEPA failure",
            "ranking": "ascending repair terminal visible-goal residual in the agent panel; descending residual gap as tie-breaker",
            "artifact_scope": "qualitative showcase; full-group success rates remain the quantitative result",
            "output": "PNG only; no GIF",
        },
        "success_rate": {"jepa": jepa["success_rate"], "repair": repair["success_rate"]},
        "eligible_count": len(ranked),
        "selected": selected,
        "all_ranked": ranked,
    }
    (args.out_dir / "repair-only-showcase.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**payload, "all_ranked": "stored in JSON"}, indent=2))
    print(f"REPAIR_ONLY_SHOWCASE_COMPLETE {args.out_dir}")


if __name__ == "__main__":
    main()
