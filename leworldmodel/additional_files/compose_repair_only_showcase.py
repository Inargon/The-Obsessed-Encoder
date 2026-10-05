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


def terminal_goal_residual(frames: list[np.ndarray], tail: int = 3) -> float:
    """Fraction of visible goal-colored pixels near termination (lower is better)."""
    selected = frames[-min(tail, len(frames)) :]
    counts = [green_goal_mask(frame).mean() for frame in selected]
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


def _fit(frame: np.ndarray, size: int = 250):
    from PIL import Image

    return Image.fromarray(frame).convert("RGB").resize((size, size))


def render_pair(row: dict, output: Path, columns: int = 6) -> None:
    from PIL import Image, ImageDraw

    streams = [read_video(Path(row["jepa_video"])), read_video(Path(row["repair_video"]))]
    sampled = []
    for frames in streams:
        indices = np.rint(np.linspace(0, len(frames) - 1, columns)).astype(int)
        sampled.append([_fit(frames[index]) for index in indices])
    left = 150
    header = 78
    cell = sampled[0][0].width
    canvas = Image.new("RGB", (left + columns * cell, header + 2 * cell), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        f"Matched tagged PushT — episode {row['episode']}: repair succeeds, JEPA fails",
        fill="black",
    )
    draw.text(
        (12, 38),
        "Selection: smallest repair terminal goal residual (predefined pixel metric)",
        fill="#555555",
    )
    for col in range(columns):
        label = "start" if col == 0 else "end" if col == columns - 1 else f"t{col}"
        draw.text((left + col * cell + 8, 58), label, fill="black")
    labels = (("JEPA (fail)", "#d55e00"), ("EMA repair (success)", "#aa3377"))
    for row_index, ((label, color), frames) in enumerate(zip(labels, sampled)):
        y = header + row_index * cell
        draw.text((12, y + cell // 2 - 10), label, fill=color)
        for col, frame in enumerate(frames):
            canvas.paste(frame, (left + col * cell, y))
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
    parser.add_argument("--columns", type=int, default=6)
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
            "ranking": "ascending repair terminal visible-goal residual; descending residual gap as tie-breaker",
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
