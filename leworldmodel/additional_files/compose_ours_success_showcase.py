#!/usr/bin/env python3
"""Select and render the two best successful Ours rollouts per task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from additional_files.compose_matched_rollouts import read_video
from additional_files.compose_repair_only_showcase import _fit, _font, extract_panel


TASK_DISPLAY = {
    "clean_pusht": "Clean PushT",
    "tagged_pusht": "Tagged PushT",
    "tworoom": "TwoRoom",
    "cube": "Cube",
}


def terminal_agent_goal_rmse(frames: list[np.ndarray], tail: int = 3) -> float:
    """Terminal RGB RMSE between the agent and goal panels (lower is better)."""
    values = []
    for frame in frames[-min(tail, len(frames)) :]:
        agent = extract_panel(frame, "agent").astype(np.float32) / 255.0
        goal = extract_panel(frame, "goal").astype(np.float32) / 255.0
        if agent.shape != goal.shape:
            raise ValueError(f"agent/goal panel mismatch: {agent.shape} vs {goal.shape}")
        # The 5x5 nuisance tag is not task success. Exclude a conservative
        # corner so tagged and clean task rankings use the same physical metric.
        agent = agent.copy()
        goal = goal.copy()
        agent[:12, :12] = goal[:12, :12]
        values.append(float(np.sqrt(np.mean((agent - goal) ** 2))))
    return float(np.mean(values))


def red_agent_centroid(frame: np.ndarray) -> tuple[float, float]:
    """Locate TwoRoom's blurred red agent and return its (x, y) centroid."""
    rgb = np.asarray(frame)[..., :3].astype(np.float32)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    # Weight by red dominance rather than using a hard binary blob boundary;
    # this remains stable across video compression and the rendered glow.
    weights = np.maximum(red - np.maximum(green, blue), 0.0)
    weights[(red < 100) | (weights < 20)] = 0.0
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("could not locate TwoRoom's red agent")
    y, x = np.indices(weights.shape)
    return float((x * weights).sum() / total), float((y * weights).sum() / total)


def is_cross_room(frames: list[np.ndarray]) -> tuple[bool, dict]:
    """Return whether TwoRoom start and goal lie on opposite wall sides."""
    start = extract_panel(frames[0], "agent")
    goal = extract_panel(frames[-1], "goal")
    start_x, start_y = red_agent_centroid(start)
    goal_x, goal_y = red_agent_centroid(goal)
    divider_x = 0.5 * start.shape[1]
    cross_room = (start_x - divider_x) * (goal_x - divider_x) < 0
    return bool(cross_room), {
        "start_agent_xy": [start_x, start_y],
        "goal_agent_xy": [goal_x, goal_y],
        "divider_x": divider_x,
        "start_side": "left" if start_x < divider_x else "right",
        "goal_side": "left" if goal_x < divider_x else "right",
    }


def rank_successes(summary: dict, task: str) -> list[dict]:
    ranked = []
    video_dir = Path(summary["video_dir"])
    for episode in summary["successful_episode_indices"]:
        video = video_dir / f"env_{int(episode)}.mp4"
        frames = read_video(video)
        row = {
            "episode": int(episode),
            "terminal_agent_goal_rmse": terminal_agent_goal_rmse(frames),
            "video": str(video),
        }
        if task == "tworoom":
            eligible, geometry = is_cross_room(frames)
            row.update(geometry)
            row["cross_room"] = eligible
            if not eligible:
                continue
        ranked.append(row)
    return sorted(ranked, key=lambda row: (row["terminal_agent_goal_rmse"], row["episode"]))


def render_example(task: str, row: dict, output: Path, columns: int = 5) -> None:
    from PIL import Image, ImageDraw, ImageOps

    frames = read_video(Path(row["video"]))
    indices = np.rint(np.linspace(0, len(frames) - 1, columns)).astype(int)
    sampled = [_fit(extract_panel(frames[index], "agent")) for index in indices]
    goal = _fit(extract_panel(frames[-1], "goal"))
    left = 170
    header = 112
    gap = 14
    goal_gap = 44
    cell = sampled[0].width
    width = left + columns * cell + (columns - 1) * gap + goal_gap + cell + 20
    height = header + cell + 28
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 14),
        f"{TASK_DISPLAY[task]} — Ours — episode {row['episode']}",
        fill="black",
        font=_font(25, bold=True),
    )
    if task == "tworoom":
        subtitle = "Successful cross-room rollout; start and goal are on opposite wall sides"
    elif task == "tagged_pusht":
        subtitle = (
            "Successful tagged-observation rollout; panels show the simulator render "
            "without the observation overlay"
        )
    else:
        subtitle = "Successful rollout selected by terminal agent-to-goal alignment"
    draw.text(
        (20, 52),
        subtitle,
        fill="#555555",
        font=_font(17),
    )
    times = np.linspace(0, 100, columns).round().astype(int)
    for col in range(columns):
        label = "Start" if col == 0 else "End" if col == columns - 1 else f"{times[col]}%"
        x = left + col * (cell + gap)
        draw.text((x + 8, 84), label, fill="#333333", font=_font(16, bold=True))
    draw.text((20, header + cell // 2 - 20), "Ours", fill="#aa3377", font=_font(22, bold=True))
    draw.text((20, header + cell // 2 + 14), "SUCCESS", fill="#aa3377", font=_font(17, bold=True))
    for col, frame in enumerate(sampled):
        bordered = ImageOps.expand(frame, border=2, fill="#cccccc")
        canvas.paste(bordered, (left + col * (cell + gap), header))
    goal_x = left + columns * cell + (columns - 1) * gap + goal_gap
    draw.text((goal_x + 8, 84), "Goal reference", fill="#d62728", font=_font(16, bold=True))
    canvas.paste(ImageOps.expand(goal, border=4, fill="#d62728"), (goal_x, header))
    canvas.save(output, optimize=True)


def render_overview(task: str, paths: list[Path], output: Path) -> None:
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in paths]
    title_height = 58
    canvas = Image.new(
        "RGB",
        (max(image.width for image in images), title_height + sum(image.height for image in images)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (18, 14),
        f"{TASK_DISPLAY[task]} — top-2 successful Ours rollouts",
        fill="black",
        font=_font(24, bold=True),
    )
    y = title_height
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASK_DISPLAY))
    parser.add_argument("--per-task", type=int, default=2)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    unknown = sorted(set(tasks) - set(TASK_DISPLAY))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    payload = {
        "protocol": {
            "method_display": "Ours",
            "selection": "successful episodes ranked by terminal agent-to-goal RGB RMSE",
            "tworoom_eligibility": (
                "start and goal red-agent centroids must lie on opposite sides "
                "of the central wall"
            ),
            "per_task": args.per_task,
            "output": "PNG only; no GIF",
            "scope": "qualitative showcase; complete fixed-group success rates remain quantitative evidence",
        },
        "tasks": {},
        "not_in_source": {
            "reacher": "historical-runtime evaluation was not part of the modern rollout-video archive"
        },
    }
    args.out_dir.mkdir(parents=True)
    for task in tasks:
        summary_path = args.source / task / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ranked = rank_successes(summary, task)
        if len(ranked) < args.per_task:
            raise ValueError(f"{task}: only {len(ranked)} successful episodes")
        selected = ranked[: args.per_task]
        task_dir = args.out_dir / task
        task_dir.mkdir()
        images = []
        for rank, row in enumerate(selected, 1):
            path = task_dir / f"top-{rank:02d}-episode-{row['episode']:02d}.png"
            render_example(task, row, path)
            images.append(path)
        render_overview(task, images, task_dir / f"{task}-top2-overview.png")
        payload["tasks"][task] = {
            "success_rate": summary["success_rate"],
            "eligible_successes": len(ranked),
            "selected": selected,
            "all_ranked": ranked,
        }
    (args.out_dir / "ours-success-showcase.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**payload, "tasks": "stored in JSON"}, indent=2))
    print(f"OURS_SUCCESS_SHOWCASE_COMPLETE {args.out_dir}")


if __name__ == "__main__":
    main()
