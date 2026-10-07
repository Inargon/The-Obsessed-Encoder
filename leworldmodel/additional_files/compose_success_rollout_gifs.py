#!/usr/bin/env python3
"""Compose compact GIFs of successful Ours rollouts from existing videos.

No policy is rerun.  Modern-task episodes are selected only after simulator
success filtering, using the same ranking as the static showcase.  Tagged
PushT reconstructs the exact evaluation-time observation tag that is absent
from the simulator recorder and magnifies it in a labelled inset.  Reacher
overlays the exact recorded goal arm as a translucent ghost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from additional_files.compose_matched_rollouts import read_video
from additional_files.compose_ours_success_showcase import (
    TASK_DISPLAY,
    rank_successes,
)
from additional_files.compose_reacher_success_showcase import (
    overlay_goal_pose,
    parse_historical_metrics,
    rank_successes as rank_reacher_successes,
    split_reacher_frame,
)
from additional_files.compose_repair_only_showcase import _font, extract_panel


MODERN_TASKS = ("clean_pusht", "tagged_pusht", "tworoom", "cube")
DISPLAY = {
    **TASK_DISPLAY,
    "reacher": "Reacher",
}
OURS = "#aa3377"
TAG_RED = "#d62728"
SUCCESS_GREEN = "#18864b"


def synchronized_indices(length: int, count: int) -> np.ndarray:
    if length < 1 or count < 1:
        raise ValueError("length and count must be positive")
    return np.rint(np.linspace(0, length - 1, count)).astype(int)


def stamp_rgb(frame: np.ndarray, color: np.ndarray, size: int = 5) -> np.ndarray:
    output = np.asarray(frame)[..., :3].copy()
    if output.dtype != np.uint8 or output.ndim != 3 or output.shape[-1] != 3:
        raise ValueError("expected uint8 HWC RGB frame")
    if size < 1 or size > min(output.shape[:2]):
        raise ValueError("invalid tag size")
    output[:size, :size] = np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
    return output


def fit_square(frame: np.ndarray, size: int):
    from PIL import Image

    return Image.fromarray(np.asarray(frame)[..., :3]).convert("RGB").resize((size, size))


def save_gif(frames, output: Path, fps: int) -> None:
    if not frames:
        raise ValueError("cannot save an empty GIF")
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [max(40, round(1000 / fps))] * len(frames)
    durations[-1] += 900
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def goal_thumbnail(goal: np.ndarray, size: int = 58):
    from PIL import ImageOps

    return ImageOps.expand(fit_square(goal, size), border=2, fill=TAG_RED)


def render_modern_pair(task: str, rows: list[dict], output: Path, *, frames: int, fps: int) -> None:
    from PIL import Image, ImageDraw, ImageOps

    if len(rows) != 2:
        raise ValueError("modern pair renderer expects two episodes")
    streams = [read_video(Path(row["video"])) for row in rows]
    indices = [synchronized_indices(len(stream), frames) for stream in streams]
    goals = [extract_panel(stream[-1], "goal") for stream in streams]
    cell, gap, margin, header, footer = 300, 22, 18, 104, 42
    width = margin * 2 + cell * 2 + gap
    height = header + cell + footer
    subtitle = {
        "clean_pusht": "Two simulator-labelled successes; red inset = exact recorded goal",
        "tworoom": "Cross-room successes only; red inset = exact recorded goal",
        "cube": "Two simulator-labelled successes; red inset = exact recorded goal",
    }[task]
    output_frames = []
    for step in range(frames):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 8), f"{DISPLAY[task]} — successful Ours rollouts",
                  fill="black", font=_font(24, bold=True))
        draw.text((margin, 40), subtitle, fill="#555555", font=_font(15))
        progress = round(100 * step / max(frames - 1, 1))
        for column, (row, stream, stream_indices, goal) in enumerate(
            zip(rows, streams, indices, goals)
        ):
            x = margin + column * (cell + gap)
            current = extract_panel(stream[int(stream_indices[step])], "agent")
            panel = fit_square(current, cell)
            border = SUCCESS_GREEN if step == frames - 1 else "#cccccc"
            canvas.paste(ImageOps.expand(panel, border=3, fill=border), (x - 3, header - 3))
            draw.text((x, 69), f"episode {row['episode']}", fill=OURS,
                      font=_font(17, bold=True))
            thumb = goal_thumbnail(goal)
            tx = x + cell - thumb.width
            canvas.paste(thumb, (tx, 35))
            draw.text((tx - 42, 52), "goal", fill=TAG_RED, font=_font(13, bold=True))
            status = "SIMULATOR SUCCESS" if step == frames - 1 else f"progress {progress}%"
            draw.text((x + 4, header + cell + 10), status,
                      fill=SUCCESS_GREEN if step == frames - 1 else "#555555",
                      font=_font(15, bold=step == frames - 1))
        output_frames.append(canvas)
    save_gif(output_frames, output, fps)


def render_tagged_four(rows: list[dict], output: Path, *, frames: int, fps: int) -> None:
    from PIL import Image, ImageDraw, ImageOps
    from additional_files.pixel_tag import PixelTag

    if len(rows) != 4:
        raise ValueError("tagged renderer expects four episodes")
    streams = [read_video(Path(row["video"])) for row in rows]
    indices = [synchronized_indices(len(stream), frames) for stream in streams]
    # The fixed-group evaluator uses EvalTagStamp._VIDEO_KEY == 0 with seed 42.
    tag_color = PixelTag(mode="video", size=5, seed=42).color_for(0)
    cell, gap, margin, header, footer = 220, 12, 16, 122, 38
    width = margin * 2 + 4 * cell + 3 * gap
    height = header + cell + footer
    output_frames = []
    for step in range(frames):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 7), "Tagged PushT — four successful Ours rollouts",
                  fill="black", font=_font(23, bold=True))
        draw.text(
            (margin, 38),
            "Exact episode-constant 5×5 evaluation tag reconstructed; red inset magnifies the observation corner",
            fill="#555555", font=_font(14),
        )
        progress = round(100 * step / max(frames - 1, 1))
        for column, (row, stream, stream_indices) in enumerate(zip(rows, streams, indices)):
            x = margin + column * (cell + gap)
            raw = extract_panel(stream[int(stream_indices[step])], "agent")
            tagged = stamp_rgb(raw, tag_color, size=5)
            panel = fit_square(tagged, cell)
            border = SUCCESS_GREEN if step == frames - 1 else "#cccccc"
            canvas.paste(ImageOps.expand(panel, border=3, fill=border), (x - 3, header - 3))
            crop = Image.fromarray(tagged[:14, :14]).resize((58, 58), Image.Resampling.NEAREST)
            crop = ImageOps.expand(crop, border=2, fill=TAG_RED)
            canvas.paste(crop, (x + 4, 56))
            draw.line((x + 33, 116, x + 2, header + 2), fill=TAG_RED, width=2)
            draw.text((x + 72, 67), f"episode {row['episode']}", fill=OURS,
                      font=_font(15, bold=True))
            draw.text((x + 72, 91), "tag ×12", fill=TAG_RED,
                      font=_font(13, bold=True))
            status = "SUCCESS" if step == frames - 1 else f"{progress}%"
            draw.text((x + 5, header + cell + 9), status,
                      fill=SUCCESS_GREEN if step == frames - 1 else "#555555",
                      font=_font(14, bold=step == frames - 1))
        output_frames.append(canvas)
    save_gif(output_frames, output, fps)


def render_reacher_pair(rows: list[dict], output: Path, *, frames: int, fps: int) -> None:
    from PIL import Image, ImageDraw, ImageOps

    if len(rows) != 2:
        raise ValueError("Reacher renderer expects two episodes")
    streams = [read_video(Path(row["video"])) for row in rows]
    indices = [synchronized_indices(len(stream), frames) for stream in streams]
    cell, gap, margin, header, footer = 300, 22, 18, 82, 42
    width = margin * 2 + cell * 2 + gap
    height = header + cell + footer
    output_frames = []
    for step in range(frames):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 8), "Reacher — successful Ours rollouts",
                  fill="black", font=_font(24, bold=True))
        draw.text((margin, 40), "Yellow = current arm; magenta ghost = exact recorded goal pose",
                  fill="#555555", font=_font(15))
        progress = round(100 * step / max(frames - 1, 1))
        for column, (row, stream, stream_indices) in enumerate(zip(rows, streams, indices)):
            x = margin + column * (cell + gap)
            current, goal = split_reacher_frame(stream[int(stream_indices[step])])
            panel = fit_square(overlay_goal_pose(current, goal), cell)
            border = SUCCESS_GREEN if step == frames - 1 else "#cccccc"
            canvas.paste(ImageOps.expand(panel, border=3, fill=border), (x - 3, header - 3))
            draw.text((x + 8, header + 8), f"episode {row['episode']}", fill=OURS,
                      font=_font(16, bold=True))
            status = "SIMULATOR SUCCESS" if step == frames - 1 else f"progress {progress}%"
            draw.text((x + 4, header + cell + 10), status,
                      fill=SUCCESS_GREEN if step == frames - 1 else "#555555",
                      font=_font(15, bold=step == frames - 1))
        output_frames.append(canvas)
    save_gif(output_frames, output, fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modern-source", type=Path, required=True)
    parser.add_argument("--reacher-source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")
    if min(args.frames, args.fps) < 1:
        raise ValueError("frames and fps must be positive")
    args.out_dir.mkdir(parents=True)
    payload = {
        "protocol": {
            "method_display": "Ours",
            "source": "existing fixed-group evaluation videos; no evaluation rerun",
            "eligibility": "simulator success labels only",
            "modern_ranking": "terminal agent-to-goal RGB RMSE among successes",
            "tworoom_filter": "cross-room successes only",
            "tagged_visualization": (
                "exact seed-42 EvalTagStamp colour reconstructed because the simulator "
                "recorder stores the unmodified render"
            ),
            "reacher_visualization": "exact recorded goal arm overlaid as a magenta ghost",
            "frames_per_gif": args.frames,
            "fps": args.fps,
        },
        "tasks": {},
    }
    for task in MODERN_TASKS:
        summary = json.loads((args.modern_source / task / "summary.json").read_text())
        ranked = rank_successes(summary, task)
        count = 4 if task == "tagged_pusht" else 2
        if len(ranked) < count:
            raise ValueError(f"{task}: only {len(ranked)} eligible successful episodes")
        selected = ranked[:count]
        if task == "tagged_pusht":
            output = args.out_dir / "tagged-pusht-success-four.gif"
            render_tagged_four(selected, output, frames=args.frames, fps=args.fps)
        else:
            output = args.out_dir / f"{task.replace('_', '-')}-success.gif"
            render_modern_pair(task, selected, output, frames=args.frames, fps=args.fps)
        payload["tasks"][task] = {
            "success_rate": summary["success_rate"],
            "selected": selected,
            "output": str(output),
        }

    success_rate, successes = parse_historical_metrics(
        (args.reacher_source / "reacher-historical.txt").read_text()
    )
    reacher_ranked = rank_reacher_successes(args.reacher_source, successes)
    reacher_selected = reacher_ranked[:2]
    reacher_output = args.out_dir / "reacher-success.gif"
    render_reacher_pair(
        reacher_selected, reacher_output, frames=args.frames, fps=args.fps
    )
    payload["tasks"]["reacher"] = {
        "success_rate": success_rate,
        "selected": reacher_selected,
        "output": str(reacher_output),
    }
    (args.out_dir / "success-rollout-gifs.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print("SUCCESS_ROLLOUT_GIFS_COMPLETE", args.out_dir)


if __name__ == "__main__":
    main()
