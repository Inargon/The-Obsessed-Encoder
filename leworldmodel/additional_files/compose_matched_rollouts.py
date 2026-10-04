#!/usr/bin/env python3
"""Compose matched JEPA-versus-repair closed-loop rollout GIFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def episode_categories(jepa_successes, repair_successes, count: int) -> dict[str, list[int]]:
    jepa = set(map(int, jepa_successes))
    repair = set(map(int, repair_successes))
    universe = set(range(count))
    return {
        "repair_only": sorted(repair - jepa),
        "both_success": sorted(repair & jepa),
        "jepa_only": sorted(jepa - repair),
        "both_fail": sorted(universe - (jepa | repair)),
    }


def read_video(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


def _fit(frame: np.ndarray, height: int = 300) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(frame).convert("RGB")
    width = max(1, round(image.width * height / image.height))
    return np.asarray(image.resize((width, height)))


def compose_pair(jepa_frames, repair_frames, title: str, fps: int = 8):
    from PIL import Image, ImageDraw

    output_count = min(80, max(len(jepa_frames), len(repair_frames)))
    jepa_index = np.rint(np.linspace(0, len(jepa_frames) - 1, output_count)).astype(int)
    repair_index = np.rint(np.linspace(0, len(repair_frames) - 1, output_count)).astype(int)
    frames = []
    for left_index, right_index in zip(jepa_index, repair_index):
        left = Image.fromarray(_fit(jepa_frames[left_index]))
        right = Image.fromarray(_fit(repair_frames[right_index]))
        header = 58
        canvas = Image.new("RGB", (left.width + right.width, header + left.height), "white")
        canvas.paste(left, (0, header))
        canvas.paste(right, (left.width, header))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), title, fill="black")
        draw.text((10, 32), "JEPA", fill="#d55e00")
        draw.text((left.width + 10, 32), "EMA orthogonal repair", fill="#aa3377")
        frames.append(canvas)
    duration = max(40, round(1000 / fps))
    return frames, duration


def render_storyboard(frames, output: Path) -> None:
    from PIL import Image, ImageDraw

    positions = np.rint(np.linspace(0, len(frames) - 1, 3)).astype(int)
    selected = [frames[index] for index in positions]
    canvas = Image.new(
        "RGB",
        (selected[0].width * 3, selected[0].height + 30),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, (frame, label) in enumerate(zip(selected, ("start", "middle", "end"))):
        canvas.paste(frame, (column * frame.width, 30))
        draw.text((column * frame.width + 8, 8), label, fill="black")
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jepa-summary", type=Path, required=True)
    parser.add_argument("--repair-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=2)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")
    jepa = json.loads(args.jepa_summary.read_text(encoding="utf-8"))
    repair = json.loads(args.repair_summary.read_text(encoding="utf-8"))
    if (jepa["num_eval"], jepa["seed"]) != (repair["num_eval"], repair["seed"]):
        raise ValueError("rollout groups are not matched")
    categories = episode_categories(
        jepa["successful_episode_indices"],
        repair["successful_episode_indices"],
        int(jepa["num_eval"]),
    )
    args.out_dir.mkdir(parents=True)
    selected = []
    for category in ("repair_only", "both_success", "both_fail", "jepa_only"):
        for episode in categories[category][: args.per_category]:
            left_path = Path(jepa["video_dir"]) / f"env_{episode}.mp4"
            right_path = Path(repair["video_dir"]) / f"env_{episode}.mp4"
            left_frames, right_frames = read_video(left_path), read_video(right_path)
            title = f"Matched tagged PushT episode {episode} — {category.replace('_', ' ')}"
            frames, duration = compose_pair(left_frames, right_frames, title)
            stem = f"{category}-episode-{episode:02d}"
            frames[0].save(
                args.out_dir / f"{stem}.gif",
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                optimize=False,
            )
            render_storyboard(frames, args.out_dir / f"{stem}.png")
            selected.append({
                "category": category,
                "episode": episode,
                "jepa_video": str(left_path),
                "repair_video": str(right_path),
            })
    payload = {
        "protocol": {
            "task": "tagged PushT",
            "matched_group": {"seed": jepa["seed"], "num_eval": jepa["num_eval"]},
            "selection": f"first {args.per_category} episode indices per outcome category",
            "warning": "selection uses success labels only, never visual appearance",
        },
        "success_rate": {"jepa": jepa["success_rate"], "repair": repair["success_rate"]},
        "categories": categories,
        "selected": selected,
    }
    (args.out_dir / "matched-rollouts.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    print(f"MATCHED_ROLLOUT_VISUALS_COMPLETE {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
