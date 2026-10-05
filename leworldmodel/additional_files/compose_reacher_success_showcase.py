#!/usr/bin/env python3
"""Create a PNG-only showcase from historical Reacher rollout videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from additional_files.compose_matched_rollouts import read_video
from additional_files.compose_repair_only_showcase import _fit, _font


def parse_historical_metrics(text: str) -> tuple[float, list[bool]]:
    rate_match = re.search(r"['\"]success_rate['\"]\s*:\s*([0-9.]+)", text)
    mask_match = re.search(
        r"['\"]episode_successes['\"]\s*:\s*array\(\[(.*?)\]\)",
        text,
        flags=re.DOTALL,
    )
    if rate_match is None or mask_match is None:
        raise ValueError("historical success rate or episode mask was not found")
    mask = re.findall(r"\b(True|False)\b", mask_match.group(1))
    if not mask:
        raise ValueError("historical episode mask is empty")
    return float(rate_match.group(1)) / 100.0, [value == "True" for value in mask]


def visual_motion_score(frames: list[np.ndarray], samples: int = 9) -> float:
    """Measure visible state displacement from the rollout's initial frame."""
    indices = np.rint(np.linspace(0, len(frames) - 1, min(samples, len(frames)))).astype(int)
    selected = [np.asarray(frames[index])[..., :3].astype(np.float32) / 255.0 for index in indices]
    initial = selected[0]
    return float(max(np.sqrt(np.mean((frame - initial) ** 2)) for frame in selected[1:]))


def rank_successes(source: Path, successes: list[bool]) -> list[dict]:
    ranked = []
    for episode, success in enumerate(successes):
        video = source / f"rollout_{episode}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        if not success:
            continue
        frames = read_video(video)
        ranked.append(
            {
                "episode": episode,
                "visual_motion_score": visual_motion_score(frames),
                "video": str(video),
            }
        )
    return sorted(ranked, key=lambda row: (-row["visual_motion_score"], row["episode"]))


def render_example(row: dict, output: Path, columns: int = 5) -> None:
    from PIL import Image, ImageDraw, ImageOps

    frames = read_video(Path(row["video"]))
    indices = np.rint(np.linspace(0, len(frames) - 1, columns)).astype(int)
    sampled = [_fit(frames[index]) for index in indices]
    left = 170
    header = 112
    gap = 14
    cell = sampled[0].width
    width = left + columns * cell + (columns - 1) * gap + 20
    height = header + cell + 28
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 14),
        f"Reacher — Ours — episode {row['episode']}",
        fill="black",
        font=_font(25, bold=True),
    )
    draw.text(
        (20, 52),
        "Successful rollout under the pinned historical protocol",
        fill="#555555",
        font=_font(17),
    )
    times = np.linspace(0, 100, columns).round().astype(int)
    for column in range(columns):
        label = "Start" if column == 0 else "End" if column == columns - 1 else f"{times[column]}%"
        x = left + column * (cell + gap)
        draw.text((x + 8, 84), label, fill="#333333", font=_font(16, bold=True))
    draw.text((20, header + cell // 2 - 20), "Ours", fill="#aa3377", font=_font(22, bold=True))
    draw.text((20, header + cell // 2 + 14), "SUCCESS", fill="#aa3377", font=_font(17, bold=True))
    for column, frame in enumerate(sampled):
        canvas.paste(
            ImageOps.expand(frame, border=2, fill="#cccccc"),
            (left + column * (cell + gap), header),
        )
    canvas.save(output, optimize=True)


def render_overview(paths: list[Path], output: Path) -> None:
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
        "Reacher — top-2 successful Ours rollouts",
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
    parser.add_argument("--metrics", default="reacher-historical.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=2)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")
    success_rate, successes = parse_historical_metrics(
        (args.source / args.metrics).read_text(encoding="utf-8")
    )
    ranked = rank_successes(args.source, successes)
    if len(ranked) < args.examples:
        raise ValueError(f"only {len(ranked)} successful Reacher episodes")
    selected = ranked[: args.examples]
    args.out_dir.mkdir(parents=True)
    paths = []
    for rank, row in enumerate(selected, 1):
        path = args.out_dir / f"top-{rank:02d}-episode-{row['episode']:02d}.png"
        render_example(row, path)
        paths.append(path)
    render_overview(paths, args.out_dir / "reacher-top2-overview.png")
    payload = {
        "protocol": {
            "task": "clean Reacher",
            "method_display": "Ours",
            "runtime": "pinned historical Python 3.10 stack",
            "eligibility": "episode_successes is true in the historical 50-episode evaluation",
            "ranking": "descending maximum RGB displacement from the initial frame",
            "selection_scope": "qualitative showcase; the full-group success rate remains quantitative evidence",
            "output": "PNG only; no GIF",
        },
        "num_eval": len(successes),
        "success_rate": success_rate,
        "eligible_successes": len(ranked),
        "selected": selected,
        "all_ranked": ranked,
    }
    (args.out_dir / "reacher-success-showcase.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**payload, "all_ranked": "stored in JSON"}, indent=2))
    print(f"REACHER_SUCCESS_SHOWCASE_COMPLETE {args.out_dir}")


if __name__ == "__main__":
    main()
