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


def split_reacher_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract the latest current and goal views from the historical 2x2 tile.

    The historical recorder stores two context frames down the left column and
    two goal frames down the right column.  The lower row is the latest pair.
    """
    array = np.asarray(frame)[..., :3]
    height, width = array.shape[:2]
    if height < 2 or width < 2:
        raise ValueError(f"invalid historical Reacher frame shape: {array.shape}")
    half_height = height // 2
    half_width = width // 2
    return array[half_height:, :half_width], array[half_height:, half_width:]


def orange_fingertip(frame: np.ndarray) -> tuple[float, float]:
    """Locate the orange distal endpoint; the white point is the fixed base."""
    rgb = np.asarray(frame)[..., :3].astype(np.int16)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    mask = (
        (red > 150)
        & (red > green + 25)
        & (green > 40)
        & (green < 210)
        & (blue < 130)
    )
    # Ignore tile borders, which can acquire warm compression artifacts.
    border = max(2, min(mask.shape) // 50)
    mask[:border] = False
    mask[-border:] = False
    mask[:, :border] = False
    mask[:, -border:] = False
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError("could not locate the orange Reacher fingertip")
    return float(x.mean()), float(y.mean())


def fingertip_geometry(frames: list[np.ndarray]) -> dict:
    current_start, goal = split_reacher_frame(frames[0])
    current_end, _ = split_reacher_frame(frames[-1])
    target = np.asarray(orange_fingertip(goal))
    start_tip = np.asarray(orange_fingertip(current_start))
    end_tip = np.asarray(orange_fingertip(current_end))
    width = float(current_start.shape[1])
    return {
        "target_xy": target.tolist(),
        "start_tip_xy": start_tip.tolist(),
        "end_tip_xy": end_tip.tolist(),
        "start_distance_over_width": float(np.linalg.norm(start_tip - target) / width),
        "end_distance_over_width": float(np.linalg.norm(end_tip - target) / width),
    }


def rank_successes(source: Path, successes: list[bool]) -> list[dict]:
    ranked = []
    for episode, success in enumerate(successes):
        video = source / f"rollout_{episode}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        if not success:
            continue
        frames = read_video(video)
        geometry = fingertip_geometry(frames)
        ranked.append(
            {
                "episode": episode,
                "video": str(video),
                **geometry,
            }
        )
    # Prefer examples that begin far from the target and visibly finish near
    # it.  Success eligibility itself always comes from the simulator label.
    return sorted(
        ranked,
        key=lambda row: (
            -row["start_distance_over_width"],
            row["end_distance_over_width"],
            row["episode"],
        ),
    )


def annotate_target(
    frame: np.ndarray,
    target_xy: tuple[float, float],
    tip_xy: tuple[float, float],
):
    """Overlay a goal-derived red target disk and preserve the orange tip."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame)[..., :3]).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(7, round(min(image.size) * 0.075))
    tx, ty = target_xy
    draw.ellipse(
        (tx - radius, ty - radius, tx + radius, ty + radius),
        fill=(235, 65, 65, 105),
        outline=(210, 25, 25, 255),
        width=max(2, radius // 4),
    )
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    px, py = tip_xy
    tip_radius = max(3, radius // 3)
    draw.ellipse(
        (px - tip_radius, py - tip_radius, px + tip_radius, py + tip_radius),
        fill=(245, 145, 35, 255),
        outline=(255, 255, 255, 255),
        width=max(1, tip_radius // 2),
    )
    return image.convert("RGB")


def render_example(row: dict, output: Path, columns: int = 5) -> None:
    from PIL import Image, ImageDraw, ImageOps

    frames = read_video(Path(row["video"]))
    indices = np.rint(np.linspace(0, len(frames) - 1, columns)).astype(int)
    target = tuple(row["target_xy"])
    sampled = []
    distances = []
    for index in indices:
        current, _ = split_reacher_frame(frames[index])
        tip = orange_fingertip(current)
        annotated = annotate_target(current, target, tip)
        sampled.append(_fit(np.asarray(annotated)))
        distances.append(float(np.linalg.norm(np.asarray(tip) - np.asarray(target)) / current.shape[1]))
    left = 170
    header = 112
    gap = 14
    cell = sampled[0].width
    width = left + columns * cell + (columns - 1) * gap + 20
    height = header + cell + 56
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
        "Red disk = goal region; orange point = controlled fingertip",
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
        distance_label = f"d/W={distances[column]:.3f}"
        if column == columns - 1:
            distance_label += "  SUCCESS"
        draw.text(
            (left + column * (cell + gap) + 8, header + cell + 8),
            distance_label,
            fill="#18864b" if column == columns - 1 else "#555555",
            font=_font(15, bold=column == columns - 1),
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
            "ranking": (
                "descending initial fingertip-to-goal distance; ascending terminal "
                "distance as tie-breaker"
            ),
            "visualization": (
                "latest current view extracted from the lower-left historical tile; "
                "red target disk derived from the lower-right goal fingertip"
            ),
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
