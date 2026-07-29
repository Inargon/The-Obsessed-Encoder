#!/usr/bin/env python3
"""Collect smooth PushT push-to-destination demos with a scripted geometric expert.

Produces the randgoal dataset: the block pushed to a *random* destination-T,
with human-like, purposeful pusher motion. The push controller lives in
policy.py (ScriptedPushExpert); this file owns the simulation side: env setup,
episode rollout, success filtering, and streaming episodes into one Lance
table (dataset_writer.py) that loads through swm.data.load_dataset like the
released pusht_expert_train.lance.

Only episodes that actually reach the destination are written, so every
recorded trajectory is a successful demo. The default episode count matches
the released expert dataset (18,685 episodes).

Example:
    python additional_files/pusht_random_dest/collect.py \\
        --name pusht_scripted_goal_train --episodes 18685 --num-workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import imageio
import numpy as np
import pymunk
from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.envs.pusht import env as pusht_env
from stable_worldmodel.envs.pusht.env import PushT


class ShapesOnlyDraw(pusht_env.DrawOptions):
    """pymunk's debug_draw stamps red contact-point markers wherever bodies
    touch (agent against block, block against wall); recorded frames should
    show the scene's shapes only."""

    def __init__(self, surface):
        super().__init__(surface)
        self.flags &= ~pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS


pusht_env.DrawOptions = ShapesOnlyDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_writer import build_episode_columns, write_dataset
from policy import ScriptedPushExpert, angle_diff, signed_angle_diff, unit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="pusht_scripted_goal_train")
    p.add_argument("--episodes", type=int, default=18685,
                   help="Successful episodes to write; default matches pusht_expert_train.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--max-steps", type=int, default=400, help="Control steps allowed to reach the goal.")
    p.add_argument("--min-frames", type=int, default=40,
                   help="Pad episodes to at least this many frames by holding after the goal is reached.")
    p.add_argument("--block-start", type=float, nargs=2, metavar=("X", "Y"), default=(256.0, 256.0))
    p.add_argument("--goal-bounds", type=float, nargs=2, metavar=("LO", "HI"), default=(140.0, 372.0),
                   help="Sampling range for the destination center; keeps the T inside the walls.")
    p.add_argument("--random-goal-angle", action=argparse.BooleanOptionalAction, default=True,
                   help="Randomize destination orientation (the published dataset does). The expert "
                        "reorients the block by pushing its stem tip sideways (spins in place) then "
                        "translates it onto the goal; --no-random-goal-angle keeps goals upright.")
    p.add_argument("--pos-tol", type=float, default=8.0, help="Success: block-to-goal position (px).")
    p.add_argument("--ang-tol", type=float, default=np.pi / 36, help="Success: block-to-goal angle (rad, ~5deg).")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--num-videos", type=int, default=0)
    p.add_argument("--video-dir", default=None)
    p.add_argument("--num-workers", type=int, default=1,
                   help="Parallel collection processes (episodes are independent). Workers stream "
                        "accepted episodes to a single Lance writer, so the table is written once.")
    # Controller shaping.
    p.add_argument("--max-action", type=float, default=0.32,
                   help="Per-step action magnitude cap in [0,1]. Smaller = smoother, slower pusher.")
    p.add_argument("--ring-margin", type=float, default=14.0,
                   help="Extra clearance (px) of the orbit ring beyond the T's reach + pusher radius.")
    return p.parse_args()


def get_state(e: PushT) -> np.ndarray:
    return np.array([*e.agent.position, *e.block.position, e.block.angle, *e.agent.velocity], dtype=np.float64)


def make_env(args) -> PushT:
    env = PushT(resolution=args.resolution, with_target=True)
    gp = env.variation_space["goal"]["position"]
    gp.low[:] = args.goal_bounds[0]
    gp.high[:] = args.goal_bounds[1]
    if not args.random_goal_angle:
        env.variation_space["goal"]["angle"].set_init_value(0.0)
    return env


def reset_options(args) -> dict:
    variation = ["goal.position", "agent.start_position"]
    if args.random_goal_angle:
        variation.append("goal.angle")
    return {"variation": variation,
            "variation_values": {"block.start_position": np.asarray(args.block_start, dtype=np.float64)}}


def place_agent_clear(env, expert, rng) -> None:
    """Move the pusher off the block if the random start landed on/inside the T.

    The env can sample the agent start overlapping the block; starting embedded
    makes the first push shove the block the wrong way. Keep the sampled bearing
    for start diversity, just push it out to the safe ring.
    """
    com = np.array(env.block.local_to_world(env.block.center_of_gravity))
    agent = np.array(env.agent.position)
    if np.linalg.norm(agent - com) >= expert.r_ring:
        return
    bearing = agent - com
    if np.linalg.norm(bearing) < 1e-6:
        theta = float(rng.uniform(0, 2 * np.pi))
        bearing = np.array([np.cos(theta), np.sin(theta)])
    env.agent.position = tuple(com + unit(bearing) * expert.r_ring)
    env.agent.velocity = (0.0, 0.0)


def run_episode(env, args, rng):
    env.reset(seed=int(rng.integers(1 << 31)), options=reset_options(args))
    expert = ScriptedPushExpert(env, max_action=args.max_action,
                                ring_margin=args.ring_margin, ang_tol=args.ang_tol)
    place_agent_clear(env, expert, rng)
    goal = np.array(env.goal_pose)
    states = [get_state(env)]
    actions = []
    frames = [env.render()]

    def step(action):
        env.step(action)
        actions.append(action.astype(np.float32))
        states.append(get_state(env))
        frames.append(env.render())

    def at_goal() -> bool:
        bp = np.array(env.block.position)
        return np.linalg.norm(bp - goal[:2]) < args.pos_tol and angle_diff(env.block.angle, goal[2]) < args.ang_tol

    ok = False
    for _ in range(args.max_steps):
        step(expert.act(np.array(env.agent.position)))
        if at_goal():
            ok = True
            break
    # Hold on the goal (agent still => quasi-static block stays) to pad short episodes.
    while ok and len(frames) < args.min_frames:
        step(np.zeros(2, dtype=np.float32))
    # env.goal_pose angle is the raw sampled value (can span +/-2pi); store it
    # wrapped to (-pi, pi] so the conditioning input is bounded and matches the
    # block-angle convention. Same physical orientation.
    goal = goal.astype(np.float32)
    goal[2] = signed_angle_diff(goal[2], 0.0)
    return (
        np.asarray(states),
        np.asarray(actions, dtype=np.float32),
        np.asarray(frames, dtype=np.uint8),
        goal,
        ok,
    )


def draw_bar(done: int, total: int, start: float, width: int = 32) -> None:
    """Render a single-line progress bar with elapsed time and ETA (in place)."""
    frac = done / total if total else 1.0
    elapsed = time.time() - start
    eta = elapsed / done * (total - done) if done else 0.0
    fill = int(width * frac)
    sys.stdout.write(f"\r[{'#' * fill}{'.' * (width - fill)}] {done}/{total} eps  "
                     f"{elapsed:5.0f}s elapsed  ETA {eta:5.0f}s  ")
    sys.stdout.flush()


def worker_loop(args, n_eps, seed, out_q, n_videos, video_dir):
    """Roll episodes in one process; push each accepted episode onto out_q."""
    rng = np.random.default_rng(seed)
    env = make_env(args)
    accepted = 0
    while accepted < n_eps:
        states, actions, frames, goal, ok = run_episode(env, args, rng)
        if not ok:
            continue
        if accepted < n_videos and video_dir is not None:
            imageio.mimsave(Path(video_dir) / f"{args.name}_ep{accepted:02d}.mp4",
                            list(frames), fps=10, codec="libx264")
        out_q.put(build_episode_columns(states, actions, frames, goal))
        accepted += 1
    env.close()
    out_q.put(None)  # this worker is done


def episode_stream(args, video_dir):
    """Yield accepted episodes from worker processes, drawing the progress bar.

    Workers stream episodes through a queue into this single consumer, so the
    Lance table is written exactly once (no shard merge, no JPEG re-encode).
    """
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=4 * max(args.num_workers, 1))
    workers = max(1, min(args.num_workers, args.episodes))
    base, rem = divmod(args.episodes, workers)
    counts = [base + (1 if i < rem else 0) for i in range(workers)]
    procs = [
        ctx.Process(
            target=worker_loop,
            args=(args, counts[i], args.seed + i * 100003, out_q,
                  args.num_videos if i == 0 else 0, video_dir),
        )
        for i in range(workers) if counts[i] > 0
    ]
    for p in procs:
        p.start()
    done_workers = 0
    yielded = 0
    start = time.time()
    try:
        while done_workers < len(procs):
            item = out_q.get()
            if item is None:
                done_workers += 1
                continue
            yielded += 1
            draw_bar(yielded, args.episodes, start)
            yield item
    finally:
        for p in procs:
            p.join()
        sys.stdout.write("\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.cache_dir or get_cache_dir(sub_folder="datasets"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.name}.lance"

    video_dir = None
    if args.num_videos > 0:
        video_dir = Path(args.video_dir or (out_dir / f"{args.name}_videos"))
        video_dir.mkdir(parents=True, exist_ok=True)
    vd = str(video_dir) if video_dir else None

    kind = "random-angle+placement" if args.random_goal_angle else "placement (upright)"
    print(f"Collecting {args.episodes} scripted demos [{kind}] -> {path}", flush=True)

    write_dataset(path, episode_stream(args, vd))
    print(f"Done -> {path}")
    if video_dir is not None:
        print(f"Sample videos -> {video_dir}")


if __name__ == "__main__":
    main()
