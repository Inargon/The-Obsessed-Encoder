#!/usr/bin/env python3
"""Generate same-anchor PushT counterfactual outcome clouds for geometry training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import stable_worldmodel as swm


HS, HORIZON, FRAMESKIP = 3, 3, 5


def to_hwc(frame):
    x = np.asarray(frame)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = x.transpose(1, 2, 0)
    return np.ascontiguousarray(x, dtype=np.uint8)


def features(states, anchor):
    delta = states[:, :4] - anchor[None, :4]
    # The explicit physical metric M prevents large/easy pusher displacement
    # from drowning out the weaker block/contact directions we want to retain.
    pos = np.concatenate((delta[:, :2] / 128.0, delta[:, 2:4] / 32.0), axis=-1)
    ang = np.stack((np.sin(states[:, 4]) - np.sin(anchor[4]),
                    np.cos(states[:, 4]) - np.cos(anchor[4])), axis=-1)
    return np.concatenate((pos, ang), axis=-1)


def farthest_points(x, count, rng):
    """Greedy k-centres keeps a physically diverse rather than arbitrary cloud."""
    first = int(np.linalg.norm(x - x.mean(0), axis=1).argmax())
    chosen = [first]
    distance = np.linalg.norm(x - x[first], axis=1)
    while len(chosen) < count:
        jitter = rng.uniform(0, 1e-9, size=len(distance))
        nxt = int(np.argmax(distance + jitter))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(x - x[nxt], axis=1))
    return np.asarray(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/effect-geometry-bank/bank.npz")
    ap.add_argument("--anchors", type=int, default=128)
    ap.add_argument("--candidates", type=int, default=48)
    ap.add_argument("--branches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()
    if args.branches > args.candidates:
        ap.error("--branches cannot exceed --candidates")

    rng = np.random.default_rng(args.seed)
    ds = swm.data.load_dataset(
        "pusht_expert_train.h5", cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        num_steps=HS + HORIZON, frameskip=FRAMESKIP,
        keys_to_cache=["action", "state"],
    )
    span = np.asarray(ds[0]["pixels"]).shape[0]
    stride = 1 if span == HS + HORIZON else FRAMESKIP
    candidate_rows = rng.choice(len(ds), size=args.candidates, replace=False)
    bank = np.stack([np.asarray(ds[int(i)]["action"])[:HORIZON]
                     for i in candidate_rows])
    raw = bank.reshape(args.candidates, HORIZON * FRAMESKIP, -1)

    world = swm.World(env_name="swm/PushT-v1", num_envs=1,
                      max_episode_steps=10_000, image_shape=(224, 224))
    env = world.envs.envs[0].unwrapped
    anchor_frames, anchor_states, branch_frames, branch_states = [], [], [], []
    rows = rng.permutation(len(ds))
    for row in rows:
        d = ds[int(row)]
        states = np.asarray(d["state"])[::stride][:HS + HORIZON]
        anchor = np.asarray(states[HS - 1], dtype=np.float64)
        # Contact-near anchors expose the weak block/contact directions of interest.
        if np.linalg.norm(anchor[:2] - anchor[2:4]) > 90.0:
            continue
        env.reset(seed=0, options={"state": anchor, "goal_state": anchor})
        finals, frames = [], []
        for actions in raw:
            env._set_state(anchor)
            for action in actions:
                env.step(action)
            finals.append(np.asarray(env._get_obs(), dtype=np.float32))
            frames.append(to_hwc(env.render()))
        finals = np.stack(finals)
        keep = farthest_points(features(finals, anchor), args.branches, rng)
        anchor_frame = to_hwc(np.asarray(d["pixels"])[::stride][HS - 1])
        # One episode-constant nuisance tag per cloud, matching the Enigma arm.
        colour = rng.integers(0, 256, size=3, dtype=np.uint8)
        anchor_frame[:5, :5] = colour
        selected_frames = np.stack(frames)[keep]
        selected_frames[:, :5, :5] = colour
        anchor_frames.append(anchor_frame)
        anchor_states.append(anchor.astype(np.float32))
        branch_frames.append(selected_frames)
        branch_states.append(finals[keep])
        if len(anchor_frames) % 16 == 0:
            print(f"collected {len(anchor_frames)}/{args.anchors} clouds", flush=True)
        if len(anchor_frames) == args.anchors:
            break
    world.close()
    if len(anchor_frames) < args.anchors:
        raise RuntimeError(f"only found {len(anchor_frames)} contact-near anchors")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        anchor_frames=np.stack(anchor_frames),
        anchor_states=np.stack(anchor_states),
        branch_frames=np.stack(branch_frames),
        branch_states=np.stack(branch_states),
    )
    print(f"saved {len(anchor_frames)} x {args.branches} outcome cloud -> {out}")


if __name__ == "__main__":
    main()
