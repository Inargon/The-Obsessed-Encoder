#!/usr/bin/env python3
"""Build same-anchor branches with pusher-matched/block-divergent hard pairs."""

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


def wrap_angle(delta):
    return np.abs((delta + np.pi) % (2 * np.pi) - np.pi)


def physical_features(states, anchor):
    delta = states[:, :4] - anchor[None, :4]
    pos = np.concatenate((delta[:, :2] / 128.0, delta[:, 2:4] / 32.0), axis=-1)
    angle = np.stack((np.sin(states[:, 4]) - np.sin(anchor[4]),
                      np.cos(states[:, 4]) - np.cos(anchor[4])), axis=-1)
    return np.concatenate((pos, angle), axis=-1)


def select_branches(finals, anchor, count, hard_pairs, rng):
    candidates = []
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            pusher = np.linalg.norm(finals[i, :2] - finals[j, :2])
            block = (np.linalg.norm(finals[i, 2:4] - finals[j, 2:4])
                     + 50.0 * wrap_angle(finals[i, 4] - finals[j, 4]))
            if pusher <= 15.0 and block >= 15.0:
                candidates.append((block, i, j))
    candidates.sort(reverse=True)
    pairs, used = [], set()
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        pairs.append((i, j))
        used.update((i, j))
        if len(pairs) == hard_pairs:
            break
    if len(pairs) < hard_pairs:
        return None, None
    selected = [index for pair in pairs for index in pair]
    features = physical_features(finals, anchor)
    remaining = [i for i in range(len(finals)) if i not in used]
    while len(selected) < count:
        if not remaining:
            return None, None
        chosen_features = features[selected]
        distance = np.stack([
            np.linalg.norm(features[i] - chosen_features, axis=1).min()
            for i in remaining
        ])
        distance += rng.uniform(0, 1e-9, size=len(distance))
        pos = int(distance.argmax())
        selected.append(remaining.pop(pos))
    partner = np.full(count, -1, dtype=np.int64)
    for p in range(hard_pairs):
        partner[2 * p], partner[2 * p + 1] = 2 * p + 1, 2 * p
    return np.asarray(selected), partner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/action-binding-bank/bank.npz")
    ap.add_argument("--anchors", type=int, default=128)
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--branches", type=int, default=8)
    ap.add_argument("--hard-pairs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=59)
    args = ap.parse_args()
    if 2 * args.hard_pairs > args.branches or args.branches > args.candidates:
        ap.error("need 2*hard_pairs <= branches <= candidates")
    rng = np.random.default_rng(args.seed)
    ds = swm.data.load_dataset(
        "pusht_expert_train.h5", cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        num_steps=HS + HORIZON, frameskip=FRAMESKIP,
        keys_to_cache=["action", "state"],
    )
    span = np.asarray(ds[0]["pixels"]).shape[0]
    stride = 1 if span == HS + HORIZON else FRAMESKIP
    bank_rows = rng.choice(len(ds), size=args.candidates, replace=False)
    action_bank = np.stack([
        np.asarray(ds[int(row)]["action"])[:HORIZON] for row in bank_rows
    ]).astype(np.float32)
    raw_bank = action_bank.reshape(args.candidates, HORIZON * FRAMESKIP, -1)
    raw_actions = np.asarray(ds.get_col_data("action"), dtype=np.float32)
    raw_actions = raw_actions[~np.isnan(raw_actions).any(1)]
    repeats = action_bank.shape[-1] // raw_actions.shape[-1]
    action_mean = np.tile(raw_actions.mean(0), repeats).astype(np.float32)
    action_std = np.tile(raw_actions.std(0, ddof=1), repeats).astype(np.float32)

    world = swm.World(env_name="swm/PushT-v1", num_envs=1,
                      max_episode_steps=10_000, image_shape=(224, 224))
    env = world.envs.envs[0].unwrapped
    kept = []
    for row in rng.permutation(len(ds)):
        d = ds[int(row)]
        states = np.asarray(d["state"])[::stride][:HS + HORIZON]
        anchor = np.asarray(states[HS - 1], dtype=np.float64)
        if np.linalg.norm(anchor[:2] - anchor[2:4]) > 90.0:
            continue
        env.reset(seed=0, options={"state": anchor, "goal_state": anchor})
        finals, frames = [], []
        for raw_actions_for_branch in raw_bank:
            env._set_state(anchor)
            for action in raw_actions_for_branch:
                env.step(action)
            finals.append(np.asarray(env._get_obs(), dtype=np.float32))
            frames.append(to_hwc(env.render()))
        finals, frames = np.stack(finals), np.stack(frames)
        selected, partner = select_branches(
            finals, anchor, args.branches, args.hard_pairs, rng
        )
        if selected is None:
            continue
        kept.append({
            "history_frames": np.stack([
                to_hwc(frame) for frame in np.asarray(d["pixels"])[::stride][:HS]
            ]),
            "history_actions": np.asarray(d["action"])[:HS].astype(np.float32),
            "anchor_state": anchor.astype(np.float32),
            "branch_frames": frames[selected],
            "branch_states": finals[selected],
            "branch_actions": action_bank[selected],
            "hard_partner": partner,
        })
        if len(kept) % 8 == 0:
            print(f"collected {len(kept)}/{args.anchors} hard clouds", flush=True)
        if len(kept) == args.anchors:
            break
    world.close()
    if len(kept) < args.anchors:
        raise RuntimeError(f"only found {len(kept)} qualifying anchors")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        history_frames=np.stack([x["history_frames"] for x in kept]),
        history_actions=np.stack([x["history_actions"] for x in kept]),
        anchor_states=np.stack([x["anchor_state"] for x in kept]),
        branch_frames=np.stack([x["branch_frames"] for x in kept]),
        branch_states=np.stack([x["branch_states"] for x in kept]),
        branch_actions=np.stack([x["branch_actions"] for x in kept]),
        hard_partner=np.stack([x["hard_partner"] for x in kept]),
        action_mean=action_mean,
        action_std=action_std,
    )
    print(f"saved {len(kept)} x {args.branches} action-binding bank -> {out}")


if __name__ == "__main__":
    main()
