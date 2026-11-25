"""Run a frozen PushT scoring-head pilot with simulated physical action labels."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from additional_files.control_relevance_audit.oracle_closed_loop import goal_error
from additional_files.mechanism_metrics import normalize_action_blocks
from additional_files.tag_candidate_audit import sample_episode_bank
from additional_files.tag_paired_planning import check_cost_parity, terminal_features


HISTORY = 3
FRAMESKIP = 5
HORIZON = 5
CANDIDATES = 32
TRAIN_QUERIES = 40
TEST_QUERIES = 24
STEPS = 160


def snapshot(env):
    return np.array([
        *env.agent.position, *env.block.position, env.block.angle,
        *env.agent.velocity, *env.block.velocity, env.block.angular_velocity,
    ], dtype=float)


def endpoint_distance(state, goal):
    xy, angle = goal_error(state, goal)
    return max(xy / 10.0, angle / 0.1)


def replay_candidates(env, initial, prefix, bank, target_index, seed):
    """Reset and replay the same prefix before every candidate branch."""
    def restore(render_history):
        env.reset(seed=seed, options={"state": initial.copy()})
        states = [snapshot(env)]
        frames = [np.asarray(env.render()).copy()] if render_history else []
        for step, action in enumerate(prefix, start=1):
            env.step(action.copy())
            states.append(snapshot(env))
            if render_history and step % FRAMESKIP == 0:
                frames.append(np.asarray(env.render()).copy())
        return np.asarray(states), frames

    expected, history = restore(True)
    if not np.isfinite(expected).all():
        raise ValueError("Nonfinite shared prefix replay")
    endpoints = []
    for index, actions in enumerate(bank):
        actual, _ = restore(False)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5,
                                   err_msg="Candidate branches did not share the realized prefix")
        for action in actions:
            env.step(action.copy())
        endpoints.append(snapshot(env))
        if index == target_index:
            target = np.asarray(env.render()).copy()
            np.testing.assert_array_equal(snapshot(env), endpoints[-1])

    endpoints = np.asarray(endpoints)
    if not np.isfinite(endpoints).all():
        raise ValueError("Nonfinite candidate endpoint")
    goal_state = endpoints[target_index]
    distances = np.array([endpoint_distance(state, goal_state) for state in endpoints])
    if distances[target_index] != 0:
        raise ValueError("The recorded target branch must have zero endpoint error")
    return np.stack([*history, target]), distances, endpoint_distance(expected[-1], goal_state)


def unique_palettes(rng, queries):
    color_ids = rng.choice(256 ** 3, size=queries * 2, replace=False).reshape(queries, 2)
    return np.stack((color_ids >> 16, (color_ids >> 8) & 255, color_ids & 255), axis=-1).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True, help="Previous tag-intervention.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Expected a fresh report path")

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import torch
    import stable_worldmodel as swm
    from stable_worldmodel.envs.pusht.env import PushT
    from additional_files.evaluate_reacher_checkpoint import load_inference_model
    from additional_files.tag_margin_scoring import (
        compare_holdout, cost_inputs, fit_heads, physical_pairs,
    )
    from utils import get_column_normalizer, get_img_preprocessor

    previous = json.loads(args.previous.read_text())["runs"]["base"]
    checkpoint = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / previous["run_name"] / previous["checkpoint"]
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    if sha != previous["sha256"]:
        raise ValueError("Initialization checkpoint does not match the previous report")

    dataset = swm.data.load_dataset(
        "pusht_expert_train.h5", cache_dir=os.environ["LOCAL_DATASET_DIR"],
        num_steps=HISTORY + HORIZON, frameskip=FRAMESKIP, keys_to_cache=["action"],
    )
    dataset.transform = None
    normalizer = get_column_normalizer(dataset, "action", "action")
    action_dim = int(dataset.get_dim("action"))
    states = np.asarray(dataset.get_col_data("state"))
    actions = np.asarray(dataset.get_col_data("action"))
    if states.shape[1:] != (7,) or action_dim != 2 or actions.shape[1:] != (action_dim,):
        raise ValueError("Unexpected PushT state/action columns")

    rng = np.random.default_rng(args.seed)
    queries = TRAIN_QUERIES + TEST_QUERIES
    query, bank = sample_episode_bank(
        dataset, rng, queries, 2 * (CANDIDATES - 1), HISTORY + HORIZON, FRAMESKIP, HISTORY,
    )
    palettes = unique_palettes(rng, queries)
    target_indices = rng.integers(CANDIDATES, size=queries)
    preprocess = get_img_preprocessor("pixels", "pixels", img_size=224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_inference_model(previous["run_name"], previous["checkpoint"]).to(device).eval().requires_grad_(False)
    env = PushT(resolution=224, with_target=False, relative=True)
    past_steps = (HISTORY - 1) * FRAMESKIP
    future_steps = HORIZON * FRAMESKIP

    def raw_row(item):
        ep, offset = dataset.clip_indices[item["clip"]]
        return int(dataset.offsets[ep]) + int(offset)

    def future(item):
        start = raw_row(item) + past_steps
        return actions[start:start + future_steps]

    banks = np.stack([future(item) for item in bank]).reshape(2, CANDIDATES - 1, future_steps, action_dim)
    all_pred, all_goal, all_distance, all_initial = [], [], [], []
    try:
        with torch.inference_mode():
            for index, item in enumerate(query):
                row = raw_row(item)
                start = states[row].copy()
                if not np.isfinite(start).all():
                    raise ValueError(f"Nonfinite initial state for episode {item['episode']}")
                prefix = np.clip(actions[row:row + past_steps], env.action_space.low, env.action_space.high)
                alternatives = banks[int(index >= TRAIN_QUERIES)]
                chosen = int(target_indices[index])
                candidates = np.concatenate((alternatives[:chosen], future(item)[None], alternatives[chosen:]))
                candidates = np.clip(candidates, env.action_space.low, env.action_space.high)
                frames, distance, initial_distance = replay_candidates(
                    env, start, prefix, candidates, chosen, args.seed + index,
                )
                if len(frames) != HISTORY + 1 or frames.dtype != np.uint8 or frames.shape[-1] != 3:
                    raise ValueError("Expected real native RGB history plus one target frame")
                packed = np.concatenate((np.broadcast_to(prefix, (CANDIDATES, *prefix.shape)), candidates), axis=1)
                packed = torch.from_numpy(packed.reshape(CANDIDATES, HISTORY + HORIZON - 1, -1)).float()
                normalized = normalize_action_blocks(packed, normalizer, action_dim).to(device)
                if not torch.isfinite(normalized).all():
                    raise ValueError("Nonfinite normalized replay actions")
                views = []
                for rgb in palettes[index]:
                    pixels = frames.copy()
                    pixels[:, :5, :5, :] = rgb
                    tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2)
                    views.append(preprocess({"pixels": tensor})["pixels"])
                views = torch.stack(views).to(device)
                pred, goal = terminal_features(model, views, normalized.unsqueeze(0), HISTORY)
                if index == 0:
                    check_cost_parity(model, views, normalized.unsqueeze(0), HISTORY, pred, goal)
                all_pred.append(pred.float().cpu().numpy())
                all_goal.append(goal.float().cpu().numpy())
                all_distance.append(distance)
                all_initial.append(initial_distance)
                if (index + 1) % 8 == 0:
                    print(f"PHYSICAL_MARGIN_COLLECTED {index + 1}/{queries}", flush=True)
    finally:
        env.close()

    inputs, native, normalized = cost_inputs(
        torch.from_numpy(np.stack(all_pred)).to(device), torch.from_numpy(np.stack(all_goal)).to(device),
    )
    distances = np.stack(all_distance)
    starts_outside = np.asarray(all_initial) > 1
    opportunity = starts_outside & (distances <= 1).any(1) & (distances >= 2).any(1)
    train = slice(0, TRAIN_QUERIES)
    test = slice(TRAIN_QUERIES, queries)
    pairs = physical_pairs(
        normalized[train], torch.from_numpy(distances[train]).to(device),
        torch.from_numpy(starts_outside[train]).to(device),
    )
    print(f"PHYSICAL_MARGIN_PAIRS {len(pairs[0])} queries={int(opportunity[train].sum())}", flush=True)
    fitted = fit_heads(inputs[train], normalized[train], pairs, STEPS, args.seed) if len(pairs[0]) else {}
    variants = {"native": native[test].cpu().numpy()}
    with torch.inference_mode():
        for arm, head in fitted.items():
            variants[arm] = (normalized[test] + head(inputs[test]).squeeze(-1)).cpu().numpy()
    if fitted:
        torch.save({name: head.cpu().state_dict() for name, head in fitted.items()}, args.out.with_name("heads.pt"))

    report = {
        "protocol": {
            "checkpoint_sha256": sha,
            "seed": args.seed,
            "dataset": "pusht_expert_train.h5",
            "split": {
                "train_queries": query[train], "heldout_queries": query[test],
                "train_bank": bank[:CANDIDATES - 1], "heldout_bank": bank[CANDIDATES - 1:],
            },
            "heldout_scope": "episodes, action-bank sources and RGB held out from the scoring head",
            "frozen_checkpoint_training_holdout_verified": False,
            "rgb_by_query": palettes.tolist(),
            "target_candidate_by_query": target_indices.tolist(),
            "task": "hindsight block goal from replay of the logged continuation; fixed candidate bank",
            "physical_distance": "max of xy error / 10 pixels and wrapped angle error / 0.1 radians",
            "informative": "start outside tolerance and bank contains a hit (<=1) and a clear miss (>=2)",
            "pair_selection": "all endpoint hits vs the union of four lowest native-cost clear misses per tag view",
            "steps_per_arm": STEPS,
            "raw_planner_cost_parity_verified": True,
        },
        "training_informative_queries": int(opportunity[train].sum()),
        "training_pairs": len(pairs[0]),
        "status": "trained" if fitted else "no_eligible_training_pairs",
        "heldout": compare_holdout(variants, distances[test], opportunity[test], args.seed),
        "heldout_per_query": {
            "start_distance": all_initial[test],
            "candidate_distances": distances[test].tolist(),
            "selected_candidate": {name: value.argmin(-1).tolist() for name, value in variants.items()},
        },
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"PHYSICAL_MARGIN_PILOT_COMPLETE {args.out}", flush=True)


if __name__ == "__main__":
    main()
