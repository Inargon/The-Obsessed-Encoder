"""Run a frozen PushT scoring-head pilot with simulated physical action labels."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from additional_files.control_relevance_audit.oracle_closed_loop import goal_error
from additional_files.control_relevance_audit.goal_role_data import set_render_pose
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


def replay_candidates(env, initial, prefix, bank, target_index, seed, capture_endpoints=None):
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
        if index == target_index or capture_endpoints is not None:
            frame = np.asarray(env.render()).copy()
            np.testing.assert_array_equal(snapshot(env), endpoints[-1])
            if capture_endpoints is not None:
                capture_endpoints.append((endpoints[-1], frame))
            if index == target_index:
                target = frame

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


def independent_goal_frame(env, target_state, endpoints, seed):
    """Render the same block pose with the agent parked away from the goal."""
    env.reset(seed=seed, options={"state": target_state[:7].copy()})
    set_render_pose(env.block, env.space, target_state[2:5])
    corners = np.array([[64, 64], [64, 448], [448, 64], [448, 448]])
    corner_order = np.argsort(-np.square(corners - target_state[2:4]).sum(1), kind="stable")
    for index in corner_order:
        position = corners[index]
        env.agent.position = tuple(position)
        env.space.reindex_shapes_for_body(env.agent)
        state = snapshot(env)
        np.testing.assert_allclose(state[:2], position, rtol=0, atol=1e-6)
        np.testing.assert_allclose(state[2:5], target_state[2:5], rtol=0, atol=1e-6)
        frame = np.asarray(env.render()).copy()
        np.testing.assert_array_equal(snapshot(env), state)
        if all(not np.array_equal(frame, image) for _, image in endpoints):
            return frame, position.tolist()
    raise ValueError("Independent target render duplicates a candidate endpoint image")


def prepare_pixels(frames, rgb, preprocess):
    import torch

    pixels = frames.copy()
    pixels[:, :5, :5, :] = rgb
    tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2)
    return preprocess({"pixels": tensor})["pixels"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True, help="Previous tag-intervention.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--endpoint-replay-of", type=Path, help="Replay the held-out bank from a margin-pilot report")
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
        compare_holdout, cost_inputs, fit_heads, physical_pairs, summarize_endpoint_oracle,
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

    previous_pilot = json.loads(args.endpoint_replay_of.read_text()) if args.endpoint_replay_of else None
    if previous_pilot is None:
        rng = np.random.default_rng(args.seed)
        queries = TRAIN_QUERIES + TEST_QUERIES
        query, bank = sample_episode_bank(
            dataset, rng, queries, 2 * (CANDIDATES - 1), HISTORY + HORIZON, FRAMESKIP, HISTORY,
        )
        palettes = unique_palettes(rng, queries)
        target_indices = rng.integers(CANDIDATES, size=queries)
        bank_groups = (bank[:CANDIDATES - 1], bank[CANDIDATES - 1:])
        replay_offset = 0
    else:
        protocol = previous_pilot["protocol"]
        if sha != protocol["checkpoint_sha256"] or protocol["dataset"] != "pusht_expert_train.h5":
            raise ValueError("Pilot checkpoint or dataset does not match the current replay")
        split = protocol["split"]
        query, bank = split["heldout_queries"], split["heldout_bank"]
        queries = len(query)
        replay_offset = len(split["train_queries"])
        palettes = np.asarray(protocol["rgb_by_query"][replay_offset:], dtype=np.uint8)
        target_indices = np.asarray(protocol["target_candidate_by_query"][replay_offset:])
        if (
            queries == 0 or len(bank) != CANDIDATES - 1 or palettes.shape != (queries, 2, 3)
            or target_indices.shape != (queries,) or np.any((target_indices < 0) | (target_indices >= CANDIDATES))
        ):
            raise ValueError("Invalid held-out split, RGB values or recorded target indices")
        args.seed = protocol["seed"]
        bank_groups = (bank, bank)
    preprocess = get_img_preprocessor("pixels", "pixels", img_size=224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_inference_model(previous["run_name"], previous["checkpoint"]).to(device).eval().requires_grad_(False)
    env = PushT(resolution=224, with_target=False, relative=True)
    goal_env = PushT(resolution=224, with_target=False, relative=True) if previous_pilot is not None else None
    past_steps = (HISTORY - 1) * FRAMESKIP
    future_steps = HORIZON * FRAMESKIP

    def raw_row(item):
        ep, offset = dataset.clip_indices[item["clip"]]
        return int(dataset.offsets[ep]) + int(offset)

    def future(item):
        start = raw_row(item) + past_steps
        return actions[start:start + future_steps]

    banks = [np.stack([future(item) for item in group]) for group in bank_groups]
    all_pred, all_goal, all_distance, all_initial = [], [], [], []
    actual_endpoints, independent_goals, goal_agent_xy = [], [], []
    try:
        with torch.inference_mode():
            for index, item in enumerate(query):
                row = raw_row(item)
                start = states[row].copy()
                if not np.isfinite(start).all():
                    raise ValueError(f"Nonfinite initial state for episode {item['episode']}")
                prefix = np.clip(actions[row:row + past_steps], env.action_space.low, env.action_space.high)
                alternatives = banks[int(previous_pilot is not None or index >= TRAIN_QUERIES)]
                chosen = int(target_indices[index])
                candidates = np.concatenate((alternatives[:chosen], future(item)[None], alternatives[chosen:]))
                candidates = np.clip(candidates, env.action_space.low, env.action_space.high)
                capture = [] if previous_pilot is not None else None
                seed = args.seed + replay_offset + index
                frames, distance, initial_distance = replay_candidates(env, start, prefix, candidates, chosen, seed, capture)
                if len(frames) != HISTORY + 1 or frames.dtype != np.uint8 or frames.shape[-1] != 3:
                    raise ValueError("Expected real native RGB history plus one target frame")
                packed = np.concatenate((np.broadcast_to(prefix, (CANDIDATES, *prefix.shape)), candidates), axis=1)
                packed = torch.from_numpy(packed.reshape(CANDIDATES, HISTORY + HORIZON - 1, -1)).float()
                normalized = normalize_action_blocks(packed, normalizer, action_dim).to(device)
                if not torch.isfinite(normalized).all():
                    raise ValueError("Nonfinite normalized replay actions")
                views = torch.stack([prepare_pixels(frames, rgb, preprocess) for rgb in palettes[index]]).to(device)
                pred, goal = terminal_features(model, views, normalized.unsqueeze(0), HISTORY)
                if index == 0:
                    check_cost_parity(model, views, normalized.unsqueeze(0), HISTORY, pred, goal)
                if capture is not None:
                    old = previous_pilot["heldout_per_query"]
                    np.testing.assert_allclose(distance, old["candidate_distances"][index], rtol=0, atol=1e-5)
                    np.testing.assert_allclose(initial_distance, old["start_distance"][index], rtol=0, atol=1e-5)
                    independent, park_xy = independent_goal_frame(goal_env, capture[chosen][0], capture, seed)
                    terminal_frames = np.stack([frame for _, frame in capture])
                    terminal_and_goal = np.concatenate((terminal_frames, independent[None]), axis=0)
                    encoded = []
                    for rgb in palettes[index]:
                        inputs = prepare_pixels(terminal_and_goal, rgb, preprocess).to(device)
                        encoded.append(torch.stack([
                            model.encode({"pixels": image[None, None]})["emb"][0, 0] for image in inputs
                        ]))
                    encoded = torch.stack(encoded)
                    torch.testing.assert_close(encoded[:, chosen], goal, rtol=1e-4, atol=1e-5)
                    if index == 0:
                        independent_views = torch.stack([
                            prepare_pixels(np.concatenate((frames[:-1], independent[None])), rgb, preprocess)
                            for rgb in palettes[index]
                        ]).to(device)
                        check_cost_parity(
                            model, independent_views, normalized.unsqueeze(0), HISTORY, pred, encoded[:, -1],
                        )
                    actual_endpoints.append(encoded[:, :-1].float().cpu().numpy())
                    independent_goals.append(encoded[:, -1].float().cpu().numpy())
                    goal_agent_xy.append(park_xy)
                all_pred.append(pred.float().cpu().numpy())
                all_goal.append(goal.float().cpu().numpy())
                all_distance.append(distance)
                all_initial.append(initial_distance)
                if (index + 1) % 8 == 0:
                    print(f"PHYSICAL_MARGIN_COLLECTED {index + 1}/{queries}", flush=True)
    finally:
        env.close()
        if goal_env is not None:
            goal_env.close()

    distances = np.stack(all_distance)
    starts_outside = np.asarray(all_initial) > 1
    opportunity = starts_outside & (distances <= 1).any(1) & (distances >= 2).any(1)
    predicted = torch.from_numpy(np.stack(all_pred)).to(device)
    original_goals = torch.from_numpy(np.stack(all_goal)).to(device)
    if previous_pilot is not None:
        goals = torch.from_numpy(np.stack(independent_goals)).to(device)
        actual = torch.from_numpy(np.stack(actual_endpoints)).to(device)
        # Use the same scoring rule for predicted and simulated terminal vectors.
        variants = {
            name: cost_inputs(endpoints, target)[1].cpu().numpy()
            for name, (endpoints, target) in {
                "native": (predicted, goals),
                "simulated_endpoint": (actual, goals),
                "forecast_original_goal": (predicted, original_goals),
            }.items()
        }
        prior_choices = previous_pilot["heldout_per_query"]["selected_candidate"]["native"]
        report = {
            "protocol": {
                "checkpoint_sha256": sha, "seed": args.seed, "source_pilot": str(args.endpoint_replay_of),
                "heldout_queries": query, "heldout_bank": bank, "rgb_by_query": palettes.tolist(),
                "parked_goal_agent_xy": goal_agent_xy,
                "goal_image": "separately rendered block pose; pusher parked at a distant corner",
                "native": "forecast endpoints scored against the independent goal",
                "simulated_endpoint": "simulated endpoints scored against the same independent goal",
                "physical_distances_match_original_pilot": True,
                "native_cost_parity_checked": True,
                "original_goal_matches_recorded_endpoint_encoding": True,
                "forecast_original_goal_choice_agreement": float(
                    (variants["forecast_original_goal"].argmin(-1) == np.asarray(prior_choices)).mean()
                ),
            },
            "heldout": summarize_endpoint_oracle(variants, distances, opportunity, args.seed),
            "candidate_distances": distances.tolist(),
            "selected_candidate": {name: costs.argmin(-1).tolist() for name, costs in variants.items()},
        }
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(f"ENDPOINT_ORACLE_COMPLETE {args.out}", flush=True)
        return

    inputs, native, normalized = cost_inputs(predicted, original_goals)
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
