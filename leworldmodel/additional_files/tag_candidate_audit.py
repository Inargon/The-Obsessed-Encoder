"""Episode-separated sampling and fixed-candidate rank diagnostics."""

import numpy as np

REFERENCE = "reference"
TAG_MODES = ("context_tag", "goal_tag", "both_tags")
CONDITIONS = (REFERENCE, *TAG_MODES, "action_sequence")


def sample_episode_bank(dataset, rng, queries, candidates, num_steps, frameskip, history):
    """Choose one complete window per episode and a disjoint, fixed future bank."""
    clips = np.asarray(dataset.clip_indices, dtype=np.int64)
    offsets = np.asarray(dataset.offsets, dtype=np.int64)
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    if clips.shape != (len(dataset), 2):
        raise ValueError("Expected dataset clip indices (episode position, start)")
    ep_col = next(
        (key for key in ("episode_idx", "ep_idx") if key in dataset.column_names), None
    )
    if ep_col is None:
        raise ValueError("No physical episode ID column in this dataset")
    episode_ids = np.asarray(dataset.get_col_data(ep_col)).reshape(-1)
    steps = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    raw_actions = np.asarray(dataset.get_col_data("action"))

    windows = {}
    span = num_steps * frameskip
    for index, (episode, start) in enumerate(clips):
        if 0 <= start and start + span <= lengths[episode]:
            windows.setdefault(int(episode), []).append((index, int(start)))

    selected = []
    bank_futures = set()
    seen_episodes = set()
    for episode in rng.permutation(list(windows)):
        options = windows[int(episode)]
        for option in rng.permutation(len(options)):
            index, start = options[option]
            row = int(offsets[episode]) + start
            end = row + span
            physical_id = int(episode_ids[row])
            if physical_id in seen_episodes:
                break
            if end > len(raw_actions) or not np.all(episode_ids[row:end] == physical_id):
                raise ValueError(f"Clip {index} crosses a physical episode boundary")
            if not np.array_equal(steps[row:end], steps[row] + np.arange(span)):
                raise ValueError(f"Clip {index} contains nonconsecutive raw steps")
            source = raw_actions[row:end]
            if not np.isfinite(source).all():
                continue
            action = np.asarray(dataset[index]["action"])
            if action.size != source.size or not np.array_equal(
                action.reshape(source.shape), source.astype(action.dtype, copy=False)
            ):
                raise ValueError(f"Native action window differs from raw rows for clip {index}")
            if len(selected) >= queries:
                # Match the rollout: keep history-1 past blocks and use the next
                # num_steps-history blocks, ending at the recorded goal frame.
                first = (history - 1) * frameskip
                last = (num_steps - 1) * frameskip
                future = tuple(action.reshape(span, -1)[first:last].ravel())
                if future in bank_futures:
                    continue
                bank_futures.add(future)
            selected.append({"clip": index, "episode": physical_id, "start": int(steps[row])})
            seen_episodes.add(physical_id)
            break
        if len(selected) == queries + candidates:
            return selected[:queries], selected[queries:]
    raise ValueError(
        f"Need {queries} query episodes and {candidates} disjoint episodes with distinct "
        f"candidate plans, found {len(selected)} valid episodes"
    )


def candidate_gaps(costs):
    """Compare changed costs against the reference winner and retain exact ties."""
    arrays = {name: np.asarray(costs[name], dtype=np.float64) for name in CONDITIONS}
    ref = arrays[REFERENCE]
    if ref.ndim != 2 or min(ref.shape) < 2:
        raise ValueError("Expected at least two matched clips and candidates")
    if any(value.shape != ref.shape or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Candidate costs must be finite and aligned")
    rows = np.arange(len(ref))
    winner = ref.argmin(axis=1)
    rivals = np.arange(ref.shape[1])[None, :] != winner[:, None]
    scale = ref.std(axis=1)
    raw_gaps = {name: value - value[rows, winner, None] for name, value in arrays.items()}
    ref_gap = np.min(np.where(rivals, raw_gaps[REFERENCE], np.inf), axis=1)
    normalized = np.divide(ref_gap, scale, out=np.zeros_like(ref_gap), where=scale > 0)
    worst_tag_gap = np.minimum.reduce([raw_gaps[name] for name in TAG_MODES])
    rival = np.where(rivals, worst_tag_gap, np.inf).argmin(axis=1)
    rival_gaps = {name: raw_gaps[name][rows, rival].tolist() for name in (REFERENCE, *TAG_MODES)}
    interaction = (
        raw_gaps["both_tags"][rows, rival] - raw_gaps["context_tag"][rows, rival]
        - raw_gaps["goal_tag"][rows, rival] + raw_gaps[REFERENCE][rows, rival]
    )
    changed = {}
    for name in (*TAG_MODES, "action_sequence"):
        value = arrays[name]
        selected = value.argmin(axis=1)
        changed[name] = {
            "winner": selected.tolist(),
            "index_flip": (selected != winner).tolist(),
            "strict_flip": (raw_gaps[name][rows, selected] < 0).tolist(),
            "winner_reference_rank": (1 + (ref < ref[rows, selected, None]).sum(axis=1)).tolist(),
        }
    any_flip = np.logical_or.reduce([changed[name]["strict_flip"] for name in TAG_MODES])
    return {
        "reference_winner": winner.tolist(),
        "reference_gap": ref_gap.tolist(),
        "reference_cost_std": scale.tolist(),
        "normalized_reference_gap": normalized.tolist(),
        "strict_any_tag_flip": any_flip.tolist(),
        "conditions": changed,
        "worst_tag_rival": rival.tolist(),
        "worst_tag_rival_gaps": rival_gaps,
        "worst_tag_rival_interaction": interaction.tolist(),
    }


def frozen_anchor_gaps(model, own, changed, own_actions, negative_actions, history):
    """Replay the training expert-vs-batch-roll pair with frozen features."""
    import torch

    frames = torch.stack((own[:history + 1], changed[:history + 1]))
    actions = torch.stack((own_actions[:history + 1], negative_actions[:history + 1]))
    encoded = model.encode({"pixels": frames, "action": actions.to(frames.device)})
    emb = encoded["emb"]
    contexts = emb[:, :history]
    action = encoded["act_emb"][:, :history]
    pos = model.predict(contexts, action[:1].expand_as(action)).float()
    neg = model.predict(contexts, action[1:].expand_as(action)).float()
    targets = emb[:, 1:history + 1].float()
    gaps = {}
    for name, inputs, target in (
        (REFERENCE, 0, 0), ("context_tag", 1, 0), ("goal_tag", 0, 1), ("both_tags", 1, 1),
    ):
        gaps[name] = (
            (neg[inputs] - targets[target]).square().mean(-1)
            - (pos[inputs] - targets[target]).square().mean(-1)
        ).tolist()
    return gaps


def summarize_candidate_audit(audit, seed, margin=0.05, margin_fraction=0.25):
    """Summarize episode-level effects conditional on the shared candidate bank."""
    runs = {}
    baseline = np.asarray(audit["both"]["rank"]["strict_any_tag_flip"], dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(baseline), size=(2000, len(baseline)))
    for name, run in audit.items():
        rank = run["rank"]
        flips = np.asarray(rank["strict_any_tag_flip"], dtype=bool)
        gap = np.asarray(rank["normalized_reference_gap"])
        median = np.median(gap)
        small = gap <= median
        eligible = np.asarray(run["anchor"][REFERENCE]) >= margin
        deficits = {
            mode: eligible & (np.asarray(run["anchor"][mode])
                              < margin_fraction * np.asarray(run["anchor"][REFERENCE]))
            for mode in TAG_MODES
        }
        # A clip is covered only when the diagnostic finds a guard violation
        # in a condition that also displaced the original planning winner.
        covered_flip = np.logical_or.reduce([
            deficits[mode].any(axis=1) & rank["conditions"][mode]["strict_flip"]
            for mode in TAG_MODES
        ])
        delta = flips.astype(float) - baseline
        runs[name] = {
            "strict_any_tag_flip": int(flips.sum()),
            "strict_action_order_flip": sum(rank["conditions"]["action_sequence"]["strict_flip"]),
            "reference_ties": int(np.count_nonzero(np.asarray(rank["reference_gap"]) == 0)),
            "normalized_gap_median": float(median),
            "any_tag_flip_small_gap": [int((flips & small).sum()), int(small.sum())],
            "any_tag_flip_large_gap": [int((flips & ~small).sum()), int((~small).sum())],
            "frozen_anchor_eligible_fraction": float(eligible.mean()),
            "any_tag_flips_with_same_condition_anchor_deficit": int(covered_flip.sum()),
            "paired_any_tag_flip_vs_both": {
                "difference": float(delta.mean()),
                "bootstrap_episode_ci95": np.quantile(delta[sample].mean(1), [0.025, 0.975]).tolist(),
            },
            "by_tag_mode": {
                mode: {
                    "strict_flips": sum(rank["conditions"][mode]["strict_flip"]),
                    "strict_flips_by_reference_top_four": int(np.count_nonzero(
                        np.asarray(rank["conditions"][mode]["strict_flip"])
                        & (np.asarray(rank["conditions"][mode]["winner_reference_rank"]) <= 4)
                    )),
                }
                for mode in TAG_MODES
            },
        }
    return runs
