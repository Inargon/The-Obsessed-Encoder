#!/usr/bin/env python3
"""Create matched nuisance-intervention figures and GIFs for tagged PushT.

Every comparison holds the physical video, goal, and candidate action bank
fixed.  Only the small RGB tag changes.  The output is deliberately visual:
one aggregate figure plus paired storyboards/GIFs for examples where a
baseline changes its decision more than the repaired model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MODEL_ORDER = ("jepa", "full", "cycle", "bloop")
DISPLAY_NAMES = {
    "jepa": "JEPA",
    "full": "Full",
    "cycle": "Cycle",
    "bloop": "EMA orthogonal repair",
}


def normalized_cost_rmse(reference, changed):
    import numpy as np

    reference = np.asarray(reference, dtype=float)
    changed = np.asarray(changed, dtype=float)
    if reference.shape != changed.shape or reference.ndim != 2:
        raise ValueError("cost arrays must be matched (clips, candidates)")
    scale = reference.std(axis=1)
    return np.sqrt(np.mean((reference - changed) ** 2, axis=1)) / np.maximum(
        scale, 1e-12
    )


def select_examples(runs, count):
    """Prefer JEPA decision flips repaired by Bloop, then largest RMSE gap."""
    import numpy as np

    j_ref = np.asarray(runs["jepa"]["reference_costs"])
    j_alt = np.asarray(runs["jepa"]["changed_costs"])
    b_ref = np.asarray(runs["bloop"]["reference_costs"])
    b_alt = np.asarray(runs["bloop"]["changed_costs"])
    j_flip = j_ref.argmin(1) != j_alt.argmin(1)
    b_flip = b_ref.argmin(1) != b_alt.argmin(1)
    gap = normalized_cost_rmse(j_ref, j_alt) - normalized_cost_rmse(b_ref, b_alt)
    priority = (j_flip & ~b_flip).astype(float) * 1_000_000 + gap
    return np.argsort(-priority, kind="stable")[:count].astype(int).tolist()


def add_tag(raw, color, size):
    tagged = raw.clone()
    tagged[:, :, :size, :size] = color.view(1, 3, 1, 1)
    return tagged


def to_rgb(frame):
    import numpy as np

    value = frame.detach().cpu().numpy()
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    return np.asarray(value, dtype=np.uint8)


def render_assets(out_dir, pixels, colors, runs, selected, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [DISPLAY_NAMES[name] for name in MODEL_ORDER]
    x = np.arange(len(labels))
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    flips = [100 * summary[name]["selected_candidate_changed"] for name in MODEL_ORDER]
    reversals = [100 * summary[name]["pair_order_reversal"] for name in MODEL_ORDER]
    rmse = [100 * summary[name]["cost_rmse_over_reference_std"] for name in MODEL_ORDER]
    axes[0].bar(x - width / 2, flips, width, label="Selected candidate changed")
    axes[0].bar(x + width / 2, reversals, width, label="Pair-order reversal")
    axes[0].set_ylabel("Decision sensitivity (%)")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].legend(frameon=False)
    axes[1].bar(x, rmse)
    axes[1].set_ylabel("Cost RMSE / reference std (%)")
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Same scene, goal, and candidates; only the RGB tag changes")
    figure.savefig(out_dir / "counterfactual-summary.png", dpi=220)
    plt.close(figure)

    font = ImageFont.load_default()
    for rank, clip_index in enumerate(selected, start=1):
        raw = pixels[clip_index]
        phases = []
        for phase, color_index in (("REFERENCE TAG", 0), ("CHANGED TAG", 1)):
            tagged = add_tag(raw, colors[clip_index, color_index], summary["tag_size"])
            context = Image.fromarray(to_rgb(tagged[0])).resize((280, 280))
            goal = Image.fromarray(to_rgb(tagged[-1])).resize((280, 280))
            canvas = Image.new("RGB", (760, 430), "white")
            canvas.paste(context, (15, 45))
            canvas.paste(goal, (305, 45))
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 15), f"{phase} | matched clip {clip_index}", fill="black", font=font)
            draw.text((15, 330), "context", fill="black", font=font)
            draw.text((305, 330), "goal", fill="black", font=font)
            y = 65
            for name in MODEL_ORDER:
                reference = np.asarray(runs[name]["reference_costs"])[clip_index]
                changed = np.asarray(runs[name]["changed_costs"])[clip_index]
                ref_choice = int(reference.argmin())
                alt_choice = int(changed.argmin())
                current = ref_choice if color_index == 0 else alt_choice
                stable = "stable" if ref_choice == alt_choice else "CHANGED"
                draw.text(
                    (605, y),
                    f"{DISPLAY_NAMES[name]}\nchoice {current}\n{ref_choice}->{alt_choice} {stable}",
                    fill="black",
                    font=font,
                    spacing=4,
                )
                y += 78
            phases.append(canvas)
        stem = f"counterfactual-{rank:02d}-clip-{clip_index}"
        phases[0].save(out_dir / f"{stem}.png")
        phases[0].save(
            out_dir / f"{stem}.gif",
            save_all=True,
            append_images=phases[1:],
            duration=1100,
            loop=0,
            optimize=False,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--num-clips", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--tag-size", type=int, default=5)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")

    import numpy as np
    import torch
    import stable_worldmodel as swm
    from additional_files.diagnose_frozen_representation import parse_checkpoint
    from additional_files.diagnose_tag_intervention import choose_finite_action_indices
    from additional_files.evaluate_reacher_checkpoint import load_inference_model
    from additional_files.mechanism_metrics import cost_changes, normalize_action_blocks
    from utils import get_column_normalizer, get_img_preprocessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dataset = swm.data.load_dataset(
        args.dataset,
        cache_dir=os.environ["LOCAL_DATASET_DIR"],
        num_steps=args.history + args.horizon,
        frameskip=args.frameskip,
        keys_to_cache=["action"],
    )
    dataset.transform = None
    requested = min(args.num_clips, len(dataset))
    ids = choose_finite_action_indices(dataset, rng.permutation(len(dataset)), requested)
    normalizer = get_column_normalizer(dataset, "action", "action")
    action_dim = int(dataset.get_dim("action"))
    preprocess = get_img_preprocessor("pixels", "pixels", img_size=224)
    pixels, actions = [], []
    for index in ids:
        sample = dataset[index]
        raw = torch.as_tensor(sample["pixels"]).clone()
        pixels.append(raw)
        action = normalize_action_blocks(torch.as_tensor(sample["action"]), normalizer, action_dim)
        actions.append(action.reshape(args.history + args.horizon, -1))
    actions = torch.stack(actions)
    colors = torch.as_tensor(rng.integers(0, 256, (len(ids), 2, 3)), dtype=torch.uint8)
    bank_ids = rng.integers(len(ids), size=(len(ids), args.candidates))
    root = Path(os.environ["STABLEWM_HOME"]) / "checkpoints"
    runs = {}

    def process(raw, color):
        return preprocess({"pixels": add_tag(raw, color, args.tag_size)})["pixels"].to(device)

    with torch.no_grad():
        for spec in args.checkpoint:
            label, run, filename = parse_checkpoint(spec)
            model = load_inference_model(run, filename).to(device).eval().requires_grad_(False)
            reference_costs, changed_costs = [], []
            for i, raw in enumerate(pixels):
                reference = process(raw, colors[i, 0])
                changed = process(raw, colors[i, 1])
                plans = actions[bank_ids[i], : args.history + args.horizon - 1].clone()
                plans[:, : args.history - 1] = actions[i, : args.history - 1]
                plans = plans.unsqueeze(0).to(device)
                costs = []
                for video in (reference, changed):
                    info = {
                        "pixels": video[: args.history][None, None].expand(1, args.candidates, -1, -1, -1, -1),
                        "goal": video[-1:][None, None].expand(1, args.candidates, -1, -1, -1, -1),
                        "action": plans,
                    }
                    costs.append(model.get_cost(info, plans.clone()).cpu().numpy()[0])
                reference_costs.append(costs[0])
                changed_costs.append(costs[1])
            runs[label] = {
                "run_name": run,
                "checkpoint": filename,
                "reference_costs": np.asarray(reference_costs).tolist(),
                "changed_costs": np.asarray(changed_costs).tolist(),
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    missing = [name for name in MODEL_ORDER if name not in runs]
    if missing:
        raise ValueError(f"Missing required checkpoint labels: {missing}")
    first_actions = actions[bank_ids, args.history - 1].numpy()
    summary = {"tag_size": args.tag_size}
    for name in MODEL_ORDER:
        changes = cost_changes(
            runs[name]["reference_costs"], runs[name]["changed_costs"], first_actions
        )
        summary[name] = {key: float(np.mean(value)) for key, value in changes.items()}
    selected = select_examples(runs, min(args.examples, len(ids)))
    render_assets(args.out_dir, pixels, colors, runs, selected, summary)
    payload = {
        "protocol": {
            "dataset": args.dataset,
            "seed": args.seed,
            "num_clips": len(ids),
            "candidates": args.candidates,
            "intervention": "RGB tag only; scene, goal, and candidate bank fixed",
            "selection": "prefer JEPA choice flips absent in repair; then JEPA-minus-repair cost RMSE gap",
            "warning": "illustrative examples are selected; aggregate figure uses every sampled clip",
            "device": str(device),
        },
        "dataset_indices": ids,
        "selected_examples": selected,
        "summary": summary,
        "runs": runs,
    }
    (args.out_dir / "counterfactual-storyboard.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"COUNTERFACTUAL_STORYBOARD_COMPLETE {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
