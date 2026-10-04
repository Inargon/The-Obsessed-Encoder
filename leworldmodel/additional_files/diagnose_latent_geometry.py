#!/usr/bin/env python3
"""Paper-style latent interpolation diagnostics for tagged PushT.

Inspired by the Reacher analysis in ``When Does LeJEPA Learn a World
Model?``: interpolate linearly between fixed endpoint embeddings and decode
each point by nearest-neighbour retrieval from one shared image gallery.

This is an empirical geometry diagnostic, not a test of that paper's
Gaussian-world identifiability theorem.  Every model receives identical
endpoint pairs and the same gallery.  A second pass changes only the corner
tag at the endpoints to measure nuisance-induced path changes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


MODEL_ORDER = ("jepa", "full", "cycle", "bloop")
DISPLAY_NAMES = {
    "jepa": "JEPA",
    "full": "Full",
    "cycle": "Cycle",
    "bloop": "EMA orthogonal repair",
}
COLORS = {
    "oracle": "#777777",
    "jepa": "#d55e00",
    "full": "#0072b2",
    "cycle": "#009e73",
    "bloop": "#cc79a7",
}


def state_features(states: np.ndarray) -> np.ndarray:
    """PushT state with circular block angle represented continuously."""
    states = np.asarray(states, dtype=np.float64)
    if states.shape[-1] < 5:
        raise ValueError("PushT state must contain pusher xy, block xy, angle")
    angle = states[..., 4]
    return np.concatenate(
        [states[..., :4], np.sin(angle)[..., None], np.cos(angle)[..., None]],
        axis=-1,
    )


def straight_path(endpoints: np.ndarray, steps: int) -> np.ndarray:
    endpoints = np.asarray(endpoints)
    if endpoints.shape[0] != 2 or steps < 2:
        raise ValueError("expected two endpoints and at least two steps")
    alpha = np.linspace(0.0, 1.0, steps)
    return (1.0 - alpha[:, None]) * endpoints[0] + alpha[:, None] * endpoints[1]


def nearest_indices(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Euclidean 1-NN, matching the paper's qualitative retrieval figure."""
    queries = np.asarray(queries, dtype=np.float64)
    gallery = np.asarray(gallery, dtype=np.float64)
    distances = (
        np.sum(queries * queries, axis=1, keepdims=True)
        + np.sum(gallery * gallery, axis=1)[None]
        - 2.0 * queries @ gallery.T
    )
    return np.argmin(distances, axis=1)


def quadratic_path_cost(path: np.ndarray) -> float:
    """O(n)-invariant state-to-goal plus step-energy cost."""
    path = np.asarray(path, dtype=np.float64)
    goal = path[-1]
    state_cost = np.square(path - goal).sum()
    step_cost = np.square(np.diff(path, axis=0)).sum()
    return float(state_cost + step_cost)


def path_metrics(path: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    path = np.asarray(path, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    oracle_length = np.linalg.norm(np.diff(oracle, axis=0), axis=1).sum()
    path_length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    return {
        "state_rmse_to_oracle": float(np.sqrt(np.mean(np.square(path - oracle)))),
        "path_length_ratio": float(path_length / max(oracle_length, 1e-12)),
        "control_cost_ratio": float(
            quadratic_path_cost(path) / max(quadratic_path_cost(oracle), 1e-12)
        ),
    }


def bootstrap_mean_ci(values, seed: int, samples: int = 20_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), [float(lo), float(hi)]


def _to_rgb(frame) -> np.ndarray:
    value = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    if value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    return value


def _border(axis, color, width=3):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(width)


def render_plan_grid(
    out_dir, example_index, rows, oracle_frames, model_order, ghost_alpha=0.25
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    names = ["Physical oracle", *(DISPLAY_NAMES[name] for name in model_order)]
    row_keys = ["oracle", *model_order]
    steps = len(oracle_frames)
    figures = []
    for overlay in (False, True):
        fig, axes = plt.subplots(
            len(row_keys), steps, figsize=(1.72 * steps, 1.64 * len(row_keys))
        )
        for r, (key, title) in enumerate(zip(row_keys, names)):
            for c in range(steps):
                image = np.asarray(rows[key][c], dtype=np.float32)
                if overlay and key != "oracle":
                    image = np.clip(
                        (1.0 - ghost_alpha) * image
                        + ghost_alpha * np.asarray(oracle_frames[c], dtype=np.float32),
                        0,
                        255,
                    )
                axes[r, c].imshow(image.astype(np.uint8))
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                if c == 0:
                    axes[r, c].set_ylabel(title, rotation=0, ha="right", va="center")
                if c == 0:
                    _border(axes[r, c], "#22aa44")
                if c == steps - 1:
                    _border(axes[r, c], "#dd2222")
        axes[0, 0].set_title("Start", color="#159447", fontweight="bold")
        axes[0, -1].set_title("Goal", color="#cc2222", fontweight="bold")
        subtitle = "ghosted physical oracle reveals deviation" if overlay else "1-NN decoding from a shared gallery"
        fig.suptitle(f"Latent straight-line planning — pair {example_index:02d}\n{subtitle}")
        fig.tight_layout()
        suffix = "-oracle-overlay" if overlay else ""
        path = out_dir / f"latent-plan-{example_index:02d}{suffix}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        figures.append(path)
        plt.close(fig)

    # Animation is intentionally task-centric: each frame advances one point
    # along every displayed path rather than alternating two nearly identical tags.
    gif_frames = []
    for step in range(steps):
        tiles = []
        for key, title in zip(row_keys, names):
            image = Image.fromarray(np.asarray(rows[key][step], dtype=np.uint8)).convert("RGB")
            image = image.resize((224, 224))
            canvas = Image.new("RGB", (360, 252), "white")
            canvas.paste(image, (128, 22))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 104), title, fill="black")
            tiles.append(canvas)
        frame = Image.new("RGB", (360, 252 * len(tiles) + 42), "white")
        draw = ImageDraw.Draw(frame)
        draw.text((8, 10), f"Latent interpolation step {step + 1}/{steps}", fill="black")
        for r, tile in enumerate(tiles):
            frame.paste(tile, (0, 42 + 252 * r))
        gif_frames.append(frame)
    gif_frames[0].save(
        out_dir / f"latent-plan-{example_index:02d}.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=650,
        loop=0,
        optimize=False,
    )
    return figures


def render_state_paths(out_dir, example_index, paths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    for key, path in paths.items():
        label = "Physical oracle" if key == "oracle" else DISPLAY_NAMES[key]
        color = COLORS[key]
        axes[0].plot(path[:, 0], path[:, 1], "-o", ms=3, lw=1.8, label=label, color=color)
        axes[1].plot(path[:, 2], path[:, 3], "-o", ms=3, lw=1.8, label=label, color=color)
    axes[0].set_title("Pusher path")
    axes[1].set_title("Block-center path")
    for axis in axes:
        axis.set_xlabel("x (standardized)")
        axis.set_ylabel("y (standardized)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Decoded physical trajectories — pair {example_index:02d}")
    fig.savefig(out_dir / f"state-path-{example_index:02d}.png", dpi=220)
    plt.close(fig)


def render_embedding_pca(
    out_dir, gallery_embeddings, example_plans, gallery_state, model_order
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    columns = min(2, len(model_order))
    rows = int(np.ceil(len(model_order) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(5 * columns, 4.5 * rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).reshape(-1)
    color = np.arctan2(gallery_state[:, 4], gallery_state[:, 5])
    for axis, name in zip(axes, model_order):
        pca = PCA(n_components=2).fit(gallery_embeddings[name])
        gallery_2d = pca.transform(gallery_embeddings[name])
        path_2d = pca.transform(example_plans[name])
        axis.scatter(gallery_2d[:, 0], gallery_2d[:, 1], c=color, cmap="twilight", s=4, alpha=0.28)
        axis.plot(path_2d[:, 0], path_2d[:, 1], "-o", color=COLORS[name], lw=2, ms=4)
        axis.scatter(*path_2d[0], color="#22aa44", s=70, zorder=4)
        axis.scatter(*path_2d[-1], color="#dd2222", marker="*", s=110, zorder=4)
        axis.set_title(DISPLAY_NAMES[name])
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[len(model_order):]:
        axis.remove()
    fig.suptitle("Gallery geometry and the same latent straight-line plan\ncolor = physical block angle")
    fig.savefig(out_dir / "latent-gallery-pca.png", dpi=220)
    plt.close(fig)


def render_summaries(out_dir, summary, model_order):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("linear_state_r2", "Linear physical-state R²", False),
        ("state_rmse_to_oracle", "Physical RMSE to oracle", False),
        ("path_length_ratio", "Path length / oracle", False),
        ("control_cost_ratio", "Control cost / oracle", False),
        ("counterfactual_path_rmse", "Tag-only path displacement", False),
        ("counterfactual_retrieval_change_rate", "Retrieved frames changed (%)", True),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    x = np.arange(len(model_order))
    for axis, (metric, title, percent) in zip(axes.flat, metrics):
        if metric == "linear_state_r2":
            means = np.array([summary[name][metric] for name in model_order])
            cis = np.stack([means, means], axis=1)
        else:
            means = np.array([summary[name][metric]["mean"] for name in model_order])
            cis = np.array([summary[name][metric]["ci95"] for name in model_order])
        if percent:
            means, cis = 100 * means, 100 * cis
        errors = np.stack([means - cis[:, 0], cis[:, 1] - means])
        axis.bar(x, means, color=[COLORS[name] for name in model_order], alpha=0.88)
        axis.errorbar(x, means, yerr=errors, fmt="none", ecolor="black", capsize=3, lw=1)
        axis.set_title(title)
        axis.set_xticks(x, [DISPLAY_NAMES[name] for name in model_order], rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        if metric in ("path_length_ratio", "control_cost_ratio"):
            axis.axhline(1.0, color="0.3", ls="--", lw=1)
    fig.suptitle("Matched PushT latent-geometry diagnostics (mean and bootstrap 95% CI)")
    fig.savefig(out_dir / "latent-geometry-summary.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--cache-dir", default=os.environ.get("LOCAL_DATASET_DIR"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gallery-size", type=int, default=4096)
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--models",
        default=",".join(MODEL_ORDER),
        help="comma-separated checkpoint labels to render",
    )
    args = parser.parse_args()
    model_order = tuple(name.strip() for name in args.models.split(",") if name.strip())
    unknown = [name for name in model_order if name not in DISPLAY_NAMES]
    if not model_order or unknown:
        parser.error(f"invalid --models value; unknown={unknown}")
    if not {"jepa", "bloop"}.issubset(model_order):
        parser.error("--models must include jepa and bloop for matched example selection")
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")
    if not args.cache_dir:
        raise SystemExit("set LOCAL_DATASET_DIR or pass --cache-dir")

    import torch
    import stable_worldmodel as swm
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from additional_files.diagnose_frozen_representation import (
        load_probe_model,
        parse_checkpoint,
    )
    from additional_files.pixel_tag import PixelTag, attach_pixel_tag
    from utils import get_img_preprocessor

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = get_img_preprocessor("pixels", "pixels", img_size=224)

    def load_raw(num_steps):
        dataset = swm.data.load_dataset(
            args.dataset,
            cache_dir=args.cache_dir,
            num_steps=num_steps,
            frameskip=1,
            keys_to_cache=["state"],
        )
        dataset = attach_pixel_tag(
            dataset,
            PixelTag(mode="video", size=5, seed=0, counterfactual_key="pixels_cf"),
        )
        dataset.transform = None
        return dataset

    gallery_dataset = load_raw(1)
    pair_dataset = load_raw(args.goal_offset + 1)
    gallery_ids = rng.choice(
        len(gallery_dataset), size=min(args.gallery_size, len(gallery_dataset)), replace=False
    ).tolist()
    pair_ids = rng.choice(
        len(pair_dataset), size=min(args.pairs, len(pair_dataset)), replace=False
    ).tolist()

    def collect(dataset, ids, endpoints=False):
        raw, raw_cf, states = [], [], []
        for index in ids:
            sample = dataset[int(index)]
            pixels = torch.as_tensor(sample["pixels"]).clone()
            counterfactual = torch.as_tensor(sample["pixels_cf"]).clone()
            state = torch.as_tensor(sample["state"]).cpu().numpy()
            if endpoints:
                keep = [0, -1]
                raw.append(pixels[keep])
                raw_cf.append(counterfactual[keep])
                states.append(state[keep])
            else:
                raw.append(pixels[0])
                raw_cf.append(counterfactual[0])
                states.append(state[0])
        return torch.stack(raw), torch.stack(raw_cf), np.stack(states)

    gallery_raw, _, gallery_states_raw = collect(gallery_dataset, gallery_ids)
    pair_raw, pair_cf_raw, pair_states_raw = collect(pair_dataset, pair_ids, endpoints=True)
    gallery_physical = state_features(gallery_states_raw)
    pair_physical = state_features(pair_states_raw)
    mean = gallery_physical.mean(0)
    scale = gallery_physical.std(0) + 1e-6
    gallery_state = (gallery_physical - mean) / scale
    pair_state = (pair_physical - mean) / scale

    def prep(frames):
        outputs = []
        for start in range(0, len(frames), args.batch_size):
            chunk = frames[start:start + args.batch_size]
            outputs.append(preprocess({"pixels": chunk})["pixels"])
        return torch.cat(outputs)

    gallery_input = prep(gallery_raw)
    pair_input = prep(pair_raw.reshape(-1, *pair_raw.shape[2:]))
    pair_cf_input = prep(pair_cf_raw.reshape(-1, *pair_cf_raw.shape[2:]))

    @torch.no_grad()
    def encode(model, inputs):
        outputs = []
        for start in range(0, len(inputs), args.batch_size):
            x = inputs[start:start + args.batch_size].to(device)
            cls = model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0].float()
            outputs.append(model.projector(cls).cpu().numpy())
        return np.concatenate(outputs)

    specs = [parse_checkpoint(value) for value in args.checkpoint]
    labels = [label for label, _, _ in specs]
    missing = [name for name in model_order if name not in labels]
    if missing:
        raise ValueError(f"Missing checkpoint labels: {missing}")

    embeddings, endpoint_embeddings, cf_endpoint_embeddings, linear_r2 = {}, {}, {}, {}
    for label, run_name, filename in specs:
        print(f"loading {label}: {run_name}/{filename}", flush=True)
        model = load_probe_model(f"{run_name}/{filename}").to(device).eval().requires_grad_(False)
        embeddings[label] = encode(model, gallery_input)
        endpoint_embeddings[label] = encode(model, pair_input).reshape(len(pair_ids), 2, -1)
        cf_endpoint_embeddings[label] = encode(model, pair_cf_input).reshape(len(pair_ids), 2, -1)
        split = int(0.8 * len(gallery_state))
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        probe.fit(embeddings[label][:split], gallery_state[:split])
        linear_r2[label] = float(
            r2_score(gallery_state[split:], probe.predict(embeddings[label][split:]))
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gallery_rgb = [_to_rgb(frame) for frame in gallery_raw]
    results = {name: [] for name in model_order}
    examples = []
    first_latent_plans = {}
    for pair_index in range(len(pair_ids)):
        oracle = straight_path(pair_state[pair_index], args.steps)
        oracle_indices = nearest_indices(oracle, gallery_state)
        oracle_frames = [gallery_rgb[i] for i in oracle_indices]
        # Exact endpoint images make every row share the same task endpoints.
        oracle_frames[0] = _to_rgb(pair_raw[pair_index, 0])
        oracle_frames[-1] = _to_rgb(pair_raw[pair_index, 1])
        rows = {"oracle": oracle_frames}
        physical_paths = {"oracle": oracle}
        pair_record = {"pair_index": pair_index, "dataset_index": int(pair_ids[pair_index]), "models": {}}
        for name in model_order:
            plan = straight_path(endpoint_embeddings[name][pair_index], args.steps)
            cf_plan = straight_path(cf_endpoint_embeddings[name][pair_index], args.steps)
            if pair_index == 0:
                first_latent_plans[name] = plan
            indices = nearest_indices(plan, embeddings[name])
            cf_indices = nearest_indices(cf_plan, embeddings[name])
            decoded = gallery_state[indices].copy()
            decoded_cf = gallery_state[cf_indices].copy()
            decoded[[0, -1]] = pair_state[pair_index]
            decoded_cf[[0, -1]] = pair_state[pair_index]
            metrics = path_metrics(decoded, oracle)
            metrics.update({
                "counterfactual_path_rmse": float(np.sqrt(np.mean(np.square(decoded_cf - decoded)))),
                "counterfactual_retrieval_change_rate": float(np.mean(cf_indices[1:-1] != indices[1:-1])),
            })
            results[name].append(metrics)
            frames = [gallery_rgb[i] for i in indices]
            frames[0] = _to_rgb(pair_raw[pair_index, 0])
            frames[-1] = _to_rgb(pair_raw[pair_index, 1])
            rows[name] = frames
            physical_paths[name] = decoded
            pair_record["models"][name] = {
                "retrieval_indices": indices.tolist(),
                "counterfactual_retrieval_indices": cf_indices.tolist(),
                **metrics,
            }
        examples.append((rows, oracle_frames, physical_paths, pair_record))

    summary = {}
    for model_index, name in enumerate(model_order):
        summary[name] = {"linear_state_r2": linear_r2[name]}
        for metric in results[name][0]:
            mean_value, ci = bootstrap_mean_ci(
                [record[metric] for record in results[name]], args.seed + model_index
            )
            summary[name][metric] = {"mean": mean_value, "ci95": ci}

    # Prefer examples where JEPA deviates more than repair, but declare this
    # selection and keep all-pair quantitative summaries untouched.
    score = np.array([
        examples[i][3]["models"]["jepa"]["state_rmse_to_oracle"]
        - examples[i][3]["models"]["bloop"]["state_rmse_to_oracle"]
        for i in range(len(examples))
    ])
    selected = np.argsort(-score, kind="stable")[: min(args.examples, len(examples))]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for rank, pair_index in enumerate(selected, start=1):
        rows, oracle_frames, physical_paths, _ = examples[int(pair_index)]
        render_plan_grid(args.out_dir, rank, rows, oracle_frames, model_order)
        render_state_paths(args.out_dir, rank, physical_paths)
    render_embedding_pca(
        args.out_dir, embeddings, first_latent_plans, gallery_state, model_order
    )
    render_summaries(args.out_dir, summary, model_order)

    payload = {
        "protocol": {
            "dataset": args.dataset,
            "gallery_size": len(gallery_ids),
            "pairs": len(pair_ids),
            "steps": args.steps,
            "goal_offset": args.goal_offset,
            "seed": args.seed,
            "decoder": "Euclidean 1-nearest-neighbour over one shared gallery",
            "representation": "projector output",
            "tag_intervention": "endpoint RGB tag only; physical endpoints and gallery fixed",
            "example_selection": "largest JEPA-minus-repair physical RMSE gap; aggregate metrics use all pairs",
            "scope_warning": "empirical diagnostic inspired by LeJEPA identifiability; theorem assumptions are not asserted",
            "device": str(device),
            "models": list(model_order),
        },
        "checkpoints": {label: f"{run}/{filename}" for label, run, filename in specs},
        "gallery_dataset_indices": [int(i) for i in gallery_ids],
        "pair_dataset_indices": [int(i) for i in pair_ids],
        "selected_pair_positions": [int(i) for i in selected],
        "summary": summary,
        "pairs": [record for _, _, _, record in examples],
    }
    (args.out_dir / "latent-geometry.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"LATENT_GEOMETRY_VISUALIZATION_COMPLETE {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
