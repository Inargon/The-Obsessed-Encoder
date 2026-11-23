"""Compare the training SIGReg statistic with geometry and candidate rankings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

if __package__ and "." in __package__:
    from ..module import SIGReg
else:
    from module import SIGReg

if __package__:
    from .control_relevance_audit.tag_rank_metrics import compare
    from .tag_folding_probe_metrics import participation_ratio
else:
    from control_relevance_audit.tag_rank_metrics import compare
    from tag_folding_probe_metrics import participation_ratio


class ProjectionAudit:
    """Keep directions and Gaussian reference draws fixed across matched clouds."""

    def __init__(self, samples: int, dimension: int, projections: int, seed: int, repeats: int):
        if samples < 2 or min(dimension, projections, repeats) < 1:
            raise ValueError("require at least two samples and positive dimensions, projections and repeats")
        self.shape = (samples, dimension)
        self.sigreg = SIGReg()
        direction_rng = torch.Generator().manual_seed(seed)
        directions = torch.randn(dimension, projections, generator=direction_rng, dtype=torch.float32)
        self.directions = torch.nn.functional.normalize(directions, dim=0)
        self.axes = torch.eye(dimension, dtype=torch.float32)
        null_rng = torch.Generator().manual_seed(seed + 1)
        null_means = np.array([
            self._scores(
                torch.randn(self.shape, generator=null_rng, dtype=torch.float32),
                self.directions,
            ).mean().item()
            for _ in range(repeats)
        ])
        self.reference = {
            "draws": repeats,
            "mean": float(null_means.mean()),
            "p95": float(np.quantile(null_means, 0.95)),
        }

    @torch.no_grad()
    def _scores(self, values: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        # Limit the temporary [samples, directions, knots] tensor on CPU.
        return torch.cat([
            self.sigreg.projected_statistics(values @ block)
            for block in directions.split(64, dim=1)
        ])

    def measure(self, embeddings: np.ndarray) -> dict:
        values = torch.as_tensor(embeddings, dtype=torch.float32)
        if tuple(values.shape) != self.shape or not torch.isfinite(values).all():
            raise ValueError(f"expected finite embeddings with shape {self.shape}")
        scores = self._scores(values, self.directions)
        axis_scores = self._scores(values, self.axes)
        mean = scores.mean().item()
        dimension = self.shape[1]
        return {
            "random_projection_mean": mean,
            "random_projection_p95": scores.quantile(0.95).item(),
            "axis_projection_max": axis_scores.max().item(),
            "mean_per_sample": mean / self.shape[0],
            "squared_norm_variance_over_gaussian": (
                values.square().sum(dim=1).var(correction=1).item() / (2 * dimension)
            ),
        }


def audit_tag_sweeps(
    embeddings: dict[str, np.ndarray], *, projections: int = 512,
    seed: int = 73, repeats: int = 32,
) -> dict:
    """Evaluate raw projected embeddings for each fixed-scene RGB sweep."""
    if not embeddings:
        raise ValueError("provide at least one model's embeddings")
    auditors = {}
    runs = {}
    matched_shape = None
    for name, scenes in embeddings.items():
        scenes = np.asarray(scenes)
        if scenes.ndim != 3 or scenes.shape[0] == 0:
            raise ValueError(f"{name} must contain [scene, color, dimension] embeddings")
        if matched_shape is None:
            matched_shape = scenes.shape[:2]
        elif scenes.shape[:2] != matched_shape:
            raise ValueError("models must have the same numbers of scenes and colors")
        shape = scenes.shape[1:]
        if shape not in auditors:
            auditors[shape] = ProjectionAudit(*shape, projections, seed, repeats)
        audit = auditors[shape]
        rows = [audit.measure(scene) for scene in scenes]
        runs[name] = {
            "dimension": shape[1],
            "gaussian_reference_for_random_projection_mean": audit.reference,
            "aggregate_mean": {
                key: float(np.mean([row[key] for row in rows])) for key in rows[0]
            },
            "scenes": rows,
        }
    return {
        "protocol": {
            "scope": "tag-conditional distributions with physical pixels fixed within each scene",
            "statistic": "LeWorldModel SIGReg on raw embeddings, including batch-size scaling",
            "scenes": matched_shape[0],
            "colors_per_scene": matched_shape[1],
            "random_projections": projections,
            "seed": seed,
        },
        "runs": runs,
    }


def harmonic_embedding(phases: np.ndarray, dimension: int) -> np.ndarray:
    frequencies = np.arange(1, dimension // 2 + 1)
    angles = 2 * np.pi * np.asarray(phases)[..., None] * frequencies
    return np.sqrt(2) * np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)


def synthetic_folding(
    dimensions: list[int], *, samples: int, fold: int, projections: int, seed: int, repeats: int,
) -> dict:
    """Hold the empirical marginal fixed while changing the tag-to-latent map."""
    if not dimensions or any(d < 2 or d % 2 or d >= samples for d in dimensions):
        raise ValueError("dimensions must be positive even numbers below the sample count")
    if fold < 3 or math.gcd(fold, samples) != 1:
        raise ValueError("fold must be at least three and coprime to the sample count")
    indices = np.arange(samples)
    runs = []
    for dimension in dimensions:
        base = harmonic_embedding(indices / samples, dimension)
        audit = ProjectionAudit(samples, dimension, projections, seed, repeats)
        # Multiplication by an integer coprime to samples permutes the grid exactly.
        folded = base[(fold * indices) % samples]
        gap = 1 / (2 * dimension)
        states = np.array([0.0, gap])
        tags = np.array([0.0, -gap / fold])
        goal = harmonic_embedding(0.0, dimension)
        speed = 2 * np.pi * math.sqrt(2 * sum(j * j for j in range(1, dimension // 2 + 1)))

        def evaluate(values: np.ndarray, frequency: int) -> dict:
            endpoints = harmonic_embedding(states + frequency * tags[:, None], dimension)
            costs = np.mean(np.square(endpoints - goal), axis=-1)
            return {
                "fold_frequency": frequency,
                "path_length": frequency * speed,
                **audit.measure(values),
                "candidate_ranking": compare(costs, (states / gap) ** 2),
            }

        runs.append({
            "dimension": dimension,
            "intrinsic_dimension": 1,
            "covariance_participation_ratio": participation_ratio(base),
            "gaussian_reference_for_random_projection_mean": audit.reference,
            "candidate_positions": states.tolist(),
            "reference_and_changed_tags": tags.tolist(),
            "base": evaluate(base, 1),
            "folded": evaluate(folded, fold),
        })
    return {
        "protocol": {
            "samples": samples,
            "random_projections": projections,
            "seed": seed,
            "goal_position_and_tag": [0.0, 0.0],
            "candidate_geometry": "shared tag per action bank, fixed goal tag",
            "physical_regret_units": "squared candidate position in units of the candidate spacing",
        },
        "runs": runs,
    }


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--projections", type=int, default=512)
    common.add_argument("--gaussian-repeats", type=int, default=32)
    common.add_argument("--seed", type=int, default=73)
    common.add_argument("--out", type=Path)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("synthetic", parents=[common])
    synth.add_argument("--dimensions", nargs="+", type=int, default=[16, 64, 192])
    synth.add_argument("--samples", type=int, default=1024)
    synth.add_argument("--fold", type=int, default=7)
    frozen = commands.add_parser("embeddings", parents=[common])
    frozen.add_argument("input", type=Path, help="tag-folding-probe-embeddings.npz")
    args = parser.parse_args()
    options = dict(projections=args.projections, seed=args.seed, repeats=args.gaussian_repeats)
    if args.command == "synthetic":
        report = synthetic_folding(args.dimensions, samples=args.samples, fold=args.fold, **options)
    else:
        with np.load(args.input, allow_pickle=False) as bundle:
            report = audit_tag_sweeps({name: bundle[name] for name in ("jepa", "ours")}, **options)
            report["protocol"]["source"] = str(args.input)
            if "dataset_indices" in bundle:
                report["protocol"]["dataset_indices"] = bundle["dataset_indices"].tolist()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
