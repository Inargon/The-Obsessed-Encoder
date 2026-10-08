"""Numerical diagnostics for the controlled tag-folding probe.

The functions in this module deliberately operate on NumPy arrays only.  Model
and dataset loading live in the campaign script; keeping the numerical core
separate makes the scientific quantities easy to test and reuse.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def _as_samples(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("embeddings must have shape [samples, dimensions]")
    if values.shape[0] < 3:
        raise ValueError("at least three samples are required")
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain non-finite values")
    return values


def covariance_eigenvalues(embeddings: np.ndarray) -> np.ndarray:
    """Return descending sample-covariance eigenvalues."""

    values = _as_samples(embeddings)
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = singular_values**2 / max(values.shape[0] - 1, 1)
    return np.maximum(eigenvalues, 0.0)


def participation_ratio(embeddings: np.ndarray, eps: float = 1e-12) -> float:
    """Global linear effective dimension from the covariance spectrum."""

    eigenvalues = covariance_eigenvalues(embeddings)
    total = float(eigenvalues.sum())
    if total <= eps:
        return 0.0
    return total * total / float(np.square(eigenvalues).sum() + eps)


def pca_dimensions_for_variance(
    embeddings: np.ndarray,
    fractions: tuple[float, ...] = (0.90, 0.95, 0.99),
    eps: float = 1e-12,
) -> dict[str, int]:
    """Number of principal directions needed for requested variance fractions."""

    eigenvalues = covariance_eigenvalues(embeddings)
    total = float(eigenvalues.sum())
    if total <= eps:
        return {f"pca_{int(round(100 * fraction))}": 0 for fraction in fractions}
    cumulative = np.cumsum(eigenvalues) / total
    return {
        f"pca_{int(round(100 * fraction))}": int(np.searchsorted(cumulative, fraction) + 1)
        for fraction in fractions
    }


def twonn_intrinsic_dimension(embeddings: np.ndarray, eps: float = 1e-12) -> float:
    """Estimate local intrinsic dimension with the TwoNN maximum likelihood rule.

    Duplicate samples are ignored because their first-neighbour distance is zero.
    The estimator is intended as a comparative diagnostic, not a theorem about
    the true manifold dimension.
    """

    values = _as_samples(embeddings)
    squared_norms = np.sum(values * values, axis=1, keepdims=True)
    squared_distances = squared_norms + squared_norms.T - 2.0 * values @ values.T
    squared_distances = np.maximum(squared_distances, 0.0)
    np.fill_diagonal(squared_distances, np.inf)
    nearest_two = np.partition(squared_distances, kth=1, axis=1)[:, :2]
    nearest_two.sort(axis=1)
    r1 = np.sqrt(nearest_two[:, 0])
    r2 = np.sqrt(nearest_two[:, 1])
    valid = (r1 > eps) & (r2 > r1)
    if int(valid.sum()) < 3:
        return float("nan")
    log_ratios = np.log(r2[valid] / r1[valid])
    denominator = float(log_ratios.sum())
    if denominator <= eps:
        return float("nan")
    return float(valid.sum() / denominator)


def absolute_tag_sensitivity(embeddings: np.ndarray) -> dict[str, float]:
    """Absolute displacement caused by a controlled tag sweep.

    The centroid-based quantity avoids privileging an arbitrary reference colour.
    Pairwise RMS is reported in the same latent units for interpretability.
    """

    values = _as_samples(embeddings)
    centered = values - values.mean(axis=0, keepdims=True)
    mean_squared_radius = float(np.mean(np.sum(centered * centered, axis=1)))
    # E ||X-X'||^2 = 2 E ||X-E X||^2 for independent samples.
    return {
        "centroid_rms": float(np.sqrt(mean_squared_radius)),
        "pairwise_rms": float(np.sqrt(2.0 * mean_squared_radius)),
    }


def heldout_projection_gaussian_gap(
    embeddings: np.ndarray,
    *,
    projections: int = 512,
    seed: int = 0,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Quantile gap to N(0, 1) over held-out random one-dimensional projections.

    Each projection is standardized before comparison, so this measures shape
    rather than mean or scale.  It is not the SIGReg training statistic; it is a
    transparent held-out audit using independent directions.
    """

    values = _as_samples(embeddings)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(values.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + eps
    projected = values @ directions
    projected -= projected.mean(axis=0, keepdims=True)
    projected /= projected.std(axis=0, ddof=1, keepdims=True) + eps
    projected.sort(axis=0)

    probabilities = (np.arange(values.shape[0], dtype=np.float64) + 0.5) / values.shape[0]
    normal = NormalDist()
    reference = np.asarray([normal.inv_cdf(float(p)) for p in probabilities])[:, None]
    per_projection = np.mean(np.square(projected - reference), axis=0)
    return {
        "mean_quantile_mse": float(per_projection.mean()),
        "median_quantile_mse": float(np.median(per_projection)),
        "p95_quantile_mse": float(np.quantile(per_projection, 0.95)),
        "projections": int(projections),
    }


def summarize_tag_sweep(
    embeddings: np.ndarray,
    *,
    projections: int = 512,
    seed: int = 0,
) -> dict[str, object]:
    """Compute the complete geometry summary for one model's tag sweep."""

    global_dimension = participation_ratio(embeddings)
    local_dimension = twonn_intrinsic_dimension(embeddings)
    folding_ratio = (
        float(global_dimension / local_dimension)
        if np.isfinite(local_dimension) and local_dimension > 0
        else float("nan")
    )
    return {
        "samples": int(np.asarray(embeddings).shape[0]),
        "ambient_dimension": int(np.asarray(embeddings).shape[1]),
        "absolute_tag_sensitivity": absolute_tag_sensitivity(embeddings),
        "global_participation_dimension": float(global_dimension),
        "local_twonn_dimension": float(local_dimension),
        "folding_ratio": folding_ratio,
        **pca_dimensions_for_variance(embeddings),
        "heldout_projection_gaussian_gap": heldout_projection_gaussian_gap(
            embeddings,
            projections=projections,
            seed=seed,
        ),
    }
