from __future__ import annotations

import numpy as np

from leworldmodel.additional_files.tag_folding_probe_metrics import (
    absolute_tag_sensitivity,
    participation_ratio,
    summarize_tag_sweep,
    twonn_intrinsic_dimension,
)


def test_participation_ratio_recovers_linear_rank() -> None:
    rng = np.random.default_rng(7)
    factors = rng.normal(size=(512, 3))
    mixing = rng.normal(size=(3, 24))
    embeddings = factors @ mixing
    assert 1.0 <= participation_ratio(embeddings) <= 3.01


def test_curved_low_dimensional_structure_spreads_globally() -> None:
    rng = np.random.default_rng(19)
    values = np.sort(rng.uniform(-3.0, 3.0, size=600))
    frequencies = np.arange(1.0, 17.0)
    embeddings = np.concatenate(
        [np.sin(values[:, None] * frequencies), np.cos(values[:, None] * frequencies)],
        axis=1,
    )
    global_dimension = participation_ratio(embeddings)
    local_dimension = twonn_intrinsic_dimension(embeddings)
    assert global_dimension > 8.0
    assert np.isfinite(local_dimension)
    assert local_dimension < global_dimension


def test_absolute_sensitivity_is_zero_for_constant_embeddings() -> None:
    embeddings = np.ones((16, 8))
    result = absolute_tag_sensitivity(embeddings)
    assert result["centroid_rms"] == 0.0
    assert result["pairwise_rms"] == 0.0


def test_summary_contains_absolute_and_geometry_metrics() -> None:
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(128, 12))
    summary = summarize_tag_sweep(embeddings, projections=16, seed=3)
    assert summary["samples"] == 128
    assert summary["ambient_dimension"] == 12
    assert "absolute_tag_sensitivity" in summary
    assert "folding_ratio" in summary
    assert summary["heldout_projection_gaussian_gap"]["projections"] == 16
