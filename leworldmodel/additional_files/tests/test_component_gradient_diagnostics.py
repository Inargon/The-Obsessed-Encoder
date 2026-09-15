from __future__ import annotations

import math

import torch

from additional_files.component_gradient_diagnostics import (
    measure_component_gradient_geometry,
)


def test_joint_span_covers_prediction_across_orthogonal_components():
    embedding = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    prediction = embedding[0, 0] + embedding[0, 1]
    components = {
        "h1": embedding[0, 0],
        "h2": embedding[0, 1],
    }

    metrics = measure_component_gradient_geometry(
        prediction, components, embedding
    )

    torch.testing.assert_close(
        metrics["component_grad/joint_span/retained_fraction"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        metrics["component_grad/h1/cosine"],
        torch.tensor(1.0 / math.sqrt(2.0)),
    )


def test_cancellation_ratio_detects_opposed_control_components():
    embedding = torch.tensor([[1.0, 1.0]], requires_grad=True)
    prediction = embedding.sum()
    components = {
        "positive": embedding[0, 0],
        "negative": -embedding[0, 0],
    }

    metrics = measure_component_gradient_geometry(
        prediction, components, embedding
    )

    torch.testing.assert_close(
        metrics["component_grad/joint_span/cancellation_ratio"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["component_grad/negative/negative_fraction"],
        torch.tensor(1.0),
    )


def test_joint_span_solve_stays_float32_under_bfloat16_autocast():
    embedding = torch.tensor([[1.0, 2.0]], requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        metrics = measure_component_gradient_geometry(
            embedding.sum(),
            {"first": embedding[0, 0], "second": embedding[0, 1]},
            embedding,
        )

    retained = metrics["component_grad/joint_span/retained_fraction"]
    assert retained.dtype == torch.float32
    torch.testing.assert_close(retained, torch.tensor(1.0))
