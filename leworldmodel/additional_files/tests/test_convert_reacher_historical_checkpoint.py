from collections import OrderedDict

import pytest
import torch

from additional_files.convert_reacher_historical_checkpoint import (
    convert_state_dict,
    historical_key,
)


@pytest.mark.parametrize(
    ("modern", "historical"),
    [
        (
            "encoder.layers.3.attention.q_proj.weight",
            "encoder.encoder.layer.3.attention.attention.query.weight",
        ),
        (
            "encoder.layers.11.attention.o_proj.bias",
            "encoder.encoder.layer.11.attention.output.dense.bias",
        ),
        (
            "encoder.layers.0.mlp.fc1.weight",
            "encoder.encoder.layer.0.intermediate.dense.weight",
        ),
        (
            "encoder.layers.7.mlp.fc2.bias",
            "encoder.encoder.layer.7.output.dense.bias",
        ),
        (
            "encoder.layers.2.layernorm_before.weight",
            "encoder.encoder.layer.2.layernorm_before.weight",
        ),
    ],
)
def test_historical_key(modern, historical):
    assert historical_key(modern) == historical


def test_conversion_removes_only_control_and_preserves_values():
    shared = torch.tensor([1.0])
    layer = torch.tensor([2.0])
    state = OrderedDict(
        (
            ("projector.weight", shared),
            ("encoder.layers.0.attention.k_proj.weight", layer),
            ("control_objective.inverse.weight", torch.tensor([3.0])),
        )
    )
    converted, removed, renamed = convert_state_dict(state)
    assert list(removed) == ["control_objective.inverse.weight"]
    assert renamed == [
        (
            "encoder.layers.0.attention.k_proj.weight",
            "encoder.encoder.layer.0.attention.attention.key.weight",
        )
    ]
    assert converted["projector.weight"] is shared
    assert (
        converted["encoder.encoder.layer.0.attention.attention.key.weight"]
        is layer
    )
