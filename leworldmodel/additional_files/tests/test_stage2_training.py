from __future__ import annotations

from types import SimpleNamespace

import torch

from additional_files.stage2_training import (
    configure_stage2,
    enforce_frozen_representation_eval,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 3), torch.nn.Dropout(0.5)
        )
        self.projector = torch.nn.BatchNorm1d(3)
        self.predictor = torch.nn.Linear(3, 3)


def cfg(path, freeze):
    return {
        "stage2": SimpleNamespace(
            enabled=True,
            init_checkpoint=str(path),
            freeze_representation=freeze,
            get=lambda key, default=None: {
                "enabled": True,
                "freeze_representation": freeze,
            }.get(key, default),
        )
    }


def test_stage2_loads_strictly_and_freezes_complete_coordinate_system(tmp_path):
    source = TinyModel()
    with torch.no_grad():
        source.encoder[0].weight.fill_(2.0)
    checkpoint = tmp_path / "stage1.pt"
    torch.save(source.state_dict(), checkpoint)

    target = TinyModel()
    metadata = configure_stage2(target, cfg(checkpoint, True))
    assert torch.equal(target.encoder[0].weight, source.encoder[0].weight)
    assert not any(p.requires_grad for p in target.encoder.parameters())
    assert not any(p.requires_grad for p in target.projector.parameters())
    assert all(p.requires_grad for p in target.predictor.parameters())
    assert metadata["optimizer_state_resumed"] is False

    target.train()
    enforce_frozen_representation_eval(target)
    assert not target.encoder.training
    assert not target.projector.training
    assert target.predictor.training


def test_unfrozen_reset_control_loads_weights_without_freezing(tmp_path):
    source = TinyModel()
    checkpoint = tmp_path / "stage1.pt"
    torch.save(source.state_dict(), checkpoint)
    target = TinyModel()
    metadata = configure_stage2(target, cfg(checkpoint, False))
    assert all(p.requires_grad for p in target.parameters())
    assert metadata["freeze_representation"] is False
