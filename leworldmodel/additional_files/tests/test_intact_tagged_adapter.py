from __future__ import annotations

from types import SimpleNamespace

import torch

from additional_files.intact_tagged_adapter import (
    install_evaluation_tag,
    install_training_tag,
)
from additional_files.intact_tagged_campaign import EVAL_MODES


def test_official_lewm_evaluation_mode_names():
    assert EVAL_MODES == ("direct", "cem")


def test_training_adapter_attaches_exact_pixel_tag(monkeypatch):
    captured = {}
    dataset = object()
    module = SimpleNamespace(
        swm=SimpleNamespace(
            data=SimpleNamespace(load_dataset=lambda *args, **kwargs: dataset)
        )
    )

    def fake_attach(value, tag):
        captured["dataset"] = value
        captured["tag"] = tag
        return "tagged"

    monkeypatch.setattr("pixel_tag.attach_pixel_tag", fake_attach)
    install_training_tag(module, mode="video", size=5, seed=7)
    assert module.swm.data.load_dataset("pusht") == "tagged"
    assert captured["dataset"] is dataset
    assert captured["tag"].mode == "video"
    assert captured["tag"].size == 5
    assert captured["tag"].seed == 7


def test_eval_adapter_stamps_before_upstream_preprocessing():
    calls = []

    class Base:
        def __call__(self, value):
            calls.append("base")
            assert value[:, :5, :5].sum() > 0
            return value

    module = SimpleNamespace(img_transform=lambda cfg: Base())
    install_evaluation_tag(module, mode="video", size=5, seed=3)
    transform = module.img_transform(SimpleNamespace())
    transform(torch.zeros(3, 16, 16, dtype=torch.uint8))
    assert calls == ["base"]
