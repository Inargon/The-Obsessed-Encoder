from __future__ import annotations

from types import SimpleNamespace

import torch

from additional_files.intact_tagged_adapter import (
    install_evaluation_tag,
    install_training_tag,
    split_hydra_invocation,
)
from additional_files.intact_tagged_campaign import (
    EVAL_MODES,
    adapter_prefix,
    validate_clean_audit,
)


def test_official_lewm_evaluation_mode_names():
    assert EVAL_MODES == ("direct", "cem")


def test_adapter_global_flags_precede_remainder_phase(tmp_path):
    args = SimpleNamespace(
        intact_python=tmp_path / "python",
        intact_root=tmp_path / "INTACT-JEPA",
    )
    command = adapter_prefix(args, "train", 7)
    phase_index = command.index("train")
    assert command.index("--intact-root") < phase_index
    assert command.index("--tag-seed") < phase_index
    assert command[phase_index + 1] == "--"


def test_adapter_splits_hydra_config_name_from_overrides():
    config_name, overrides = split_hydra_invocation(
        "train",
        [
            "--config-name=intact_goal",
            "output_model_name=tagged",
            "trainer.max_epochs=1",
        ],
    )
    assert config_name == "intact_goal"
    assert overrides == [
        "output_model_name=tagged",
        "trainer.max_epochs=1",
    ]


def test_adapter_uses_phase_default_config_name():
    assert split_hydra_invocation("train", ["seed=3"]) == (
        "intact_goal",
        ["seed=3"],
    )
    assert split_hydra_invocation("eval", ["seed=3"]) == (
        "pusht",
        ["seed=3"],
    )


def test_clean_audit_is_content_and_checkpoint_hash_gated(tmp_path):
    import hashlib
    import json

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"released checkpoint")
    record = {
        "status": "pass",
        "benchmark": "clean PushT",
        "upstream_commit": (
            "653ee22266a34a74efca21b0b03dfc1fd6fa37ff"
        ),
        "protocol": "official LeWM",
        "inference_mode": "direct",
        "search_enabled": False,
        "num_eval": 100,
        "success_rate": 0.86,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(record), encoding="utf-8")
    valid, _, error = validate_clean_audit(audit)
    assert valid
    assert error is None

    checkpoint.write_bytes(b"changed")
    valid, _, error = validate_clean_audit(audit)
    assert not valid
    assert "SHA-256" in error


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
