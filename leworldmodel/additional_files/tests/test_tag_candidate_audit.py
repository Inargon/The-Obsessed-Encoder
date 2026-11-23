import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from additional_files import diagnose_tag_intervention
from additional_files.tag_candidate_audit import CONDITIONS, candidate_gaps, sample_episode_bank


class Dataset:
    column_names = ("episode_idx", "step_idx", "action", "pixels")

    def __init__(self):
        self.lengths = [8] * 8
        self.offsets = np.arange(8) * 8
        self.clip_indices = [(episode, start) for episode in range(8) for start in range(3)]
        self.columns = {
            "episode_idx": np.repeat(np.arange(100, 108), 8),
            "step_idx": np.tile(np.arange(8), 8),
            "action": np.arange(64, dtype=np.float32)[:, None],
        }

    def __len__(self):
        return len(self.clip_indices)

    def get_col_data(self, column):
        return self.columns[column]

    def get_dim(self, column):
        assert column == "action"
        return 1

    def __getitem__(self, index):
        episode, start = self.clip_indices[index]
        row = self.offsets[episode] + start
        frames = torch.arange(row, row + 6, dtype=torch.uint8)
        return {
            "action": torch.from_numpy(self.columns["action"][row:row + 6]),
            "pixels": frames[:, None, None, None].expand(6, 3, 3, 3).clone(),
        }


def test_sampler_separates_physical_episodes_and_rejects_wrong_action_alignment():
    dataset = Dataset()
    query, bank = sample_episode_bank(dataset, np.random.default_rng(73), 4, 3, 6, 1, 3)
    assert len({item["episode"] for item in query + bank}) == 7
    chosen = query[0]["clip"]
    episode, start = dataset.clip_indices[chosen]
    dataset.offsets[episode] += 4 - start
    with pytest.raises(ValueError, match="boundary|nonconsecutive"):
        sample_episode_bank(dataset, np.random.default_rng(73), 4, 3, 6, 1, 3)

    class MisalignedActions(Dataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["action"] = sample["action"].roll(shifts=1, dims=0)
            return sample

    with pytest.raises(ValueError, match="differs from raw rows"):
        sample_episode_bank(MisalignedActions(), np.random.default_rng(73), 4, 3, 6, 1, 3)


def test_gap_analysis_distinguishes_ties_and_compares_the_same_rival():
    costs = dict(
        reference=[[1, 2, 3], [5, 4, 6]],
        context_tag=[[10, 10, 11], [4, 4, 5]],
        goal_tag=[[2, 3, 0], [1, 2, 3]],
        both_tags=[[4, 5, 2], [9, 8, 7]],
        action_sequence=[[0, 2, 4], [1, 2, 3]],
    )
    result = candidate_gaps(costs)
    assert result["reference_winner"] == [0, 1]
    assert result["reference_gap"] == [1, 1]
    context = result["conditions"]["context_tag"]
    assert context["index_flip"] == [False, True]
    assert context["strict_flip"] == [False, False]
    assert result["conditions"]["goal_tag"]["winner_reference_rank"] == [3, 2]
    assert result["strict_any_tag_flip"] == [True, True]
    assert result["worst_tag_rival"] == [2, 0]
    assert result["worst_tag_rival_gaps"]["goal_tag"] == [-2, -1]
    assert result["worst_tag_rival_interaction"] == [1, 3]
    offsets = {key: np.asarray(value) + i * 20 for i, (key, value) in enumerate(costs.items())}
    assert candidate_gaps(offsets) == result


def test_candidate_audit_reuses_exact_plans_and_only_retags_intended_endpoint(tmp_path, monkeypatch):
    dataset = Dataset()
    calls = []

    class Model(torch.nn.Module):
        def get_cost(self, info, plans):
            calls.append({key: tensor.clone() for key, tensor in info.items()})
            torch.testing.assert_close(info["action"], plans)
            info["action"].zero_()
            return (plans[:, :, -1, 0] - info["goal"][:, :, 0, 0, 0, 0]).square()

        def encode(self, info):
            return {"emb": info["pixels"][:, :, :, 1, 1], "act_emb": info["action"]}

        def predict(self, emb, action):
            return emb + action

    monkeypatch.setitem(sys.modules, "stable_worldmodel",
                        SimpleNamespace(data=SimpleNamespace(load_dataset=lambda *a, **k: dataset)))
    monkeypatch.setitem(sys.modules, "additional_files.evaluate_reacher_checkpoint",
                        SimpleNamespace(load_inference_model=lambda *a: Model()))
    monkeypatch.setitem(sys.modules, "additional_files.diagnose_frozen_representation",
                        SimpleNamespace(parse_checkpoint=lambda s: ("example", "run", "test.pt")))
    monkeypatch.setitem(sys.modules, "utils", SimpleNamespace(
        get_img_preprocessor=lambda *a, **k: lambda item: {"pixels": item["pixels"].float()},
        get_column_normalizer=lambda *a: lambda item: item,
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("LOCAL_DATASET_DIR", str(tmp_path))
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    checkpoint = tmp_path / "checkpoints/run/test.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    output = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", [
        "diagnose", "--candidate-audit", "--checkpoint", "example=run/test.pt", "--out",
        str(output), "--num-clips", "4", "--candidates", "3", "--frameskip", "1",
        "--history", "3", "--horizon", "3", "--tag-size", "1",
    ])
    diagnose_tag_intervention.main()
    report = json.loads(output.read_text())
    protocol = report["protocol"]
    assert len({item["episode"] for group in protocol["episode_split"].values() for item in group}) == 7
    assert len({tuple(row) for row in protocol["candidate_source_indices"]}) == 1
    assert not protocol["training_independent_episodes_verified"]
    for start in range(0, len(calls), 5):
        reference, both, context, goal, action = calls[start:start + 5]
        for view in (both, context, goal):
            torch.testing.assert_close(reference["action"], view["action"])
        torch.testing.assert_close(reference["action"][:, :, :2], action["action"][:, :, :2])
        torch.testing.assert_close(reference["action"][:, :, 2:].roll(1, 2), action["action"][:, :, 2:])
        torch.testing.assert_close(reference["pixels"], goal["pixels"])
        torch.testing.assert_close(reference["goal"], context["goal"])
        torch.testing.assert_close(context["pixels"], both["pixels"])
        torch.testing.assert_close(goal["goal"], both["goal"])
        assert torch.all((reference["pixels"] != context["pixels"]).any(dim=3).sum((-2, -1)) == 1)
        assert torch.all((reference["goal"] != goal["goal"]).any(dim=3).sum((-2, -1)) == 1)
    audit = report["runs"]["example"]["candidate_audit"]
    assert set(audit["costs"]) == set(CONDITIONS)
    assert np.asarray(audit["costs"]["reference"]).shape == (4, 3)
    assert np.asarray(audit["anchor"]["reference"]).shape == (4, 3)
