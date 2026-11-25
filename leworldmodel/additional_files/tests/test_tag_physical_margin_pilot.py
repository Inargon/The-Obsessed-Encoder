import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from additional_files import tag_physical_margin_pilot as pilot
from additional_files.tag_margin_scoring import ARMS, margin_loss, physical_pairs


def test_pairs_use_clear_physical_outcomes_and_union_of_hard_negatives():
    costs = torch.tensor([[[1., 0., -9., 2., 3.], [1., 2., -9., 0., 3.],
                           [1., 2., -9., 3., 0.], [3., 0., -9., 2., 1.]]])
    distance = torch.tensor([[0., 2.1, 1.5, 3., 4.]])
    qs, better, worse = physical_pairs(costs, distance, torch.tensor([True]), hard_negatives=1)
    assert qs.tolist() == better.tolist() == [0, 0, 0]
    assert worse.tolist() == [1, 3, 4]
    assert not len(physical_pairs(costs, distance, torch.tensor([False]))[0])


def test_worst_view_is_taken_for_each_pair_before_averaging():
    scores = torch.tensor([[[0., 1., 1.], [0., -1., 1.], [0., 1., -1.], [0., 1., 1.]]])
    pairs = (torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([1, 2]))
    assert margin_loss(scores, pairs, "reference_only", margin=0.5).item() == 0
    assert margin_loss(scores, pairs, "mean_over_tags", margin=0.5).item() == 0.375
    assert margin_loss(scores, pairs, "worst_tag", margin=0.5).item() == 1.5


class ToyPushT:
    def __init__(self, **kwargs):
        self.action_space = SimpleNamespace(low=-1.0, high=1.0)
        self.agent = SimpleNamespace(position=np.zeros(2), velocity=np.zeros(2))
        self.block = SimpleNamespace(
            position=np.zeros(2), velocity=np.zeros(2), angle=0., angular_velocity=0.,
        )

    def reset(self, seed, options):
        state = options["state"]
        self.agent.position = state[:2].copy()
        self.block.position = state[2:4].copy()
        self.block.angle = state[4]
        self.agent.velocity = state[5:7].copy()
        self.block.velocity = np.zeros(2)
        self.block.angular_velocity = 0.

    def render(self):
        intensity = int(self.block.position[0] + 500) % 256
        return np.full((6, 6, 3), intensity, dtype=np.uint8)

    def step(self, action):
        self.agent.position += action
        self.block.position += 10 * action
        self.block.velocity = action.copy()

    def close(self):
        pass


def test_replay_matches_goal_to_the_selected_branch_and_wraps_angle():
    initial = np.zeros(7)
    past = np.zeros((10, 2))
    bank = np.array([[[0.0, 0.0]] * 25, [[0.5, 0.0]] * 25, [[1., 0.0]] * 25], dtype=float)
    frames, distances, start = pilot.replay_candidates(ToyPushT(), initial, past, bank, 1, 74)
    assert frames.shape == (4, 6, 6, 3)
    assert frames[-1, 0, 0, 0] == (125 + 500) % 256
    assert distances.tolist() == [12.5, 0., 12.5]
    assert start == 12.5
    state = initial.copy()
    state[4] = 2 * np.pi - 0.05
    assert pilot.endpoint_distance(state, initial) == pytest.approx(0.5)

    class DriftingReset(ToyPushT):
        def reset(self, seed, options):
            super().reset(seed, options)
            self.resets = getattr(self, "resets", 0) + 1
            self.agent.position[0] = self.resets

    with pytest.raises(AssertionError, match="did not share the realized prefix"):
        pilot.replay_candidates(DriftingReset(), initial, past, bank, 1, 74)


def test_pilot_trains_all_arms_and_reports_disjoint_holdout(tmp_path, monkeypatch):
    class Dataset:
        column_names = ("episode_idx", "step_idx", "action", "state")

        def __init__(self):
            self.lengths = np.repeat(44, 13)
            self.offsets = 44 * np.arange(13)
            self.clip_indices = [(i, start) for i in range(13) for start in range(5)]
            self.columns = {
                "episode_idx": np.repeat(np.arange(13), 44),
                "step_idx": np.tile(np.arange(44), 13),
                "action": np.repeat(np.stack((np.linspace(-.9, .9, 13), np.zeros(13)), axis=1), 44, axis=0),
                "state": np.zeros((13 * 44, 7)),
            }

        def __len__(self):
            return len(self.clip_indices)

        def get_col_data(self, column):
            return self.columns[column]

        def get_dim(self, _):
            return 2

        def __getitem__(self, index):
            ep, offset = self.clip_indices[index]
            start = int(self.offsets[ep]) + offset
            return {"action": self.columns["action"][start:start + 40]}

    swm = SimpleNamespace(data=SimpleNamespace(load_dataset=lambda *a, **k: Dataset()))
    monkeypatch.setitem(sys.modules, "stable_worldmodel", swm)
    monkeypatch.setitem(sys.modules, "stable_worldmodel.envs.pusht.env", SimpleNamespace(PushT=ToyPushT))
    monkeypatch.setitem(sys.modules, "utils", SimpleNamespace(
        get_column_normalizer=lambda *a: lambda item: item,
        get_img_preprocessor=lambda *a, **k: lambda item: {"pixels": item["pixels"].float() / 255},
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pilot, "TRAIN_QUERIES", 4)
    monkeypatch.setattr(pilot, "TEST_QUERIES", 3)
    monkeypatch.setattr(pilot, "CANDIDATES", 3)
    monkeypatch.setattr(pilot, "STEPS", 4)
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_DATASET_DIR", str(tmp_path))

    from additional_files.tests.test_tag_paired_planning import make_model

    def make_packed_model():
        model = make_model()
        model.action_encoder = torch.nn.Linear(10, 16)
        return model

    monkeypatch.setitem(sys.modules, "additional_files.evaluate_reacher_checkpoint", SimpleNamespace(
        load_inference_model=lambda *a: make_packed_model(),
    ))
    checkpoint = tmp_path / "checkpoints/base/weights.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"runs": {"base": {"run_name": "base", "checkpoint": checkpoint.name, "sha256": sha}}}))
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["pilot", "--previous", str(previous), "--out", str(report)])
    pilot.main()
    data = json.loads(report.read_text())
    split = data["protocol"]["split"]
    episodes = [item["episode"] for group in split.values() for item in group]
    assert len(episodes) == len(set(episodes)) == 11
    colors = np.asarray(data["protocol"]["rgb_by_query"]).reshape(-1, 3)
    assert len(np.unique(colors, axis=0)) == 14
    assert data["training_pairs"] > 0 and data["protocol"]["raw_planner_cost_parity_verified"]
    assert data["protocol"]["checkpoint_sha256"] == sha
    variants = data["heldout"]["all_queries"]["variants"]
    assert set(variants) == {"native", *ARMS}
    assert "paired_vs_mean_over_tags" in variants["worst_tag"]["worst_endpoint_regret"]
    assert (tmp_path / "heads.pt").is_file()
