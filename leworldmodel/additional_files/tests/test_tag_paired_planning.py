import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from additional_files import tag_paired_planning as audit
from jepa import JEPA


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(3, 16)

    def forward(self, pixels, interpolate_pos_encoding):
        values = self.feature(pixels.mean((-1, -2)))
        return SimpleNamespace(last_hidden_state=values[:, None])


class Predictor(nn.Module):
    def forward(self, embedding, action):
        return embedding * 0.75 + action


def make_model():
    return JEPA(Encoder(), Predictor(), nn.Linear(2, 16)).eval()


def test_low_rank_mapping_preserves_the_complement_on_held_out_vectors():
    rng = np.random.default_rng(33)
    axis = rng.normal(size=20)
    axis /= np.linalg.norm(axis)
    reference = rng.normal(size=(32, 20))
    shifted = reference - 2 * (reference @ axis)[:, None] * axis
    alignment = audit.fit_alignment(reference, shifted, rank=1, relative_ridge=1e-8)
    held = rng.normal(size=(20, 20))
    reflected = held - 2 * (held @ axis)[:, None] * axis
    np.testing.assert_allclose(alignment(reflected), held, atol=1e-6)
    delta = alignment(reflected) - reflected
    np.testing.assert_allclose(delta, (delta @ axis)[:, None] * axis, atol=1e-12)


def test_recorded_actions_reach_the_goal_frame_with_correct_observed_prefix():
    recorded = torch.arange(16).reshape(8, 2)
    bank = torch.arange(1, 49).reshape(3, 8, 2) * -1
    plans = audit.query_actions(recorded, bank, candidate_index=2, history=3)
    assert plans.shape == (1, 4, 7, 2)
    torch.testing.assert_close(plans[0, 2], recorded[:-1])
    torch.testing.assert_close(plans[0, :, :2], recorded[:2].expand(4, -1, -1))
    torch.testing.assert_close(plans[0, [0, 1, 3], 2:], bank[:, 2:-1])
    np.testing.assert_array_equal(bank, -np.arange(1, 49).reshape(3, 8, 2))


def test_native_extraction_matches_the_actual_jepa_cost_path():
    class BatchSensitiveEncoder(Encoder):
        def forward(self, pixels, interpolate_pos_encoding):
            result = super().forward(pixels, interpolate_pos_encoding)
            # Reproduce the numerical drift caused by concatenating color requests.
            if len(pixels) > 3:
                result.last_hidden_state = result.last_hidden_state + 0.002
            return result

    model = JEPA(BatchSensitiveEncoder(), Predictor(), nn.Linear(2, 16)).eval()
    frames = torch.rand(2, 8, 3, 4, 4)
    actions = torch.rand(1, 5, 7, 2)
    with torch.inference_mode():
        predictions, goals = audit.terminal_features(model, frames, actions, 3)
        assert predictions.shape == (2, 5, 16)
        assert goals.shape == (2, 16)
        audit.check_cost_parity(model, frames, actions, 3, predictions, goals)


def test_paired_report_preserves_reference_and_penalizes_recorded_ties():
    predictions = np.array([
        [[[0.0], [1.0], [3.0]], [[0.0], [1.0], [1.0]]],
        [[[2.0], [3.0], [5.0]], [[2.0], [3.0], [3.0]]],
    ])
    goals = np.array([[[0.0], [1.0]], [[2.0], [3.0]]])
    recorded = np.array([0, 1])
    raw = lambda values: values
    align = lambda values: values - 2
    report = audit.summarize_variants(
        predictions, goals, predictions[1][:, ::-1], recorded,
        {"raw": (raw, raw), "paired": (align, align)}, seed=73,
    )
    assert report["raw"]["strict_any_tag_flip"]["mean"] == 1
    assert report["paired"]["strict_any_tag_flip"]["mean"] == 0
    assert report["paired"]["worst_recorded_rank"]["mean"] == 1
    assert report["paired"]["recorded_strictly_best_in_all_views"]["mean"] == 0.5
    assert report["paired"]["changed_tag_action_order_flip"]["mean"] > 0


def test_frozen_worker_writes_auditable_episode_split_and_checkpoint_hash(tmp_path, monkeypatch):
    class Dataset:
        column_names = ("episode_idx", "step_idx", "action")

        def __init__(self):
            self.lengths = [8] * 25
            self.offsets = 8 * np.arange(25)
            self.clip_indices = [(episode, i) for episode in range(25) for i in range(3)]
            rng = np.random.default_rng(19)
            self.columns = {
                "episode_idx": np.repeat(np.arange(25), 8),
                "step_idx": np.tile(np.arange(8), 25),
                "action": rng.normal(size=(200, 2)).astype(np.float32),
                "pixels": rng.integers(256, size=(200, 3, 4, 4), dtype=np.uint8),
            }

        def __len__(self):
            return len(self.clip_indices)

        def get_col_data(self, column):
            return self.columns[column]

        def get_dim(self, _):
            return 2

        def __getitem__(self, index):
            episode, start = self.clip_indices[index]
            row = self.offsets[episode] + start
            return {key: torch.from_numpy(self.columns[key][row:row + 6]) for key in ("pixels", "action")}

    dataset = Dataset()
    monkeypatch.setitem(sys.modules, "stable_worldmodel",
                        SimpleNamespace(data=SimpleNamespace(load_dataset=lambda *a, **kw: dataset)))
    monkeypatch.setitem(sys.modules, "additional_files.evaluate_reacher_checkpoint",
                        SimpleNamespace(load_inference_model=lambda *a: make_model()))
    monkeypatch.setitem(sys.modules, "utils", SimpleNamespace(
        get_img_preprocessor=lambda *a, **k: lambda item: {"pixels": item["pixels"].float() / 255},
        get_column_normalizer=lambda *a: lambda item: item,
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("LOCAL_DATASET_DIR", str(tmp_path))
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    checkpoint = tmp_path / "checkpoints/run/weights.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen")
    sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "protocol": {"indices": [1, 2, 3, 4], "candidates": 4, "seed": 73,
                     "history": 3, "horizon": 3, "frameskip": 1, "tag_size": 1},
        "runs": {"fixture": {"run_name": "run", "checkpoint": "weights.pt", "sha256": sha256}},
    }))
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "audit", "--previous", str(previous), "--out", str(output),
        "--dataset", "pusht_expert_train.h5", "--anchors", "17",
    ])
    audit.main()
    report = json.loads(output.read_text())
    protocol = report["protocol"]
    groups = [protocol[key] for key in ("calibration", "queries", "candidate_bank")]
    assert [len(group) for group in groups] == [17, 4, 3]
    assert len({row["episode"] for group in groups for row in group}) == 24
    run = report["runs"]["fixture"]
    assert run["checkpoint_sha256"] == sha256 and run["raw_cost_parity_checked"]
    assert set(run["variants"]) == {
        "raw", "mean_shift", "diagonal_moments", "paired_rank_1",
        "paired_rank_4", "paired_rank_16", "shuffled_rank_4",
    }
    assert all(len(row["per_episode"]["strict_any_tag_flip"]) == 4 for row in run["variants"].values())
