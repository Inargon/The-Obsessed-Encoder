from additional_files.gcidm_cube_campaign import (
    DEFAULT_STABLEWM,
    DATASET_RELATIVE,
    HDF5_BOOTSTRAP,
    make_inference_state_dict,
    parse_eval_summary,
)
from additional_files.gcidm_matched_eval import sample_matched_eval_episodes


def test_inference_bundle_drops_only_training_components() -> None:
    state = {
        "encoder.weight": 1,
        "projector.weight": 2,
        "control_objective.head.weight": 3,
        "bloop_router.ema": 4,
    }
    inference, removed = make_inference_state_dict(state)
    assert set(inference) == {"encoder.weight", "projector.weight"}
    assert set(removed) == {
        "control_objective.head.weight",
        "bloop_router.ema",
    }


def test_parse_official_eval_summary() -> None:
    text = """
IDM metrics: {'success_rate': 88.0, 'episode_successes': array([ True ])}
IDM time: 12.5s total, 250 ms/episode
IDM avg plan time: 3.25 ms
"""
    result = parse_eval_summary(text)
    assert result["success_rate"] == 0.88
    assert result["idm_plan_ms"] == 3.25
    assert result["idm_ms_per_episode"] == 250.0
    assert result["bloop_cem_reference_success_rate"] == 0.80


class _Dataset:
    def __init__(self):
        self.columns = {
            "episode_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "step_idx": [0, 1, 2, 3, 0, 1, 2, 3],
        }

    def get_col_data(self, name):
        if name not in self.columns:
            raise KeyError(name)
        return self.columns[name]


def test_matched_sampler_includes_full_valid_population() -> None:
    episodes, starts = sample_matched_eval_episodes(
        _Dataset(), num_eval=6, goal_offset=1, seed=42
    )
    assert sorted(zip(episodes, starts)) == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
    ]
    assert hasattr(episodes, "tolist")
    assert hasattr(starts, "tolist")


def test_cube_dataset_location_is_under_stablewm_datasets() -> None:
    assert DEFAULT_STABLEWM / DATASET_RELATIVE == (
        DEFAULT_STABLEWM / "datasets/ogbench/cube_single_expert.h5"
    )


def test_hdf5_plugin_is_registered_before_official_extractor() -> None:
    assert HDF5_BOOTSTRAP.startswith("import hdf5plugin,")
    assert "runpy.run_path" in HDF5_BOOTSTRAP
