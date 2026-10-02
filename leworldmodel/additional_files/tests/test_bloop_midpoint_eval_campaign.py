from pathlib import Path

from additional_files.bloop_midpoint_eval_campaign import run_names


def test_midpoint_run_names_match_bloop_training_contract() -> None:
    names = run_names(Path("interface-cycle-20261002-064431"), 0)

    assert names == {
        "tagged_pusht": (
            "interface-cycle-20261002-064431_"
            "tagged_pusht_bloop_cycle_full_seed0"
        ),
        "clean_reacher": (
            "interface-cycle-20261002-064431_"
            "clean_reacher_bloop_cycle_full_seed0"
        ),
    }
