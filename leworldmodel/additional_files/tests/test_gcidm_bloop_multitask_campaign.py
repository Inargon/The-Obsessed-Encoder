from pathlib import Path

from additional_files.gcidm_bloop_multitask_campaign import TASKS, parse_tasks


def test_default_followup_tasks_have_protocol_specs() -> None:
    assert set(TASKS) == {"tworoom", "reacher", "clean_pusht"}
    assert TASKS["tworoom"]["action_dim"] == 2
    assert TASKS["clean_pusht"]["action_dim"] == 2
    assert TASKS["tworoom"]["official_dataset"] == "tworoom"
    assert TASKS["clean_pusht"]["official_dataset"] == "pusht"
    assert TASKS["reacher"]["official_dataset"] == "reacher"
    assert TASKS["reacher"]["runtime"] == "historical"
    assert Path(TASKS["tworoom"]["dataset"]).name == "tworoom.h5"


def test_task_parser_deduplicates_without_reordering() -> None:
    assert parse_tasks("clean_pusht,tworoom,clean_pusht") == [
        "clean_pusht",
        "tworoom",
    ]
