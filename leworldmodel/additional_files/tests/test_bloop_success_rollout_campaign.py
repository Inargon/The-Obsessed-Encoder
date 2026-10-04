from pathlib import Path

from additional_files.bloop_success_rollout_campaign import TASKS, task_command


def test_rollout_matrix_covers_main_modern_tasks():
    assert list(TASKS) == ["clean_pusht", "tagged_pusht", "tworoom", "cube"]


def test_tagged_pusht_command_records_fixed_group_video():
    command = task_command("tagged_pusht", Path("campaign"), Path("python"))
    assert "--tagged" in command
    assert "--num-eval 50" in command
    assert "--seed 42" in command
    assert "--video-dir" in command
    assert "tagged_pusht" in command
    assert "all-rollouts" in command
