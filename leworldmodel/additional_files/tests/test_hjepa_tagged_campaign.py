from pathlib import Path

from additional_files import hjepa_tagged_pusht_campaign as campaign


def test_campaign_pins_released_hjepa_commit():
    assert campaign.PINNED_HJEPA_COMMIT == (
        "fd4f7de927daa7831a6d51e95e0959a813c13901"
    )


def test_campaign_uses_local_adapter_and_official_upstream():
    assert campaign.ADAPTER.name == "hjepa_tagged_adapter.py"
    assert campaign.UPSTREAM_URL == "https://github.com/tmz-lab/hjepa.git"
    assert Path(campaign.ADAPTER).is_file()
