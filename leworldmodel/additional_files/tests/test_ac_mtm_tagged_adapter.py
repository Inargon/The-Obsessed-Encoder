from types import SimpleNamespace

from additional_files.ac_mtm_tagged_adapter import install_training_tag


def test_ac_mtm_training_adapter_tags_direct_hdf5_constructor(monkeypatch):
    captured = {}
    dataset = object()
    module = SimpleNamespace(
        swm=SimpleNamespace(
            data=SimpleNamespace(HDF5Dataset=lambda *args, **kwargs: dataset)
        )
    )

    def fake_attach(value, tag):
        captured["dataset"] = value
        captured["tag"] = tag
        return "tagged"

    monkeypatch.setattr("pixel_tag.attach_pixel_tag", fake_attach)
    install_training_tag(module, mode="video", size=5, seed=11)
    assert module.swm.data.HDF5Dataset(path="ignored") == "tagged"
    assert captured["dataset"] is dataset
    assert captured["tag"].mode == "video"
    assert captured["tag"].size == 5
    assert captured["tag"].seed == 11
