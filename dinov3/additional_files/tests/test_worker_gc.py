"""The worker-GC guard: dataloader workers forked from a gc-disabled parent
(do_train's interpreter state) must run with the collector re-enabled.

Without the guard, workers inherit the disabled collector and accumulate
per-item reference cycles for the whole run (measured at ~6 GiB of worker RSS
per 1000 batches of 128 on the shipped item path).
"""
import gc

import pytest
import torch


class _GcProbeDataset(torch.utils.data.Dataset):
    """Each item reports whether the cyclic collector is enabled in the worker."""

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return int(gc.isenabled())


@pytest.fixture
def gc_disabled_parent():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _worker_gc_states(worker_init_fn):
    loader = torch.utils.data.DataLoader(
        _GcProbeDataset(), batch_size=1, num_workers=2,
        worker_init_fn=worker_init_fn,
        multiprocessing_context="fork",
    )
    return {int(x) for batch in loader for x in batch}


def test_forked_workers_inherit_disabled_gc(gc_disabled_parent):
    """The failure mode being guarded: fork propagates the disabled collector."""
    assert _worker_gc_states(None) == {0}


def test_train_loader_guard_reenables_gc(gc_disabled_parent):
    from dinov3.train.train import _enable_gc_in_workers

    assert _worker_gc_states(_enable_gc_in_workers) == {1}


def test_observer_val_loader_guard_reenables_gc(gc_disabled_parent):
    from additional_files.train_hooks import _enable_gc_in_workers

    assert _worker_gc_states(_enable_gc_in_workers) == {1}


def test_train_loader_construction_wires_the_guard():
    """The marked block in build_data_loader_from_cfg must pass the guard to
    make_data_loader; checked on the call's source so a silent drop of the
    kwarg fails loudly."""
    import inspect

    from dinov3.train import train as vendored_train

    source = inspect.getsource(vendored_train.build_data_loader_from_cfg)
    assert "worker_init_fn=(_enable_gc_in_workers" in source
    # ... and only when the observer is on, so the default invocation keeps the
    # upstream loader construction bit for bit.
    assert 'os.environ.get("DINOV3_PROBE") == "1" else None' in source