"""Pair-metric determinism and semantics on fixture batches (CPU)."""
import numpy as np
import pytest
import torch

from common.pair_metrics import (
    DISTRIBUTIONS,
    SwapPayloadTransform,
    cyclic_derangement,
    pair_metric_means,
)


def test_cyclic_derangement_properties():
    for n in (2, 5, 64):
        p = cyclic_derangement(n, seed=0)
        assert sorted(p.tolist()) == list(range(n))
        assert not np.any(p == np.arange(n))
    assert np.array_equal(cyclic_derangement(16, 3), cyclic_derangement(16, 3))
    assert not np.array_equal(cyclic_derangement(16, 3), cyclic_derangement(16, 4))


def test_pair_metric_means_deterministic():
    g = torch.Generator().manual_seed(0)
    own = torch.randn(32, 8, generator=g)
    swap = torch.randn(32, 8, generator=g)
    partner = cyclic_derangement(32, seed=1)
    a = pair_metric_means(own, swap, partner)
    b = pair_metric_means(own, swap, partner)
    assert a == b
    assert set(a) == set(DISTRIBUTIONS)


def test_key_dominated_features_invert_the_signature():
    """If features are a function of the key alone, different-input/same-key
    pairs read ~1 and same-input/different-key pairs read ~ the null."""
    n, d = 64, 16
    g = torch.Generator().manual_seed(0)
    keys = torch.randn(n, d, generator=g)
    partner = cyclic_derangement(n, seed=2)
    own = keys
    swap = keys[torch.as_tensor(partner)]  # image j re-rendered with partner's key
    stats = pair_metric_means(own, swap, partner)
    assert stats["diff_content_same_payload"] > 0.95
    assert abs(stats["same_content_diff_payload"] - stats["null"]) < 0.3


def test_content_dominated_features_keep_the_content_reading():
    n, d = 64, 16
    g = torch.Generator().manual_seed(1)
    content = torch.randn(n, d, generator=g)
    partner = cyclic_derangement(n, seed=2)
    stats = pair_metric_means(content, content.clone(), partner)
    assert stats["same_content_diff_payload"] > 0.95
    assert stats["diff_content_same_payload"] < 0.3
    assert stats["null"] < 0.3


def test_pairing_validation():
    own = torch.randn(8, 4)
    with pytest.raises(ValueError):
        pair_metric_means(own, own[:, :2], np.arange(8))
    with pytest.raises(ValueError):
        pair_metric_means(own, own.clone(), np.arange(8))  # fixed points


def test_swap_payload_transform_remaps_only_the_index():
    seen = []

    def bound(img, index):
        seen.append(index)
        return img

    t = SwapPayloadTransform(bound, {3: 7, 7: 3})
    obj = object()
    assert t(obj, 3) is obj
    assert seen == [7]
