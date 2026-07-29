"""Label pass-through collate for the online probe.

``collate_data_and_cast`` drops the per-sample label (the SSL loss never needs
it).  The online probe does, so this thin wrapper appends ``collated_labels``
and forwards everything else verbatim.  ImageNet1kParquet always returns an int
label; the wrapper is only wired in when the observer is on (``DINOV3_PROBE=1``
in build_data_loader_from_cfg).
"""
from __future__ import annotations

import torch

from dinov3.data.collate import collate_data_and_cast


def collate_data_and_cast_with_labels(samples_list, **kwargs):
    out = collate_data_and_cast(samples_list, **kwargs)
    out["collated_labels"] = torch.tensor([int(s[1]) for s in samples_list], dtype=torch.long)
    return out
