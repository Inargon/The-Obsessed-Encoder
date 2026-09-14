from __future__ import annotations

import torch

from leworldmodel.additional_files.diagnose_generic_tag_geometry import (
    cosine_metrics,
)


def test_cosine_metrics_reports_content_minus_tag_margin():
    own = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    content = own.clone()
    tag = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    passes = {
        "own": {"backbone": own, "projection": own},
        "same_content": {"backbone": content, "projection": content},
        "same_tag": {"backbone": tag, "projection": tag},
        "baseline": {"backbone": tag, "projection": tag},
    }

    result = cosine_metrics(passes)

    for representation in ("backbone", "projection"):
        assert result[representation]["same_content"] == 1.0
        assert result[representation]["same_tag"] == 0.0
        assert result[representation]["content_minus_tag_margin"] == 1.0
