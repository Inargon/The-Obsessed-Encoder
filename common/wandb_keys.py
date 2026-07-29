"""One metric-naming scheme for the wandb stream across all three examples.

Each example inherits its upstream's metric names (``train/lejepa``,
``ssl/total_loss``, ``fit/pred_loss``, ...), which makes a combined wandb
workspace hard to read: the same semantic track wears a different section
name per example.  The remap below translates keys at each example's wandb
seam into four shared sections:

    objective/*   what training optimizes and reports as healthy
                  (``objective/total`` exists in every example, per-term
                  children where the recipe has them)
    downstream/*  the readouts the blog tracks failing (probe top-1,
                  NYU-depth RMSE, planning success rate)
    pair/*        the swap-pair cosine distributions (already emitted under
                  one scheme by construction -- passed through)
    sched/*       schedules and progress housekeeping (lr, wd, epoch)

Only the wandb stream is renamed.  The local ``metrics.jsonl`` mirrors keep
the original upstream keys: they are the source of truth the figures render
from, and renaming them would orphan existing runs from ``--plot-only``.
"""

from typing import Dict, Mapping, Optional

CANONICAL_PREFIXES = ("objective/", "downstream/", "pair/", "sched/", "media/")

LEJEPA = {
    "train/lejepa": "objective/total",
    "train/sigreg": "objective/sigreg",
    "train/inv": "objective/inv",
    "train/probe": "downstream/probe_loss",
    "test/acc": "downstream/top1",
    "test/epoch": "sched/epoch",
}

# Exact keys first; every other ``ssl/<term>`` (the upstream per-term loss
# dict) falls under objective/ via the prefix rule.
DINOV3 = {
    "ssl/total_loss": "objective/total",
    "probe/val_top1": "downstream/top1",
    "probe/train_loss": "downstream/probe_loss",
    "probe/train_acc": "downstream/probe_acc",
}
DINOV3_PREFIXES = {"ssl/": "objective/"}

LEWM = {
    # The world model's objective is the single prediction term.
    "fit/pred_loss": "objective/total",
    "eval/success_rate": "downstream/success_rate",
}


def remap(payload: Mapping[str, object], exact: Mapping[str, str],
          prefixes: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """Translate payload keys via the exact map, then the prefix map; keys
    matching neither pass through unchanged."""
    out: Dict[str, object] = {}
    for key, value in payload.items():
        if key in exact:
            out[exact[key]] = value
            continue
        for old, new in (prefixes or {}).items():
            if key.startswith(old):
                out[new + key[len(old):]] = value
                break
        else:
            out[key] = value
    return out
