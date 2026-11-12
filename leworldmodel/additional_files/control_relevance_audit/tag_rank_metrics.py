"""Two matched colors, identical physical branches; smaller cost is better."""
import json
import sys
from pathlib import Path

import numpy as np


def compare(costs, physical):
    a = np.asarray(costs, dtype=float)
    p = np.asarray(physical, dtype=float)
    if p.ndim != 1 or p.size < 2 or a.shape != (2, p.size) or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Expected two finite cost vectors aligned to physical outcomes")
    i, j = np.triu_indices(p.size, 1)
    margins = a[:, i] - a[:, j]
    usable = ~np.isclose(a[:, i], a[:, j], rtol=1e-6, atol=1e-8).any(axis=0)
    chosen = a.argmin(axis=1)
    return dict(costs=a.tolist(), chosen=chosen.tolist(), regrets=(p[chosen]-p.min()).tolist(),
                flips=int(((margins[0]*margins[1] < 0) & usable).sum()), eligible=int(usable.sum()))


if __name__ == "__main__":
    queries = json.loads(Path(sys.argv[1]).read_text())["queries"]
    rows = [q["tag_ranking"] for q in queries if "tag_ranking" in q]
    if not rows:
        raise ValueError("No eligible decisions")
    for key in ("real", "predicted"):
        group = [row[key] for row in rows]
        print("TAG_RANK_SUMMARY", key, "scenes", len(group), "of", len(queries),
              "pair_flips", sum(r["flips"] for r in group), "of", sum(r["eligible"] for r in group),
              "mean_regret_by_color", np.mean([r["regrets"] for r in group], axis=0).tolist())
