#!/usr/bin/env python3
"""Audit a results tree's metrics.jsonl streams before figures render from it.

Per run: summary presence, per-metric duplicate steps (expected only after a
native resume; figures supersede them keep-last), and non-monotone step order.
Per tree: each metric's final step across runs (unequal spans mean the bands
cover the overlap only).

    python tools/check_metrics_integrity.py results/lejepa results/dinov3 ...

Exit code 1 when any run is summary-less or non-monotone; duplicates alone
(a resume artifact) exit 0 and are only reported.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import results_io  # noqa: E402


def audit_tree(results_dir: str) -> bool:
    ok = True
    finals: dict[str, dict[str, int]] = defaultdict(dict)
    for run_dir in results_io.iter_run_dirs(results_dir):
        name = os.path.basename(run_dir)
        if not results_io.has_summary(run_dir):
            print(f"FAIL {name}: no summary.json (incomplete run)")
            ok = False
        last_step: dict[str, int] = {}
        dups: Counter = Counter()
        backwards = 0
        for payload in results_io.read_metrics(run_dir):
            step = payload.get("step")
            if step is None:
                continue
            for key in payload:
                if key == "step":
                    continue
                prev = last_step.get(key)
                if prev is not None:
                    if step == prev:
                        dups[key] += 1
                    elif step < prev:
                        backwards += 1
                        dups[key] += 1  # a resume rewinds, then re-covers
                last_step[key] = max(step, prev or 0)
            finals[name].update({k: last_step[k] for k in last_step})
        if dups:
            total = sum(dups.values())
            print(f"note {name}: {total} duplicate/rewound step rows across "
                  f"{len(dups)} metrics (resume artifact; figures keep last)")
        if backwards and not results_io.has_summary(run_dir):
            ok = False

    by_metric: dict[str, set[int]] = defaultdict(set)
    for name, metrics in finals.items():
        for key, final in metrics.items():
            by_metric[key].add(final)
    uneven = {k: sorted(v) for k, v in by_metric.items() if len(v) > 1}
    for key, spans in sorted(uneven.items()):
        print(f"note {results_dir}: {key!r} final steps differ across runs: "
              f"{spans[0]}..{spans[-1]}")
    print(f"{'OK  ' if ok else 'FAIL'} {results_dir}: "
          f"{len(finals)} runs audited")
    return ok


def main() -> int:
    trees = sys.argv[1:]
    if not trees:
        raise SystemExit(__doc__)
    return 0 if all([audit_tree(t) for t in trees]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
