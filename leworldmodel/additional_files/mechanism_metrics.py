"""Pure NumPy summaries for paired interventions (no training dependencies)."""
import numpy as np


def normalize_action_blocks(actions, normalizer, action_dim):
    """Normalize native actions before restoring their frameskip-packed shape.

    Works with tensors or arrays; the supplied training transform owns dtype
    and statistics. Packing order is unchanged.
    """
    shape = actions.shape
    if len(shape) < 2 or action_dim < 1 or shape[-1] % action_dim:
        raise ValueError("Packed action width must be a multiple of native action_dim")
    unpacked = actions.reshape(-1, action_dim)
    normalized = normalizer({"action": unpacked})["action"]
    if normalized.shape != unpacked.shape:
        raise ValueError("Action normalizer unexpectedly changed shape")
    return normalized.reshape(shape)


def summarize(values, seed=0, repeats=1000):
    x = np.asarray(values, dtype=float).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected nonempty, finite paired measurements")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(repeats, len(x)))].mean(1)
    return dict(n=len(x), mean=float(x.mean()),
                bootstrap_clip_ci95=np.quantile(means, [.025, .975]).tolist())


def cost_changes(reference, changed, first_actions):
    """Per-clip candidate-cost and selection changes. Not closed-loop SR."""
    a, b = np.asarray(reference), np.asarray(changed)
    actions = np.asarray(first_actions)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("Costs must be matched (clips, candidates), candidates >= 2")
    if actions.shape[:2] != a.shape or not all(np.isfinite(x).all() for x in (a,b,actions)):
        raise ValueError("Mismatched or nonfinite candidate arrays")
    left, right = np.triu_indices(a.shape[1], 1)
    da, db = a[:, left]-a[:, right], b[:, left]-b[:, right]
    informative = (da != 0) & (db != 0)
    denom = informative.sum(1)
    flips = ((da * db < 0) & informative).sum(1) / np.maximum(denom, 1)
    ia, ib = a.argmin(1), b.argmin(1)
    rows = np.arange(len(a))
    scale = a.std(1)
    return {
        "cost_rmse": np.sqrt(np.mean((a-b)**2, axis=1)),
        "reference_cost_std": scale,
        "cost_rmse_over_reference_std": np.sqrt(np.mean((a-b)**2, axis=1))/np.maximum(scale, 1e-12),
        "reference_cost_degenerate": (scale <= 1e-12).astype(float),
        "pair_order_reversal": flips,
        "comparable_pair_fraction": denom / len(left),
        "selected_candidate_changed": (ia != ib).astype(float),
        "selected_action_normalized_l2": np.linalg.norm(actions[rows, ia]-actions[rows, ib], axis=-1),
    }
