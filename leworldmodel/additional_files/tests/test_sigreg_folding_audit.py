import numpy as np
import torch

from leworldmodel.additional_files.sigreg_folding_audit import audit_tag_sweeps, synthetic_folding
from leworldmodel.module import SIGReg


def test_training_forward_and_gradient_match_original_statistic():
    torch.manual_seed(9)
    embeddings = torch.randn(3, 13, 5, requires_grad=True)
    regularizer = SIGReg(num_proj=7)
    rng_state = torch.random.get_rng_state()
    actual = regularizer(embeddings)
    torch.random.set_rng_state(rng_state)
    directions = torch.randn(5, 7)
    directions.div_(directions.norm(p=2, dim=0))
    phase = (embeddings @ directions).unsqueeze(-1) * regularizer.t
    error = (phase.cos().mean(-3) - regularizer.phi).square() + phase.sin().mean(-3).square()
    expected = ((error @ regularizer.weights) * embeddings.size(-2)).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, embeddings)[0],
        torch.autograd.grad(expected, embeddings)[0],
    )


def test_folding_leaves_sigreg_unchanged_but_flips_candidate_rank():
    row, = synthetic_folding([16], samples=128, fold=7, projections=24, seed=72, repeats=3)["runs"]
    base, folded = row["base"], row["folded"]
    np.testing.assert_allclose(row["covariance_participation_ratio"], 16)
    np.testing.assert_allclose(folded["path_length"], 7 * base["path_length"])
    for key in ("random_projection_mean", "random_projection_p95", "axis_projection_max"):
        np.testing.assert_allclose(base[key], folded[key], rtol=1e-6)
    assert folded["squared_norm_variance_over_gaussian"] < 1e-10
    assert base["candidate_ranking"]["chosen"] == [0, 0]
    assert folded["candidate_ranking"]["chosen"] == [0, 1]
    assert folded["candidate_ranking"]["regrets"] == [0.0, 1.0]


def test_raw_audit_uses_matched_reference_and_preserves_mean_shift():
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(2, 96, 8))
    report = audit_tag_sweeps(
        {"reference": reference, "shifted": reference + 3},
        projections=24, seed=7, repeats=4,
    )
    before = report["runs"]["reference"]
    after = report["runs"]["shifted"]
    key = "gaussian_reference_for_random_projection_mean"
    assert before[key] == after[key]
    assert before[key]["draws"] == 4
    assert len(before["scenes"]) == 2
    assert after["aggregate_mean"]["random_projection_mean"] > before["aggregate_mean"]["random_projection_mean"]
