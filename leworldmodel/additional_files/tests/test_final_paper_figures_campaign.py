from pathlib import Path


def test_campaign_is_parallel_and_excludes_gcidm():
    path = Path(__file__).resolve().parents[1] / "final_paper_figures_campaign.py"
    text = path.read_text(encoding="utf-8")
    assert '"--array=0-2%3"' in text
    assert "GC-IDM is excluded" in text
    assert "dense-example" not in text or "render_dense_selectivity_magnitude.py" in text
    assert "afterok:" in text
