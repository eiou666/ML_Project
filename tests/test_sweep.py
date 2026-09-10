from src.sweep import build_jobs


def test_suite_sizes_match_research_matrix() -> None:
    assert len(build_jobs("main", "configs/main.yaml", [0, 1, 2, 3, 4], None)) == 50
    assert len(build_jobs("ablation", "configs/main.yaml", [0, 1, 2], None)) == 9
    assert len(build_jobs("active_learning", "configs/main.yaml", [0, 1, 2], None)) == 3
    assert len(build_jobs("smoke", "configs/smoke.yaml", [0], None)) == 3

