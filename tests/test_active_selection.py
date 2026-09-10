import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from src.active_learning.selection import predictive_entropy, select_from_inference


def test_entropy_prefers_uncertain_probabilities() -> None:
    probabilities = np.array([[0.99, 0.01], [0.50, 0.50], [0.60, 0.40]])
    entropy = predictive_entropy(probabilities)
    assert entropy[1] > entropy[2] > entropy[0]
    selected = select_from_inference(
        strategy="entropy",
        pool_indices=np.array([10, 11, 12]),
        probabilities=probabilities,
        features=None,
        query_size=1,
        candidate_multiplier=2,
        seed=0,
    )
    assert selected.tolist() == [11]


def test_uncertainty_diversity_returns_unique_pool_members() -> None:
    rng = np.random.default_rng(3)
    probabilities = rng.dirichlet(np.ones(3), size=30)
    features = rng.normal(size=(30, 8))
    pool = np.arange(100, 130)
    selected = select_from_inference(
        strategy="uncertainty_diversity",
        pool_indices=pool,
        probabilities=probabilities,
        features=features,
        query_size=5,
        candidate_multiplier=3,
        seed=4,
    )
    assert len(selected) == len(set(selected.tolist())) == 5
    assert set(selected).issubset(set(pool))
