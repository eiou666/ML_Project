import numpy as np

from src.splits import class_count_map, create_explicit_split, create_stratified_split


TARGETS = np.repeat(np.arange(10, dtype=np.int64), 6000)


def test_assignment_budget_counts_and_balance() -> None:
    expected = {
        1.00: (54000, 6000, 0, 5400),
        0.10: (5400, 600, 54000, 540),
        0.05: (2700, 300, 57000, 270),
        0.01: (540, 60, 59400, 54),
    }
    for ratio, (labeled, validation, unlabeled, per_class) in expected.items():
        split = create_stratified_split(TARGETS, ratio, seed=0)
        assert split.labeled.size == labeled
        assert split.validation.size == validation
        assert split.unlabeled.size == unlabeled
        assert set(class_count_map(split.labeled, TARGETS).values()) == {per_class}


def test_annotation_budgets_are_nested_for_same_seed() -> None:
    splits = [create_stratified_split(TARGETS, ratio, seed=3) for ratio in (0.01, 0.05, 0.10, 1.0)]
    for smaller, larger in zip(splits, splits[1:]):
        assert set(smaller.annotated).issubset(set(larger.annotated))


def test_splits_are_reproducible_and_seed_sensitive() -> None:
    first = create_stratified_split(TARGETS, 0.01, seed=7)
    repeated = create_stratified_split(TARGETS, 0.01, seed=7)
    different = create_stratified_split(TARGETS, 0.01, seed=8)
    assert first.split_hash == repeated.split_hash
    assert np.array_equal(first.labeled, repeated.labeled)
    assert first.split_hash != different.split_hash


def test_explicit_active_split_keeps_fixed_validation() -> None:
    initial = create_stratified_split(TARGETS, 0.01, seed=0)
    selected = initial.unlabeled[:600]
    active = create_explicit_split(
        labeled=np.concatenate([initial.labeled, selected]),
        validation=initial.validation,
        universe_size=TARGETS.size,
        seed=0,
        label_ratio=0.02,
    )
    assert active.labeled.size == 1140
    assert active.validation.size == 60
    assert active.unlabeled.size == 58800
    assert np.array_equal(active.validation, initial.validation)

