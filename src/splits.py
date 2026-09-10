from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    labeled: np.ndarray
    validation: np.ndarray
    unlabeled: np.ndarray
    annotated: np.ndarray
    seed: int
    label_ratio: float
    split_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "label_ratio": self.label_ratio,
            "split_hash": self.split_hash,
            "counts": {
                "labeled": int(self.labeled.size),
                "validation": int(self.validation.size),
                "unlabeled": int(self.unlabeled.size),
                "annotated": int(self.annotated.size),
            },
            "labeled": self.labeled.tolist(),
            "validation": self.validation.tolist(),
            "unlabeled": self.unlabeled.tolist(),
            "annotated": self.annotated.tolist(),
        }


def _hash_arrays(*arrays: np.ndarray, metadata: str = "") -> str:
    digest = hashlib.sha256(metadata.encode("utf-8"))
    for array in arrays:
        normalized = np.asarray(array, dtype=np.int64)
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def create_stratified_split(
    targets: np.ndarray,
    label_ratio: float,
    seed: int,
    validation_fraction: float = 0.10,
) -> SplitIndices:
    """Create a class-balanced annotated budget with a held-out validation slice.

    The annotated set is a per-class prefix of a deterministic permutation. This
    makes the *annotation budgets* nested across 1%, 5%, 10%, and 100% for a
    fixed seed. Validation labels count against the annotation budget.
    """

    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array")
    if not 0.0 < label_ratio <= 1.0:
        raise ValueError("label_ratio must be in (0, 1]")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")

    labeled_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    annotated_parts: list[np.ndarray] = []
    unlabeled_parts: list[np.ndarray] = []

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        class_rng = np.random.default_rng(seed + int(class_id) * 1009)
        permutation = class_rng.permutation(class_indices)
        budget = int(round(class_indices.size * label_ratio))
        budget = min(class_indices.size, max(2, budget))
        validation_count = int(round(budget * validation_fraction))
        validation_count = min(budget - 1, max(1, validation_count))
        labeled_count = budget - validation_count

        annotated = permutation[:budget]
        labeled_parts.append(annotated[:labeled_count])
        validation_parts.append(annotated[labeled_count:])
        annotated_parts.append(annotated)
        unlabeled_parts.append(permutation[budget:])

    labeled = np.sort(np.concatenate(labeled_parts)).astype(np.int64)
    validation = np.sort(np.concatenate(validation_parts)).astype(np.int64)
    annotated = np.sort(np.concatenate(annotated_parts)).astype(np.int64)
    unlabeled = np.sort(np.concatenate(unlabeled_parts)).astype(np.int64)
    split_hash = _hash_arrays(
        labeled,
        validation,
        unlabeled,
        metadata=f"seed={seed};ratio={label_ratio:.12g};val={validation_fraction:.12g}",
    )
    split = SplitIndices(
        labeled=labeled,
        validation=validation,
        unlabeled=unlabeled,
        annotated=annotated,
        seed=int(seed),
        label_ratio=float(label_ratio),
        split_hash=split_hash,
    )
    assert_split_invariants(split, labels.size)
    return split


def create_explicit_split(
    *,
    labeled: np.ndarray,
    validation: np.ndarray,
    universe_size: int,
    seed: int,
    label_ratio: float,
) -> SplitIndices:
    labeled_array = np.unique(np.asarray(labeled, dtype=np.int64))
    validation_array = np.unique(np.asarray(validation, dtype=np.int64))
    annotated = np.sort(np.concatenate([labeled_array, validation_array]))
    universe = np.arange(universe_size, dtype=np.int64)
    unlabeled = np.setdiff1d(universe, annotated, assume_unique=True)
    split_hash = _hash_arrays(
        labeled_array,
        validation_array,
        unlabeled,
        metadata=f"explicit;seed={seed};ratio={label_ratio:.12g}",
    )
    split = SplitIndices(
        labeled=labeled_array,
        validation=validation_array,
        unlabeled=unlabeled,
        annotated=annotated,
        seed=int(seed),
        label_ratio=float(label_ratio),
        split_hash=split_hash,
    )
    assert_split_invariants(split, universe_size)
    return split


def assert_split_invariants(split: SplitIndices, universe_size: int) -> None:
    labeled = set(split.labeled.tolist())
    validation = set(split.validation.tolist())
    unlabeled = set(split.unlabeled.tolist())
    annotated = set(split.annotated.tolist())
    if labeled & validation or labeled & unlabeled or validation & unlabeled:
        raise AssertionError("labeled, validation, and unlabeled sets must be disjoint")
    if annotated != labeled | validation:
        raise AssertionError("annotated must equal labeled union validation")
    if annotated | unlabeled != set(range(universe_size)):
        raise AssertionError("split must cover the complete training universe")


def class_count_map(indices: np.ndarray, targets: np.ndarray) -> dict[int, int]:
    labels = np.asarray(targets, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    classes, counts = np.unique(labels, return_counts=True)
    return {int(class_id): int(count) for class_id, count in zip(classes, counts)}

