from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus
from torch import nn
from torch.utils.data import DataLoader, Dataset


class PoolDataset(Dataset):
    """Selection view deliberately drops hidden labels before collation."""

    def __init__(self, base: Dataset, indices: np.ndarray, transform: Callable[[Any], torch.Tensor]):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        index = int(self.indices[position])
        image, _ = self.base[index]
        return self.transform(image), index


@torch.inference_mode()
def infer_pool(
    model: nn.Module,
    *,
    base_dataset: Dataset,
    indices: np.ndarray,
    transform: Callable[[Any], torch.Tensor],
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
    return_features: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    dataset = PoolDataset(base_dataset, indices, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for images, batch_indices in loader:
        images = images.to(device, non_blocking=True)
        if return_features:
            logits, features = model(images, return_features=True)
            feature_parts.append(features.cpu().numpy())
        else:
            logits = model(images)
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
        index_parts.append(batch_indices.numpy())
    features_array = np.concatenate(feature_parts) if return_features else None
    return np.concatenate(probabilities), features_array, np.concatenate(index_parts)


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    return -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)


def select_from_inference(
    *,
    strategy: str,
    pool_indices: np.ndarray,
    probabilities: np.ndarray,
    features: np.ndarray | None,
    query_size: int,
    candidate_multiplier: int,
    seed: int,
) -> np.ndarray:
    pool = np.asarray(pool_indices, dtype=np.int64)
    if query_size <= 0 or query_size > pool.size:
        raise ValueError("query_size must be positive and no larger than the pool")
    entropy = predictive_entropy(probabilities)
    if strategy == "entropy":
        positions = np.argsort(-entropy, kind="stable")[:query_size]
        return pool[positions]
    if strategy != "uncertainty_diversity":
        raise ValueError(f"Inference-based selection does not support {strategy!r}")
    if features is None:
        raise ValueError("uncertainty_diversity requires feature embeddings")

    candidate_count = min(pool.size, max(query_size, candidate_multiplier * query_size))
    candidate_positions = np.argsort(-entropy, kind="stable")[:candidate_count]
    candidate_features = np.asarray(features[candidate_positions], dtype=np.float64)
    norms = np.linalg.norm(candidate_features, axis=1, keepdims=True)
    candidate_features = candidate_features / np.clip(norms, 1e-12, None)
    _, selected_within_candidates = kmeans_plusplus(
        candidate_features,
        n_clusters=query_size,
        random_state=seed,
    )
    return pool[candidate_positions[np.asarray(selected_within_candidates, dtype=np.int64)]]


def select_batch(
    model: nn.Module,
    *,
    strategy: str,
    base_dataset: Dataset,
    pool_indices: np.ndarray,
    transform: Callable[[Any], torch.Tensor],
    device: torch.device,
    query_size: int,
    candidate_multiplier: int,
    seed: int,
    num_workers: int = 0,
) -> np.ndarray:
    pool = np.asarray(pool_indices, dtype=np.int64)
    if query_size > pool.size:
        raise ValueError("query_size cannot exceed the unlabeled pool")
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(pool, size=query_size, replace=False).astype(np.int64))
    need_features = strategy == "uncertainty_diversity"
    probabilities, features, ordered_indices = infer_pool(
        model,
        base_dataset=base_dataset,
        indices=pool,
        transform=transform,
        device=device,
        num_workers=num_workers,
        return_features=need_features,
    )
    selected = select_from_inference(
        strategy=strategy,
        pool_indices=ordered_indices,
        probabilities=probabilities,
        features=features,
        query_size=query_size,
        candidate_multiplier=candidate_multiplier,
        seed=seed,
    )
    return np.sort(selected.astype(np.int64))

