from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .splits import SplitIndices, create_stratified_split


class SyntheticFashionDataset(Dataset):
    """Small, balanced image dataset for offline end-to-end smoke tests."""

    def __init__(self, size: int, seed: int) -> None:
        if size < 20 or size % 10:
            raise ValueError("Synthetic dataset size must be >=20 and divisible by 10")
        rng = np.random.default_rng(seed)
        targets = np.arange(size, dtype=np.int64) % 10
        data = np.zeros((size, 28, 28), dtype=np.uint8)
        for index, label in enumerate(targets):
            image = rng.normal(loc=20, scale=10, size=(28, 28))
            row = 2 + (int(label) // 5) * 12
            column = 2 + (int(label) % 5) * 5
            image[row : row + 10, column : column + 4] += 180
            image[4 + int(label) : 7 + int(label), 4:24] += 70
            data[index] = np.clip(image, 0, 255).astype(np.uint8)
        self.data = torch.from_numpy(data)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        return Image.fromarray(self.data[index].numpy(), mode="L"), int(self.targets[index])


class LabeledDataset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray, transform: Callable[[Any], torch.Tensor]):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        index = int(self.indices[position])
        image, target = self.base[index]
        return self.transform(image), int(target), index


class UnlabeledDataset(Dataset):
    """Returns image views and indices, never hidden class labels."""

    def __init__(
        self,
        base: Dataset,
        indices: np.ndarray,
        weak_transform: Callable[[Any], torch.Tensor],
        target_transform: Callable[[Any], torch.Tensor],
    ) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.weak_transform = weak_transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[position])
        image, _ = self.base[index]
        return self.weak_transform(image), self.target_transform(image), index


class EvaluationDataset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray, transform: Callable[[Any], torch.Tensor]):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        index = int(self.indices[position])
        image, target = self.base[index]
        return self.transform(image), int(target), index


@dataclass
class DataBundle:
    train_base: Dataset
    test_base: Dataset
    split: SplitIndices
    labeled_loader: DataLoader
    unlabeled_loader: DataLoader | None
    validation_loader: DataLoader
    test_loader: DataLoader
    evaluation_transform: Callable[[Any], torch.Tensor]
    diagnostic_targets: torch.Tensor
    mean: float
    std: float


def dataset_targets(dataset: Dataset) -> np.ndarray:
    values = getattr(dataset, "targets", None)
    if values is None:
        raise AttributeError("Dataset must expose a targets attribute")
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.int64)
    return np.asarray(values, dtype=np.int64)


def build_base_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    data_config = config["data"]
    name = str(data_config.get("name", "fashion_mnist")).lower()
    if name == "fashion_mnist":
        root = Path(data_config["root"])
        download = bool(data_config.get("download", True))
        return (
            datasets.FashionMNIST(root=root, train=True, download=download),
            datasets.FashionMNIST(root=root, train=False, download=download),
        )
    if name == "synthetic":
        return (
            SyntheticFashionDataset(int(data_config["synthetic_train_size"]), seed=1234),
            SyntheticFashionDataset(int(data_config["synthetic_test_size"]), seed=5678),
        )
    raise ValueError(f"Unsupported dataset: {name}")


def normalization_stats(dataset: Dataset) -> tuple[float, float]:
    data = getattr(dataset, "data", None)
    if data is None:
        raise AttributeError("Dataset must expose raw data for normalization")
    tensor = torch.as_tensor(data, dtype=torch.float32).div_(255.0)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item())
    if std <= 0:
        raise ValueError("Dataset standard deviation must be positive")
    return mean, std


def build_transforms(mean: float, std: float) -> tuple[Callable, Callable, Callable]:
    normalize = transforms.Normalize((mean,), (std,))
    weak = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    strong = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), value="random"),
            normalize,
        ]
    )
    evaluation = transforms.Compose([transforms.ToTensor(), normalize])
    return weak, strong, evaluation


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last and len(dataset) >= batch_size,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def build_data_bundle(
    config: dict[str, Any],
    *,
    split: SplitIndices | None = None,
    base_datasets: tuple[Dataset, Dataset] | None = None,
) -> DataBundle:
    train_base, test_base = base_datasets or build_base_datasets(config)
    targets = dataset_targets(train_base)
    seed = int(config["train"]["seed"])
    if split is None:
        split = create_stratified_split(
            targets,
            float(config["data"]["label_ratio"]),
            seed,
            float(config["data"]["validation_fraction"]),
        )
    mean, std = normalization_stats(train_base)
    weak, strong, evaluation = build_transforms(mean, std)
    method = str(config["method"]["name"])
    target_transform = strong if method == "fixmatch" else weak

    data_config = config["data"]
    train_config = config["train"]
    workers = int(data_config["num_workers"])
    pin_memory = bool(data_config["pin_memory"])
    labeled_dataset = LabeledDataset(train_base, split.labeled, weak)
    validation_dataset = EvaluationDataset(train_base, split.validation, evaluation)
    test_indices = np.arange(len(test_base), dtype=np.int64)
    test_dataset = EvaluationDataset(test_base, test_indices, evaluation)

    labeled_loader = _loader(
        labeled_dataset,
        batch_size=int(train_config["batch_labeled"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        seed=seed + 17,
        drop_last=True,
    )
    unlabeled_loader: DataLoader | None = None
    if method != "supervised":
        if split.unlabeled.size == 0:
            raise ValueError("Semi-supervised methods require a non-empty unlabeled pool")
        unlabeled_dataset = UnlabeledDataset(
            train_base,
            split.unlabeled,
            weak_transform=weak,
            target_transform=target_transform,
        )
        unlabeled_loader = _loader(
            unlabeled_dataset,
            batch_size=int(train_config["batch_unlabeled"]),
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            seed=seed + 29,
            drop_last=True,
        )

    return DataBundle(
        train_base=train_base,
        test_base=test_base,
        split=split,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        validation_loader=_loader(
            validation_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            seed=seed + 31,
        ),
        test_loader=_loader(
            test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            seed=seed + 37,
        ),
        evaluation_transform=evaluation,
        diagnostic_targets=torch.as_tensor(targets, dtype=torch.long),
        mean=mean,
        std=std,
    )
