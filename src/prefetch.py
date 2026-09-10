"""Parallel preparation of exactly the same step-addressed augmented batches."""
from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _pin(batch):
    return tuple(value.pin_memory() for value in batch) if batch is not None else None


def _numpy_batch(batch):
    return tuple(value.numpy() for value in batch) if batch is not None else None


def _tensor_batch(batch):
    return tuple(torch.from_numpy(value) for value in batch) if batch is not None else None


@dataclass
class PreparedStep:
    labeled: Any
    unlabeled: Any
    python_rng: Any
    numpy_rng: Any
    torch_rng: torch.Tensor

    def pin_memory(self):
        self.labeled = _pin(self.labeled)
        self.unlabeled = _pin(self.unlabeled)
        return self

    def restore_cpu_rng(self):
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_rng)


@dataclass
class SerializedStep:
    """Only NumPy payloads cross the queue; no per-step Windows tensor mappings."""
    labeled: Any
    unlabeled: Any
    python_rng: Any
    numpy_rng: Any
    torch_rng: np.ndarray

    def materialize(self, pin_memory):
        step = PreparedStep(_tensor_batch(self.labeled), _tensor_batch(self.unlabeled),
                            self.python_rng, self.numpy_rng, torch.from_numpy(self.torch_rng))
        return step.pin_memory() if pin_memory else step


class StepDataset(Dataset):
    def __init__(self, labeled_stream, unlabeled_stream, seed, start, end):
        self.labeled_stream = labeled_stream
        self.unlabeled_stream = unlabeled_stream
        self.seed, self.start, self.end = seed, start, end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        step = self.start + index
        value = (self.seed * 1_000_003 + step * 97_409 + 17) % (2**31 - 1)
        random.seed(value)
        np.random.seed(value)
        # Workers only prepare CPU images; never initialize a CUDA context here.
        torch.random.default_generator.manual_seed(value)
        labeled = self.labeled_stream.batch(step, shared_memory=False)
        unlabeled = self.unlabeled_stream.batch(step, shared_memory=False) if self.unlabeled_stream else None
        return SerializedStep(_numpy_batch(labeled), _numpy_batch(unlabeled), random.getstate(),
                              np.random.get_state(), torch.get_rng_state().numpy())


def _identity(value):
    return value


def prepared_steps(labeled_stream, unlabeled_stream, *, seed, start, end, workers, pin_memory):
    generator = torch.Generator().manual_seed(seed + 503)
    kwargs = {}
    if workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(
        StepDataset(labeled_stream, unlabeled_stream, seed, start, end),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        collate_fn=_identity,
        pin_memory=False,
        generator=generator,
        **kwargs,
    )
    iterator = iter(loader)
    try:
        for item in iterator:
            yield item.materialize(pin_memory)
    finally:
        del iterator
