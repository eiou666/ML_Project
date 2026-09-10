from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def supervised_step(
    model: nn.Module,
    labeled_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float | list[int]]]:
    images, targets, _ = labeled_batch
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return loss, {
        "supervised_loss": float(loss.detach().item()),
        "unsupervised_loss": 0.0,
        "train_accuracy": float(accuracy.detach().item()),
        "accepted": 0.0,
        "unlabeled_seen": 0.0,
        "pseudo_correct": 0.0,
        "pseudo_class_counts": [0] * int(logits.shape[1]),
    }

