from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _semi_supervised_step(
    model: nn.Module,
    labeled_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    unlabeled_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    tau: float,
    lambda_u: float,
    diagnostic_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | list[int]]]:
    labeled_images, labeled_targets, _ = labeled_batch
    weak_images, target_images, unlabeled_indices = unlabeled_batch
    labeled_images = labeled_images.to(device, non_blocking=True)
    labeled_targets = labeled_targets.to(device, non_blocking=True)
    weak_images = weak_images.to(device, non_blocking=True)
    target_images = target_images.to(device, non_blocking=True)

    labeled_logits = model(labeled_images)
    supervised_loss = F.cross_entropy(labeled_logits, labeled_targets)

    with torch.no_grad():
        weak_logits = model(weak_images)
        probabilities = weak_logits.softmax(dim=1)
        confidence, pseudo_targets = probabilities.max(dim=1)
        mask = confidence.ge(tau)

    target_logits = model(target_images)
    per_example = F.cross_entropy(target_logits, pseudo_targets, reduction="none")
    unsupervised_loss = (per_example * mask.float()).mean()
    total_loss = supervised_loss + lambda_u * unsupervised_loss

    accepted = int(mask.sum().item())
    pseudo_correct = 0
    class_counts = torch.zeros(target_logits.shape[1], dtype=torch.long, device=device)
    if accepted:
        accepted_targets = pseudo_targets[mask]
        class_counts = torch.bincount(accepted_targets, minlength=target_logits.shape[1])
        hidden_targets = diagnostic_targets[unlabeled_indices.long()].to(device)
        pseudo_correct = int((accepted_targets == hidden_targets[mask]).sum().item())

    train_accuracy = (labeled_logits.argmax(dim=1) == labeled_targets).float().mean()
    return total_loss, {
        "supervised_loss": float(supervised_loss.detach().item()),
        "unsupervised_loss": float(unsupervised_loss.detach().item()),
        "train_accuracy": float(train_accuracy.detach().item()),
        "accepted": float(accepted),
        "unlabeled_seen": float(mask.numel()),
        "pseudo_correct": float(pseudo_correct),
        "pseudo_class_counts": class_counts.detach().cpu().tolist(),
    }


def pseudolabel_step(
    model: nn.Module,
    labeled_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    unlabeled_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    tau: float,
    lambda_u: float,
    diagnostic_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | list[int]]]:
    return _semi_supervised_step(
        model,
        labeled_batch,
        unlabeled_batch,
        device=device,
        tau=tau,
        lambda_u=lambda_u,
        diagnostic_targets=diagnostic_targets,
    )

