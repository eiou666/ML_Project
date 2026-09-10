from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@dataclass
class EvaluationResult:
    metrics: dict[str, object]
    probabilities: np.ndarray
    targets: np.ndarray
    indices: np.ndarray


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(confusion, dtype=np.int64)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    total = matrix.sum()
    accuracy = float(true_positive.sum() / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(total),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    probabilities: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    losses: list[float] = []
    confusion: np.ndarray | None = None

    for images, targets, indices in loader:
        images = images.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets_device, reduction="sum")
        probs = logits.softmax(dim=1)
        predictions = probs.argmax(dim=1)
        class_count = int(logits.shape[1])
        if confusion is None:
            confusion = np.zeros((class_count, class_count), dtype=np.int64)
        np.add.at(
            confusion,
            (targets.numpy().astype(np.int64), predictions.cpu().numpy().astype(np.int64)),
            1,
        )
        losses.append(float(loss.item()))
        probabilities.append(probs.cpu().numpy())
        target_parts.append(targets.numpy())
        index_parts.append(indices.numpy())

    if confusion is None:
        raise ValueError("Cannot evaluate an empty data loader")
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = float(sum(losses) / max(1, int(confusion.sum())))
    return EvaluationResult(
        metrics=metrics,
        probabilities=np.concatenate(probabilities),
        targets=np.concatenate(target_parts).astype(np.int64),
        indices=np.concatenate(index_parts).astype(np.int64),
    )

