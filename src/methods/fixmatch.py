from __future__ import annotations

import torch
from torch import nn

from .pseudolabel import _semi_supervised_step


def fixmatch_step(
    model: nn.Module,
    labeled_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    unlabeled_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    tau: float,
    lambda_u: float,
    diagnostic_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | list[int]]]:
    """FixMatch loss; the data module supplies weak and strong image views."""
    return _semi_supervised_step(
        model,
        labeled_batch,
        unlabeled_batch,
        device=device,
        tau=tau,
        lambda_u=lambda_u,
        diagnostic_targets=diagnostic_targets,
    )

