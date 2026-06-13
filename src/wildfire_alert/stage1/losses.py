"""Losses for severely imbalanced next-day fire segmentation."""

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class BinaryFocalLoss(nn.Module):
    """Binary focal loss operating directly on model logits."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("alpha must be between zero and one.")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute mean binary focal loss."""

        targets = targets.to(dtype=logits.dtype)
        cross_entropy = functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        correct_probabilities = probabilities * targets + (1 - probabilities) * (1 - targets)
        focal_weight = (1 - correct_probabilities).pow(self.gamma)
        if self.alpha is not None:
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = focal_weight * alpha_weight
        return (focal_weight * cross_entropy).mean()


class BinaryDiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive.")
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute mean per-sample Dice loss."""

        probabilities = torch.sigmoid(logits)
        targets = targets.to(dtype=probabilities.dtype)
        dimensions = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        score = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - score.mean()


class CombinedFocalDiceLoss(nn.Module):
    """Weighted sum of binary focal and Dice losses."""

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if focal_weight < 0 or dice_weight < 0 or focal_weight + dice_weight == 0:
            raise ValueError("Loss weights must be non-negative and not both zero.")
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = BinaryFocalLoss(gamma=focal_gamma)
        self.dice = BinaryDiceLoss()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the configured focal-plus-Dice loss."""

        return cast(
            Tensor,
            self.focal_weight * self.focal(logits, targets)
            + self.dice_weight * self.dice(logits, targets),
        )
