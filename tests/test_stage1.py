"""Tests for the Stage 1 model contract and imbalanced segmentation loss."""

import pytest
import torch

from wildfire_alert.stage1.losses import BinaryDiceLoss, BinaryFocalLoss, CombinedFocalDiceLoss
from wildfire_alert.stage1.unet import SpreadUNet


def test_combined_loss_is_finite_and_differentiable() -> None:
    logits = torch.zeros((2, 1, 16, 16), requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0, 4:6, 4:6] = 1
    loss = CombinedFocalDiceLoss()(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_better_logits_have_lower_losses() -> None:
    targets = torch.zeros((1, 1, 8, 8))
    targets[:, :, 2:4, 2:4] = 1
    incorrect = torch.zeros_like(targets)
    correct = torch.where(targets > 0, torch.tensor(5.0), torch.tensor(-5.0))
    assert BinaryFocalLoss()(correct, targets) < BinaryFocalLoss()(incorrect, targets)
    assert BinaryDiceLoss()(correct, targets) < BinaryDiceLoss()(incorrect, targets)


def test_multiday_unet_contract() -> None:
    model = SpreadUNet(channels_per_day=3, input_days=2, encoder_weights=None)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros((1, 2, 3, 32, 32)))
    assert logits.shape == (1, 1, 32, 32)


def test_multiday_unet_rejects_wrong_channel_count() -> None:
    model = SpreadUNet(channels_per_day=3, input_days=2, encoder_weights=None)
    with pytest.raises(ValueError):
        model(torch.zeros((1, 2, 2, 32, 32)))
