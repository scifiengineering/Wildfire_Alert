"""Tests for Stage 1 tensorization and spatial alignment."""

from pathlib import Path

import numpy as np
import torch

from wildfire_alert.data.synthetic import create_synthetic_dataset
from wildfire_alert.stage1.dataset import (
    Stage1TorchDataset,
    center_aligned_crop_or_pad,
    random_aligned_crop_or_pad,
)


def test_random_aligned_crop_or_pad_preserves_target_alignment() -> None:
    rng = np.random.default_rng(0)
    inputs = torch.zeros((5, 26, 4, 4))
    target = torch.zeros((1, 4, 4))
    inputs[..., 2, 2] = 1
    target[..., 2, 2] = 1
    cropped_inputs, cropped_target = random_aligned_crop_or_pad(inputs, target, 8, rng)
    assert cropped_inputs.shape[-2:] == (8, 8)
    assert cropped_target.shape[-2:] == (8, 8)
    assert cropped_inputs[..., 2, 2].sum() > 0
    assert cropped_target[..., 2, 2].sum() == 1


def test_center_aligned_crop_or_pad_is_deterministic() -> None:
    inputs = torch.zeros((5, 26, 10, 10))
    target = torch.zeros((1, 10, 10))
    inputs[..., 5, 5] = 1
    target[..., 5, 5] = 1
    first_inputs, first_target = center_aligned_crop_or_pad(inputs, target, 6)
    second_inputs, second_target = center_aligned_crop_or_pad(inputs, target, 6)
    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_target, second_target)
    assert first_target.sum() == 1


def test_stage1_dataset_filters_allowed_event_ids(tmp_path: Path) -> None:
    create_synthetic_dataset(tmp_path, years=(2019,), events_per_year=2)
    dataset = Stage1TorchDataset(
        tmp_path,
        years=(2019,),
        backend="npz",
        allowed_event_ids=frozenset({"synthetic_000"}),
    )
    assert len(dataset) == 2
    assert {dataset[index].event_id for index in range(len(dataset))} == {"synthetic_000"}
