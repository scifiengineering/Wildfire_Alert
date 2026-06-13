"""PyTorch dataset adapters for Stage 1 U-Net training."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional
from torch.utils.data import Dataset

from wildfire_alert.data.preprocessing import circular_direction_channels, prepare_raw_inputs
from wildfire_alert.data.wsts_loader import Backend, WSTSDataset


@dataclass(frozen=True)
class Stage1BatchSample:
    """One tensorized Stage 1 sample and its metadata."""

    inputs: Tensor
    target: Tensor
    event_id: str
    target_date: str


class Stage1TorchDataset(Dataset[Stage1BatchSample]):
    """Tensorize WSTS samples for Stage 1 training and smoke tests."""

    def __init__(
        self,
        root: Path,
        years: tuple[int, ...],
        input_days: int = 5,
        target_offset_days: int = 1,
        backend: Backend = "geotiff",
        crop_size: int | None = 256,
        crop_mode: str = "random",
        circular_directions: bool = True,
        exclude_empty_current_fire: bool = False,
        allowed_event_ids: frozenset[str] | None = None,
        seed: int = 42,
    ) -> None:
        if crop_size is not None and crop_size <= 0:
            raise ValueError("crop_size must be positive when provided.")
        if crop_mode not in {"random", "center"}:
            raise ValueError("crop_mode must be 'random' or 'center'.")
        self.base = WSTSDataset(
            root,
            years,
            input_days=input_days,
            backend=backend,
            target_offset_days=target_offset_days,
        )
        self.crop_size = crop_size
        self.crop_mode = crop_mode
        self.circular_directions = circular_directions
        self.rng = np.random.default_rng(seed)
        self.indices = self._select_indices(exclude_empty_current_fire, allowed_event_ids)

    def _select_indices(
        self,
        exclude_empty_current_fire: bool,
        allowed_event_ids: frozenset[str] | None,
    ) -> tuple[int, ...]:
        selected: list[int] = []
        for index in range(len(self.base)):
            event = self.base.events[self.base.samples[index].event_index]
            if allowed_event_ids is not None and event.event_id not in allowed_event_ids:
                continue
            if not exclude_empty_current_fire:
                selected.append(index)
                continue
            sample = self.base[index]
            current_active_pixels = np.nan_to_num(sample.inputs[-1, 22], nan=0.0) > 0
            if current_active_pixels.any():
                selected.append(index)
        return tuple(selected)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Stage1BatchSample:
        sample = self.base[self.indices[index]]
        inputs = prepare_raw_inputs(sample.inputs)
        if self.circular_directions:
            inputs = circular_direction_channels(inputs)
        input_tensor = torch.from_numpy(inputs)
        target_tensor = torch.from_numpy(sample.target).unsqueeze(0)
        if self.crop_size is not None:
            if self.crop_mode == "random":
                input_tensor, target_tensor = random_aligned_crop_or_pad(
                    input_tensor,
                    target_tensor,
                    self.crop_size,
                    self.rng,
                )
            else:
                input_tensor, target_tensor = center_aligned_crop_or_pad(
                    input_tensor,
                    target_tensor,
                    self.crop_size,
                )
        return Stage1BatchSample(
            inputs=input_tensor,
            target=target_tensor,
            event_id=sample.event_id,
            target_date=sample.target_date,
        )


def collate_stage1_samples(samples: list[Stage1BatchSample]) -> dict[str, object]:
    """Collate already size-aligned Stage 1 samples."""

    return {
        "inputs": torch.stack([sample.inputs for sample in samples]),
        "targets": torch.stack([sample.target for sample in samples]),
        "event_ids": [sample.event_id for sample in samples],
        "target_dates": [sample.target_date for sample in samples],
    }


def random_aligned_crop_or_pad(
    inputs: Tensor,
    target: Tensor,
    crop_size: int,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    """Pad if needed, then crop inputs and target using the same spatial window."""

    inputs, target = pad_to_minimum_size(inputs, target, crop_size)
    height, width = inputs.shape[-2:]
    top = 0 if height == crop_size else int(rng.integers(0, height - crop_size + 1))
    left = 0 if width == crop_size else int(rng.integers(0, width - crop_size + 1))
    return (
        inputs[..., top : top + crop_size, left : left + crop_size],
        target[..., top : top + crop_size, left : left + crop_size],
    )


def center_aligned_crop_or_pad(
    inputs: Tensor,
    target: Tensor,
    crop_size: int,
) -> tuple[Tensor, Tensor]:
    """Pad if needed, then take a centered crop from inputs and target."""

    inputs, target = pad_to_minimum_size(inputs, target, crop_size)
    height, width = inputs.shape[-2:]
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return (
        inputs[..., top : top + crop_size, left : left + crop_size],
        target[..., top : top + crop_size, left : left + crop_size],
    )


def pad_to_minimum_size(inputs: Tensor, target: Tensor, minimum_size: int) -> tuple[Tensor, Tensor]:
    """Pad inputs and target on bottom/right until both spatial axes meet ``minimum_size``."""

    height, width = inputs.shape[-2:]
    padding_height = max(0, minimum_size - height)
    padding_width = max(0, minimum_size - width)
    if padding_height or padding_width:
        inputs = functional.pad(inputs, (0, padding_width, 0, padding_height))
        target = functional.pad(target, (0, padding_width, 0, padding_height))
    return inputs, target
