"""Preprocessing primitives for raw WSTS rasters before model ingestion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wildfire_alert.data.channels import ACTIVE_FIRE_CHANNEL, DIRECTIONAL_CHANNELS, N_CHANNELS

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class ChannelStatistics:
    """Per-channel training-set statistics used for leakage-free normalization."""

    means: FloatArray
    standard_deviations: FloatArray

    def __post_init__(self) -> None:
        if self.means.shape != (N_CHANNELS,) or self.standard_deviations.shape != (N_CHANNELS,):
            raise ValueError(f"Statistics must contain exactly {N_CHANNELS} channels.")
        if np.any(self.standard_deviations <= 0):
            raise ValueError("Every standard deviation must be positive.")


def prepare_raw_inputs(inputs: FloatArray) -> FloatArray:
    """Prepare raw WSTS inputs while retaining the canonical 23-channel layout.

    Missing continuous values become zero. The final active-fire channel is
    converted from detection-time values and NaNs into a binary current-fire mask.
    Directional channels remain in degrees so geometric feature engineering can use
    them later; Stage 1 may optionally convert them to circular encodings.
    """

    if inputs.ndim != 4 or inputs.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected shape (days, {N_CHANNELS}, height, width); got {inputs.shape}.")
    prepared = np.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    prepared[:, ACTIVE_FIRE_CHANNEL] = (prepared[:, ACTIVE_FIRE_CHANNEL] > 0).astype(np.float32)
    return prepared


def normalize_inputs(inputs: FloatArray, statistics: ChannelStatistics) -> FloatArray:
    """Normalize continuous inputs without altering active-fire or direction channels."""

    normalized = inputs.copy()
    excluded = {*DIRECTIONAL_CHANNELS, ACTIVE_FIRE_CHANNEL}
    for channel in range(N_CHANNELS):
        if channel not in excluded:
            normalized[:, channel] = (
                normalized[:, channel] - statistics.means[channel]
            ) / statistics.standard_deviations[channel]
    return normalized


def circular_direction_channels(inputs: FloatArray) -> FloatArray:
    """Replace each degree channel with sine and cosine channels.

    The result has 26 channels: 20 non-directional raw channels, the active-fire
    mask, and six circular direction channels. This avoids the false discontinuity
    between directions near 0 and 360 degrees.
    """

    if inputs.ndim != 4 or inputs.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected shape (days, {N_CHANNELS}, height, width); got {inputs.shape}.")
    output: list[FloatArray] = []
    directional = set(DIRECTIONAL_CHANNELS)
    for channel in range(N_CHANNELS):
        values = inputs[:, channel : channel + 1]
        if channel in directional:
            radians = np.deg2rad(values)
            output.extend((np.sin(radians).astype(np.float32), np.cos(radians).astype(np.float32)))
        else:
            output.append(values)
    return np.concatenate(output, axis=1)
