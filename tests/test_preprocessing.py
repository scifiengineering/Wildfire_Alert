"""Tests for raw WSTS preprocessing."""

import numpy as np
import pytest

from wildfire_alert.data.preprocessing import (
    ChannelStatistics,
    circular_direction_channels,
    normalize_inputs,
    prepare_raw_inputs,
)


def test_prepare_raw_inputs_binarizes_active_fire_and_fills_nan() -> None:
    inputs = np.ones((2, 23, 4, 4), dtype=np.float32)
    inputs[0, 0, 0, 0] = np.nan
    inputs[:, 22] = np.nan
    inputs[1, 22, 1, 1] = 854.0
    prepared = prepare_raw_inputs(inputs)
    assert prepared[0, 0, 0, 0] == 0
    assert prepared[:, 22].sum() == 1


def test_circular_direction_channels_avoid_zero_degree_discontinuity() -> None:
    inputs = np.zeros((1, 23, 1, 2), dtype=np.float32)
    inputs[:, 7, 0] = np.asarray([359.0, 1.0])
    circular = circular_direction_channels(inputs)
    assert circular.shape == (1, 26, 1, 2)
    direction_sine = circular[:, 7]
    direction_cosine = circular[:, 8]
    assert abs(direction_sine[0, 0, 0] - direction_sine[0, 0, 1]) < 0.04
    assert abs(direction_cosine[0, 0, 0] - direction_cosine[0, 0, 1]) < 0.001


def test_normalization_excludes_direction_and_active_fire() -> None:
    inputs = np.ones((1, 23, 1, 1), dtype=np.float32)
    statistics = ChannelStatistics(
        means=np.ones(23, dtype=np.float32),
        standard_deviations=np.full(23, 2.0, dtype=np.float32),
    )
    normalized = normalize_inputs(inputs, statistics)
    assert normalized[0, 0, 0, 0] == 0
    assert normalized[0, 7, 0, 0] == 1
    assert normalized[0, 22, 0, 0] == 1


def test_statistics_reject_nonpositive_standard_deviation() -> None:
    with pytest.raises(ValueError):
        ChannelStatistics(
            means=np.zeros(23, dtype=np.float32),
            standard_deviations=np.zeros(23, dtype=np.float32),
        )
