"""Tests for physically consistent directional transforms."""

import numpy as np

from wildfire_alert.data.transforms import SpatialTransform


def test_horizontal_flip_updates_direction() -> None:
    inputs = np.zeros((1, 23, 4, 4), dtype=np.float32)
    inputs[:, 7] = 90.0
    target = np.zeros((4, 4), dtype=np.float32)
    transformed, _ = SpatialTransform(horizontal_flip=True).apply(inputs, target, (7,))
    assert np.allclose(transformed[:, 7], 270.0)


def test_counterclockwise_rotation_updates_direction() -> None:
    inputs = np.zeros((1, 23, 4, 4), dtype=np.float32)
    inputs[:, 7] = 90.0
    target = np.zeros((4, 4), dtype=np.float32)
    transformed, _ = SpatialTransform(rotations_ccw=1).apply(inputs, target, (7,))
    assert np.allclose(transformed[:, 7], 0.0)
