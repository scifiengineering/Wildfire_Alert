"""Spatial transforms that preserve the meaning of directional raster channels."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class SpatialTransform:
    """A combination of 90-degree rotations and optional axis flips."""

    rotations_ccw: int = 0
    horizontal_flip: bool = False
    vertical_flip: bool = False

    def apply(
        self,
        inputs: FloatArray,
        target: FloatArray,
        direction_indices: tuple[int, ...],
    ) -> tuple[FloatArray, FloatArray]:
        """Apply the transform and update directional channel values in degrees."""

        rotations = self.rotations_ccw % 4
        x = np.rot90(inputs, rotations, axes=(-2, -1)).copy()
        y = np.rot90(target, rotations, axes=(-2, -1)).copy()
        x[:, direction_indices] = (x[:, direction_indices] - 90.0 * rotations) % 360.0

        if self.horizontal_flip:
            x = np.flip(x, axis=-1).copy()
            y = np.flip(y, axis=-1).copy()
            x[:, direction_indices] = (360.0 - x[:, direction_indices]) % 360.0

        if self.vertical_flip:
            x = np.flip(x, axis=-2).copy()
            y = np.flip(y, axis=-2).copy()
            x[:, direction_indices] = (180.0 - x[:, direction_indices]) % 360.0

        return x, y
