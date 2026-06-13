"""Geometry helpers for discriminative Stage 2 features."""

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def unit_vector_from_to(
    origin_row: float,
    origin_col: float,
    target_row: float,
    target_col: float,
) -> tuple[float, float]:
    """Return a row/column unit vector from origin to target."""

    delta_row = target_row - origin_row
    delta_col = target_col - origin_col
    norm = math.hypot(delta_row, delta_col)
    if norm == 0:
        return 0.0, 0.0
    return delta_row / norm, delta_col / norm


def bearing_degrees_from_vector(row_component: float, col_component: float) -> float:
    """Return bearing degrees clockwise from north for a row/column vector."""

    if row_component == 0 and col_component == 0:
        return 0.0
    bearing = math.degrees(math.atan2(col_component, -row_component))
    return bearing % 360.0


def direction_degrees_to_unit_vector(direction_degrees: float) -> tuple[float, float]:
    """Convert clockwise-from-north direction degrees to a row/column unit vector."""

    radians = math.radians(direction_degrees)
    return -math.cos(radians), math.sin(radians)


def direction_alignment(
    candidate_row_unit: float,
    candidate_col_unit: float,
    direction_degrees: float,
) -> float:
    """Dot product between fire-to-candidate direction and a directional channel."""

    direction_row, direction_col = direction_degrees_to_unit_vector(direction_degrees)
    return candidate_row_unit * direction_row + candidate_col_unit * direction_col


def centroid(mask: NDArray[np.bool_]) -> tuple[float, float] | None:
    """Return row/column centroid for true pixels, or None for an empty mask."""

    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return float(rows.mean()), float(cols.mean())


def nearest_distance_pixels(
    mask: NDArray[np.bool_],
    row: int,
    col: int,
) -> float:
    """Return Euclidean pixel distance from one location to nearest true mask pixel."""

    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return float("nan")
    distances = np.hypot(rows.astype(np.float64) - row, cols.astype(np.float64) - col)
    return float(distances.min())
