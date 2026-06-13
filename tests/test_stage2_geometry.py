import math

import numpy as np

from wildfire_alert.stage2.candidates import local_ring_mask
from wildfire_alert.stage2.geometry import (
    bearing_degrees_from_vector,
    centroid,
    direction_alignment,
    direction_degrees_to_unit_vector,
    nearest_distance_pixels,
    unit_vector_from_to,
)


def test_unit_vector_and_bearing_to_east() -> None:
    row_unit, col_unit = unit_vector_from_to(10, 10, 10, 15)

    assert row_unit == 0
    assert col_unit == 1
    assert bearing_degrees_from_vector(row_unit, col_unit) == 90


def test_direction_alignment_eastward_wind_to_east_candidate() -> None:
    row_unit, col_unit = unit_vector_from_to(10, 10, 10, 15)

    assert math.isclose(direction_alignment(row_unit, col_unit, 90), 1.0)
    assert math.isclose(direction_alignment(row_unit, col_unit, 270), -1.0)


def test_direction_degrees_to_unit_vector_cardinals() -> None:
    north = direction_degrees_to_unit_vector(0)
    east = direction_degrees_to_unit_vector(90)
    south = direction_degrees_to_unit_vector(180)

    assert np.allclose(north, (-1, 0), atol=1e-12)
    assert np.allclose(east, (0, 1), atol=1e-12)
    assert np.allclose(south, (1, 0), atol=1e-12)


def test_centroid_and_nearest_distance() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[3, 3] = True

    assert centroid(mask) == (2.0, 2.0)
    assert math.isclose(nearest_distance_pixels(mask, 1, 3), 2.0)
    assert centroid(np.zeros((3, 3), dtype=bool)) is None


def test_local_ring_mask_excludes_current_fire() -> None:
    current_fire = np.zeros((5, 5), dtype=bool)
    current_fire[2, 2] = True

    ring = local_ring_mask(current_fire, radius_pixels=1.5)

    assert not ring[2, 2]
    assert ring[2, 1]
    assert ring[1, 1]
    assert not ring[0, 0]
