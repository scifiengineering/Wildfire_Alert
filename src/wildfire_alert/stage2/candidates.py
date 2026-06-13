"""Candidate extraction and local feature engineering for Stage 2."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wildfire_alert.stage1.dataset import Stage1BatchSample
from wildfire_alert.stage2.geometry import (
    bearing_degrees_from_vector,
    centroid,
    direction_alignment,
    nearest_distance_pixels,
    unit_vector_from_to,
)

PIXEL_SIZE_KM = 0.375


@dataclass(frozen=True)
class PredictionArtifact:
    """Arrays loaded from a Stage 1 prediction NPZ."""

    probability: NDArray[np.float32]
    target: NDArray[np.uint8]
    threshold_mask: NDArray[np.uint8]
    top_mask: NDArray[np.uint8]
    threshold: float
    top_fraction: float
    top_threshold: float


def load_prediction_artifact(path: Path) -> PredictionArtifact:
    """Load a Stage 1 prediction artifact."""

    with np.load(path) as data:
        return PredictionArtifact(
            probability=np.asarray(data["probability"], dtype=np.float32),
            target=np.asarray(data["target"], dtype=np.uint8),
            threshold_mask=np.asarray(data["threshold_mask"], dtype=np.uint8),
            top_mask=np.asarray(data["top_mask"], dtype=np.uint8),
            threshold=float(data["threshold"]),
            top_fraction=float(data["top_fraction"]),
            top_threshold=float(data["top_threshold"]),
        )


def candidate_mask(
    artifact: PredictionArtifact,
    current_fire: NDArray[np.bool_] | None = None,
    include_threshold: bool = False,
    include_top: bool = True,
    include_target: bool = True,
    ring_radius_km: float = 5.0,
) -> NDArray[np.bool_]:
    """Return the Stage 2 candidate mask."""

    mask = np.zeros_like(artifact.target, dtype=bool)
    if include_threshold:
        mask |= artifact.threshold_mask > 0
    if include_top:
        mask |= artifact.top_mask > 0
    if include_target:
        mask |= artifact.target > 0
    if current_fire is not None and ring_radius_km > 0:
        mask |= local_ring_mask(current_fire, radius_pixels=ring_radius_km / PIXEL_SIZE_KM)
    return mask


def local_ring_mask(
    current_fire: NDArray[np.bool_],
    radius_pixels: float,
) -> NDArray[np.bool_]:
    """Return pixels within radius of current fire, excluding current fire itself."""

    ring = np.zeros_like(current_fire, dtype=bool)
    fire_rows, fire_cols = np.nonzero(current_fire)
    if fire_rows.size == 0:
        return ring

    height, width = current_fire.shape
    radius = int(np.ceil(radius_pixels))
    radius_squared = radius_pixels * radius_pixels
    for fire_row, fire_col in zip(fire_rows, fire_cols, strict=True):
        row_min = max(0, int(fire_row) - radius)
        row_max = min(height, int(fire_row) + radius + 1)
        col_min = max(0, int(fire_col) - radius)
        col_max = min(width, int(fire_col) + radius + 1)
        rows = np.arange(row_min, row_max)[:, None]
        cols = np.arange(col_min, col_max)[None, :]
        nearby = (rows - fire_row) ** 2 + (cols - fire_col) ** 2 <= radius_squared
        ring[row_min:row_max, col_min:col_max] |= nearby
    ring &= ~current_fire
    return ring


def feature_rows_for_sample(
    sample: Stage1BatchSample,
    artifact: PredictionArtifact,
    fold: int,
    split: str,
    include_threshold_candidates: bool = False,
    include_target_candidates: bool = False,
    ring_radius_km: float = 5.0,
) -> list[dict[str, int | float | str]]:
    """Build Stage 2 feature rows for one event-day sample."""

    latest = sample.inputs[-1].numpy()
    current_fire = latest[22] > 0
    current_centroid = centroid(current_fire)
    has_current_fire = current_centroid is not None
    if current_centroid is None:
        height, width = artifact.probability.shape
        current_centroid = ((height - 1) / 2.0, (width - 1) / 2.0)

    rows: list[dict[str, int | float | str]] = []
    candidates = candidate_mask(
        artifact,
        current_fire=current_fire,
        include_threshold=include_threshold_candidates,
        include_target=include_target_candidates,
        ring_radius_km=ring_radius_km,
    )
    candidate_rows, candidate_cols = np.nonzero(candidates)
    for row, col in zip(candidate_rows, candidate_cols, strict=True):
        unit_row, unit_col = unit_vector_from_to(
            current_centroid[0],
            current_centroid[1],
            float(row),
            float(col),
        )
        distance_pixels = nearest_distance_pixels(current_fire, int(row), int(col))
        distance_km = distance_pixels * PIXEL_SIZE_KM if np.isfinite(distance_pixels) else np.nan
        label = int(artifact.target[row, col] > 0)

        wind_direction = float(latest[7, row, col])
        forecast_wind_direction = float(latest[19, row, col])
        aspect = float(latest[13, row, col])
        slope = float(latest[12, row, col])

        rows.append(
            {
                "fold": fold,
                "split": split,
                "event_id": sample.event_id,
                "target_date": sample.target_date,
                "row": int(row),
                "col": int(col),
                "label": label,
                "stage1_probability": float(artifact.probability[row, col]),
                "stage1_threshold_mask": int(artifact.threshold_mask[row, col] > 0),
                "stage1_top_mask": int(artifact.top_mask[row, col] > 0),
                "threshold": artifact.threshold,
                "top_fraction": artifact.top_fraction,
                "top_threshold": artifact.top_threshold,
                "has_current_fire": int(has_current_fire),
                "current_fire_at_candidate": int(current_fire[row, col]),
                "distance_to_current_fire_px": float(distance_pixels),
                "distance_to_current_fire_km": float(distance_km),
                "bearing_from_fire_deg": bearing_degrees_from_vector(unit_row, unit_col),
                "wind_alignment": direction_alignment(unit_row, unit_col, wind_direction),
                "forecast_wind_alignment": direction_alignment(
                    unit_row, unit_col, forecast_wind_direction
                ),
                "slope_alignment": slope * direction_alignment(unit_row, unit_col, aspect),
                "ndvi": float(latest[3, row, col]),
                "evi2": float(latest[4, row, col]),
                "total_precipitation": float(latest[5, row, col]),
                "wind_speed": float(latest[6, row, col]),
                "wind_direction": wind_direction,
                "minimum_temperature": float(latest[8, row, col]),
                "maximum_temperature": float(latest[9, row, col]),
                "energy_release_component": float(latest[10, row, col]),
                "specific_humidity": float(latest[11, row, col]),
                "slope": slope,
                "aspect": aspect,
                "elevation": float(latest[14, row, col]),
                "pdsi": float(latest[15, row, col]),
                "landcover_class": float(latest[16, row, col]),
                "forecast_total_precipitation": float(latest[17, row, col]),
                "forecast_wind_speed": float(latest[18, row, col]),
                "forecast_wind_direction": forecast_wind_direction,
                "forecast_temperature": float(latest[20, row, col]),
                "forecast_specific_humidity": float(latest[21, row, col]),
            }
        )
    return rows
