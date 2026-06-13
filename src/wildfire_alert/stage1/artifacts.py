"""Utilities for calibrated Stage 1 prediction artifacts."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def top_fraction_mask(
    probability: NDArray[np.float32],
    top_fraction: float,
) -> tuple[NDArray[np.uint8], float]:
    """Return a mask containing the highest-risk fraction of raster pixels."""

    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between zero and one.")
    flat = probability.reshape(-1)
    count = max(1, int(round(flat.size * top_fraction)))
    selected = np.argpartition(flat, -count)[-count:]
    mask = np.zeros(flat.shape, dtype=np.uint8)
    mask[selected] = 1
    threshold = float(flat[selected].min())
    return mask.reshape(probability.shape), threshold


def save_prediction_artifact(
    path: Path,
    probability: NDArray[np.float32],
    target: NDArray[np.uint8],
    event_id: str,
    target_date: str,
    threshold: float,
    top_fraction: float,
) -> dict[str, object]:
    """Save one Stage 1 artifact and return its manifest record."""

    probability = np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0).astype(
        np.float32
    )
    target = (target > 0).astype(np.uint8)
    threshold_mask = (probability >= threshold).astype(np.uint8)
    top_mask, top_threshold = top_fraction_mask(probability, top_fraction)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probability=probability,
        target=target,
        threshold_mask=threshold_mask,
        top_mask=top_mask,
        event_id=np.bytes_(event_id),
        target_date=np.bytes_(target_date),
        threshold=np.float32(threshold),
        top_fraction=np.float32(top_fraction),
        top_threshold=np.float32(top_threshold),
    )
    return {
        "event_id": event_id,
        "target_date": target_date,
        "path": str(path),
        "probability_max": float(probability.max()),
        "target_positive_pixels": int(target.sum()),
        "threshold_positive_pixels": int(threshold_mask.sum()),
        "top_positive_pixels": int(top_mask.sum()),
        "top_threshold": top_threshold,
    }


def ensemble_prediction_artifacts(
    source_paths: tuple[Path, ...],
    output_path: Path,
    threshold: float,
    top_fraction: float,
) -> dict[str, object]:
    """Average matching probability rasters and save one ensemble artifact."""

    if not source_paths:
        raise ValueError("At least one source prediction is required.")

    probabilities: list[NDArray[np.float32]] = []
    target: NDArray[np.uint8] | None = None
    event_id = ""
    target_date = ""
    for source_path in source_paths:
        with np.load(source_path, allow_pickle=False) as artifact:
            probability = np.asarray(artifact["probability"], dtype=np.float32)
            source_target = np.asarray(artifact["target"], dtype=np.uint8)
            source_event_id = _decode_scalar(artifact["event_id"])
            source_target_date = _decode_scalar(artifact["target_date"])
        if target is None:
            target = source_target
            event_id = source_event_id
            target_date = source_target_date
        elif (
            probability.shape != probabilities[0].shape
            or not np.array_equal(source_target, target)
            or source_event_id != event_id
            or source_target_date != target_date
        ):
            raise ValueError(f"Mismatched ensemble artifact: {source_path}")
        probabilities.append(probability)

    assert target is not None
    ensemble = np.asarray(
        np.mean(np.stack(probabilities), axis=0, dtype=np.float32),
        dtype=np.float32,
    )
    record = save_prediction_artifact(
        output_path,
        ensemble,
        target,
        event_id,
        target_date,
        threshold,
        top_fraction,
    )
    record["source_paths"] = [str(path) for path in source_paths]
    record["ensemble_size"] = len(source_paths)
    return record


def _decode_scalar(value: np.ndarray) -> str:
    scalar = value.item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)
