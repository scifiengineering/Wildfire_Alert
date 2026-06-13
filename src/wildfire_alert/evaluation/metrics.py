"""Metrics for imbalanced binary segmentation and discriminative alerting."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BinarySegmentationMetrics:
    """Pixel-level binary segmentation metrics."""

    average_precision: float
    dice: float
    iou: float
    precision: float
    recall: float
    positive_rate: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def binary_segmentation_metrics(
    probabilities: FloatArray,
    labels: FloatArray,
    threshold: float = 0.5,
) -> BinarySegmentationMetrics:
    """Compute AP, Dice, IoU, precision, and recall for binary raster predictions."""

    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same shape.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one.")

    scores = probabilities.reshape(-1).astype(np.float64)
    targets = (labels.reshape(-1) > 0).astype(np.uint8)
    predictions = scores >= threshold
    positives = targets == 1

    true_positives = float(np.count_nonzero(predictions & positives))
    false_positives = float(np.count_nonzero(predictions & ~positives))
    false_negatives = float(np.count_nonzero(~predictions & positives))
    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    dice = _safe_divide(2 * true_positives, 2 * true_positives + false_positives + false_negatives)
    iou = _safe_divide(true_positives, true_positives + false_positives + false_negatives)

    return BinarySegmentationMetrics(
        average_precision=average_precision(scores, targets),
        dice=dice,
        iou=iou,
        precision=precision,
        recall=recall,
        positive_rate=float(targets.mean()) if targets.size else 0.0,
    )


def average_precision(scores: FloatArray, labels: NDArray[np.uint8]) -> float:
    """Compute uninterpolated average precision from binary labels and scores."""

    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape.")
    positive_count = int(labels.sum())
    if positive_count == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(sorted_labels) + 1, dtype=np.float64)
    precision_at_rank = true_positives / ranks
    return float((precision_at_rank * sorted_labels).sum() / positive_count)


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
