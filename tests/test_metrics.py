"""Tests for segmentation metrics."""

import numpy as np

from wildfire_alert.evaluation.metrics import average_precision, binary_segmentation_metrics


def test_average_precision_rewards_correct_ranking() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.uint8)
    good_scores = np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float64)
    bad_scores = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float64)
    assert average_precision(good_scores, labels) == 1.0
    assert average_precision(bad_scores, labels) < 0.5


def test_binary_segmentation_metrics_for_perfect_prediction() -> None:
    labels = np.asarray([[1, 0], [0, 1]], dtype=np.float64)
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    metrics = binary_segmentation_metrics(probabilities, labels)
    assert metrics.average_precision == 1.0
    assert metrics.dice == 1.0
    assert metrics.iou == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
