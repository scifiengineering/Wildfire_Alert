"""Tests for Stage 1 prediction artifact creation and ensembling."""

from pathlib import Path

import numpy as np

from wildfire_alert.stage1.artifacts import (
    ensemble_prediction_artifacts,
    save_prediction_artifact,
    top_fraction_mask,
)


def test_top_fraction_mask_selects_requested_risk_pixels() -> None:
    probability = np.arange(100, dtype=np.float32).reshape(10, 10)

    mask, threshold = top_fraction_mask(probability, 0.1)

    assert int(mask.sum()) == 10
    assert threshold == 90.0
    assert np.all(probability[mask > 0] >= threshold)


def test_ensemble_prediction_artifacts_averages_probabilities(tmp_path: Path) -> None:
    target = np.zeros((2, 2), dtype=np.uint8)
    first = np.asarray([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)
    second = np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    first_path = tmp_path / "fold0.npz"
    second_path = tmp_path / "fold1.npz"
    save_prediction_artifact(first_path, first, target, "fire_1", "2020-01-06", 0.5, 0.25)
    save_prediction_artifact(second_path, second, target, "fire_1", "2020-01-06", 0.5, 0.25)

    output_path = tmp_path / "ensemble.npz"
    record = ensemble_prediction_artifacts(
        (first_path, second_path),
        output_path,
        threshold=0.5,
        top_fraction=0.25,
    )

    with np.load(output_path, allow_pickle=False) as artifact:
        assert np.allclose(artifact["probability"], (first + second) / 2)
        assert int(artifact["threshold_mask"].sum()) == 2
    assert record["ensemble_size"] == 2
