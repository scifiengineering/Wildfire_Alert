"""Tests for lightweight WSTS loading."""

from pathlib import Path

from wildfire_alert.data.synthetic import create_synthetic_dataset
from wildfire_alert.data.wsts_loader import WSTSDataset


def test_npz_loader_builds_temporal_samples(tmp_path: Path) -> None:
    create_synthetic_dataset(tmp_path, years=(2018,), events_per_year=1, days=7, size=16)
    dataset = WSTSDataset(
        tmp_path,
        years=(2018,),
        input_days=5,
        evaluation_window_days=5,
        backend="npz",
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.inputs.shape == (5, 23, 16, 16)
    assert sample.target.shape == (16, 16)
    assert sample.target.max() == 1


def test_evaluation_window_aligns_single_and_multi_day_targets(tmp_path: Path) -> None:
    create_synthetic_dataset(tmp_path, years=(2018,), events_per_year=1, days=7, size=16)
    single = WSTSDataset(tmp_path, (2018,), input_days=1, evaluation_window_days=5)
    multi = WSTSDataset(tmp_path, (2018,), input_days=5, evaluation_window_days=5)
    assert single[0].target_date == multi[0].target_date


def test_target_offset_supports_three_day_horizon(tmp_path: Path) -> None:
    create_synthetic_dataset(tmp_path, years=(2018,), events_per_year=1, days=9, size=16)
    dataset = WSTSDataset(
        tmp_path,
        years=(2018,),
        input_days=5,
        evaluation_window_days=5,
        target_offset_days=3,
        backend="npz",
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.input_dates == (
        "2018-07-01",
        "2018-07-02",
        "2018-07-03",
        "2018-07-04",
        "2018-07-05",
    )
    assert sample.target_date == "2018-07-08"
