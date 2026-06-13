"""Tests for event-level reduced-data splits."""

from pathlib import Path

from wildfire_alert.data.event_split import make_event_split
from wildfire_alert.data.synthetic import create_synthetic_dataset


def test_event_split_keeps_train_and_validation_disjoint(tmp_path: Path) -> None:
    create_synthetic_dataset(tmp_path, years=(2019,), events_per_year=5)
    split = make_event_split(tmp_path, (2019,), "npz", validation_fraction=0.4, seed=1)
    assert split.train_event_ids
    assert split.validation_event_ids
    assert split.train_event_ids.isdisjoint(split.validation_event_ids)
    assert len(split.train_event_ids | split.validation_event_ids) == 5
