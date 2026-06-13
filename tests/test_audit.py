"""Tests for bounded dataset audits."""

from pathlib import Path

from wildfire_alert.data.audit import audit_dataset
from wildfire_alert.data.synthetic import create_synthetic_dataset
from wildfire_alert.data.wsts_loader import WSTSDataset


def test_audit_reports_events_and_samples(tmp_path: Path) -> None:
    create_synthetic_dataset(tmp_path, years=(2018, 2019), events_per_year=1, days=7, size=16)
    dataset = WSTSDataset(tmp_path, years=(2018, 2019), input_days=5)
    report = audit_dataset(dataset, check_events=1)
    assert report.event_count == 2
    assert report.sample_count == 4
    assert report.events_by_year == {2018: 1, 2019: 1}
    assert report.checked_events == 1
    assert not report.invalid_events
