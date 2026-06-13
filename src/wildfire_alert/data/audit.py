"""Dataset inventory and integrity checks that avoid loading all events at once."""

from collections import Counter
from dataclasses import asdict, dataclass

from wildfire_alert.data.wsts_loader import WSTSDataset, load_event


@dataclass(frozen=True)
class AuditReport:
    """Summary of an inventoried WSTS dataset or development subset."""

    backend: str
    event_count: int
    sample_count: int
    events_by_year: dict[int, int]
    samples_by_year: dict[int, int]
    checked_events: int
    invalid_events: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""

        return asdict(self)


def audit_dataset(dataset: WSTSDataset, check_events: int = 4) -> AuditReport:
    """Inventory the dataset and fully validate a bounded number of events."""

    event_years = Counter(event.year for event in dataset.events)
    sample_years = Counter(dataset.events[sample.event_index].year for sample in dataset.samples)
    invalid: list[str] = []
    bounded_check_count = min(max(0, check_events), len(dataset.events))
    for event in dataset.events[:bounded_check_count]:
        try:
            load_event(event, dataset.backend)
        except (ImportError, KeyError, OSError, ValueError) as exc:
            invalid.append(f"{event.year}/{event.event_id}: {exc}")

    return AuditReport(
        backend=dataset.backend,
        event_count=len(dataset.events),
        sample_count=len(dataset),
        events_by_year=dict(sorted(event_years.items())),
        samples_by_year=dict(sorted(sample_years.items())),
        checked_events=bounded_check_count,
        invalid_events=tuple(invalid),
    )
