"""Event-level development splits for reduced-data experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wildfire_alert.data.wsts_loader import Backend, discover_events


@dataclass(frozen=True)
class EventSplit:
    """A train/validation split where every fire event belongs to one side."""

    train_event_ids: frozenset[str]
    validation_event_ids: frozenset[str]


def make_event_split(
    root: Path,
    years: tuple[int, ...],
    backend: Backend,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> EventSplit:
    """Create a deterministic event-level split across the selected years."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    events = discover_events(root, years, backend)
    if len(events) < 2:
        raise ValueError("At least two events are required for an event-level split.")

    event_ids = np.asarray([event.event_id for event in events])
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(event_ids))
    validation_count = max(1, int(round(len(event_ids) * validation_fraction)))
    validation_count = min(validation_count, len(event_ids) - 1)
    validation_indices = set(int(index) for index in permutation[:validation_count])
    train_ids = frozenset(
        str(event_id) for index, event_id in enumerate(event_ids) if index not in validation_indices
    )
    validation_ids = frozenset(
        str(event_id) for index, event_id in enumerate(event_ids) if index in validation_indices
    )
    return EventSplit(train_event_ids=train_ids, validation_event_ids=validation_ids)


def make_event_kfold_split(
    root: Path,
    years: tuple[int, ...],
    backend: Backend,
    fold: int,
    n_splits: int = 3,
    seed: int = 42,
) -> EventSplit:
    """Create one fold from mutually exclusive event-level k-fold partitions."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")
    if not 0 <= fold < n_splits:
        raise ValueError("fold must satisfy 0 <= fold < n_splits.")

    events = discover_events(root, years, backend)
    if len(events) < n_splits:
        raise ValueError("At least one event per fold is required.")

    event_ids = np.asarray([event.event_id for event in events])
    rng = np.random.default_rng(seed)
    shuffled = event_ids[rng.permutation(len(event_ids))]
    validation_ids = frozenset(
        str(event_id) for event_id in np.array_split(shuffled, n_splits)[fold]
    )
    train_ids = frozenset(str(event_id) for event_id in event_ids if event_id not in validation_ids)
    return EventSplit(train_event_ids=train_ids, validation_event_ids=validation_ids)


def make_event_kfolds(
    root: Path,
    years: tuple[int, ...],
    backend: Backend,
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[EventSplit, ...]:
    """Create all mutually exclusive event-level k-fold splits."""

    return tuple(
        make_event_kfold_split(root, years, backend, fold=index, n_splits=n_splits, seed=seed)
        for index in range(n_splits)
    )
