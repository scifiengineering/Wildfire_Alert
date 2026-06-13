"""Inventory-first loader for raw, HDF5, and lightweight NPZ WSTS events."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from wildfire_alert.data.channels import ACTIVE_FIRE_CHANNEL, N_CHANNELS

Backend = Literal["npz", "geotiff", "hdf5"]
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class EventRecord:
    """A discovered fire event and its ordered daily files."""

    event_id: str
    year: int
    paths: tuple[Path, ...]

    @property
    def days(self) -> int:
        """Return the number of observations available for the event."""

        return len(self.paths) if len(self.paths) > 1 else _single_file_days(self.paths[0])


@dataclass(frozen=True)
class SampleIndex:
    """Locates a temporal training sample within an event."""

    event_index: int
    start_day: int


@dataclass(frozen=True)
class WSTSSample:
    """One temporal fire spread prediction sample."""

    inputs: FloatArray
    target: FloatArray
    event_id: str
    year: int
    input_dates: tuple[str, ...]
    target_date: str


class WSTSDataset:
    """Load temporal WSTS samples without requiring the complete dataset locally."""

    def __init__(
        self,
        root: Path,
        years: Sequence[int],
        input_days: int = 5,
        backend: Backend = "npz",
        evaluation_window_days: int | None = 5,
        target_offset_days: int = 1,
    ) -> None:
        if input_days < 1:
            raise ValueError("input_days must be at least one.")
        if target_offset_days < 1:
            raise ValueError("target_offset_days must be at least one.")
        if evaluation_window_days is not None and evaluation_window_days < input_days:
            raise ValueError("evaluation_window_days cannot be smaller than input_days.")

        self.root = Path(root)
        self.backend = backend
        self.input_days = input_days
        self.target_offset_days = target_offset_days
        self.skip_initial = (evaluation_window_days or input_days) - input_days
        self.events = discover_events(self.root, years, backend)
        self.samples = self._build_sample_index()

    def _build_sample_index(self) -> tuple[SampleIndex, ...]:
        samples: list[SampleIndex] = []
        for event_index, event in enumerate(self.events):
            count = event.days - self.input_days - self.skip_initial - self.target_offset_days + 1
            samples.extend(
                SampleIndex(event_index=event_index, start_day=start + self.skip_initial)
                for start in range(max(0, count))
            )
        return tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> WSTSSample:
        sample_index = self.samples[index]
        event = self.events[sample_index.event_index]
        data, dates = load_event(event, self.backend)
        start = sample_index.start_day
        end = start + self.input_days
        inputs = data[start:end].astype(np.float32, copy=False)
        target_index = end + self.target_offset_days - 1
        target = np.nan_to_num(data[target_index, ACTIVE_FIRE_CHANNEL], nan=0.0)
        target = (target > 0).astype(np.float32)
        return WSTSSample(
            inputs=inputs,
            target=target,
            event_id=event.event_id,
            year=event.year,
            input_dates=dates[start:end],
            target_date=dates[target_index],
        )


def discover_events(root: Path, years: Sequence[int], backend: Backend) -> tuple[EventRecord, ...]:
    """Discover complete event time series beneath the configured root."""

    events: list[EventRecord] = []
    for year in sorted(set(years)):
        year_dir = root / str(year)
        if not year_dir.exists():
            continue
        if backend == "geotiff":
            for event_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
                paths = tuple(sorted(event_dir.glob("*.tif")))
                if paths:
                    events.append(EventRecord(event_dir.name, year, paths))
        else:
            suffix = ".npz" if backend == "npz" else ".hdf5"
            for path in sorted(year_dir.glob(f"*{suffix}")):
                events.append(EventRecord(path.stem, year, (path,)))
    return tuple(events)


def load_event(event: EventRecord, backend: Backend) -> tuple[FloatArray, tuple[str, ...]]:
    """Load an entire event time series and its dates."""

    if backend == "npz":
        with np.load(event.paths[0], allow_pickle=False) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            dates = tuple(str(value) for value in archive["dates"].tolist())
    elif backend == "hdf5":
        try:
            import h5py  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install the 'data' extra to read HDF5 files.") from exc
        with h5py.File(event.paths[0], "r") as handle:
            dataset = handle["data"]
            data = np.asarray(dataset, dtype=np.float32)
            dates = tuple(_decode_date(value) for value in dataset.attrs["img_dates"])
    else:
        try:
            import rasterio  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install the 'data' extra to read GeoTIFF files.") from exc
        daily_arrays: list[FloatArray] = []
        for path in event.paths:
            with rasterio.open(path) as dataset:
                daily_arrays.append(dataset.read().astype(np.float32))
        data = np.stack(daily_arrays)
        dates = tuple(path.stem.split("_")[0] for path in event.paths)

    _validate_event(data, dates, event)
    return data, dates


def iter_event_summaries(dataset: WSTSDataset) -> Iterator[dict[str, object]]:
    """Yield lightweight event metadata without loading all raster values."""

    for event in dataset.events:
        yield {"event_id": event.event_id, "year": event.year, "days": event.days}


def _single_file_days(path: Path) -> int:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return int(archive["data"].shape[0])
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install the 'data' extra to inventory HDF5 files.") from exc
    with h5py.File(path, "r") as handle:
        return int(handle["data"].shape[0])


def _validate_event(data: FloatArray, dates: tuple[str, ...], event: EventRecord) -> None:
    if data.ndim != 4 or data.shape[1] != N_CHANNELS:
        raise ValueError(
            f"Event {event.event_id} must have shape (days, {N_CHANNELS}, height, width); "
            f"received {data.shape}."
        )
    if data.shape[0] != len(dates):
        raise ValueError(f"Event {event.event_id} has different numbers of observations and dates.")


def _decode_date(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
