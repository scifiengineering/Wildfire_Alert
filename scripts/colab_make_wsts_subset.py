"""Create a representative, shareable WildfireSpreadTS subset in Google Colab.

Run this file in Colab after mounting Google Drive. It supports either the raw
GeoTIFF layout ``<root>/<year>/<event>/*.tif`` or the converted HDF5 layout
``<root>/<year>/<event>.hdf5``. Complete events are copied; individual days are
never sampled because temporal continuity is required by the project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

YEARS = (2018, 2019, 2020, 2021)
EXPECTED_CHANNELS = 23
ACTIVE_FIRE_CHANNEL_ZERO_BASED = 22
CHANNEL_NAMES = (
    "viirs_m11",
    "viirs_i2",
    "viirs_i1",
    "ndvi",
    "evi2",
    "total_precipitation",
    "wind_speed",
    "wind_direction",
    "minimum_temperature",
    "maximum_temperature",
    "energy_release_component",
    "specific_humidity",
    "slope",
    "aspect",
    "elevation",
    "pdsi",
    "landcover_class",
    "forecast_total_precipitation",
    "forecast_wind_speed",
    "forecast_wind_direction",
    "forecast_temperature",
    "forecast_specific_humidity",
    "active_fire",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Extracted WSTS root on Drive.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory on Drive.")
    parser.add_argument("--backend", choices=("auto", "geotiff", "hdf5"), default="auto")
    parser.add_argument("--events-per-year", type=int, default=3)
    parser.add_argument("--minimum-days", type=int, default=7)
    parser.add_argument("--maximum-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def detect_backend(source: Path) -> str:
    """Detect whether the extracted dataset contains GeoTIFF or HDF5 events."""

    if any(source.glob("20[12][0-9]/*/*.tif")):
        return "geotiff"
    if any(source.glob("20[12][0-9]/*.hdf5")) or any(source.glob("20[12][0-9]/*.h5")):
        return "hdf5"
    raise FileNotFoundError(
        "Could not detect WSTS layout. Expected <root>/<year>/<event>/*.tif "
        "or <root>/<year>/<event>.hdf5."
    )


def discover_events(source: Path, backend: str) -> dict[int, list[Path]]:
    """Discover candidate complete events grouped by year."""

    result: dict[int, list[Path]] = {}
    for year in YEARS:
        year_root = source / str(year)
        if backend == "geotiff":
            events = [
                path
                for path in sorted(year_root.iterdir() if year_root.exists() else [])
                if path.is_dir() and any(path.glob("*.tif"))
            ]
        else:
            events = (
                sorted(year_root.glob("*.hdf5")) + sorted(year_root.glob("*.h5"))
                if year_root.exists()
                else []
            )
        result[year] = events
    return result


def event_days(path: Path, backend: str) -> int:
    """Return an event's number of daily observations."""

    if backend == "geotiff":
        return len(list(path.glob("*.tif")))
    import h5py

    with h5py.File(path, "r") as handle:
        return int(handle["data"].shape[0])


def choose_events(
    events: dict[int, list[Path]],
    backend: str,
    count: int,
    minimum_days: int,
    maximum_days: int,
    seed: int,
) -> dict[int, list[Path]]:
    """Select complete events with useful temporal lengths from each year."""

    rng = np.random.default_rng(seed)
    selected: dict[int, list[Path]] = {}
    for year, paths in events.items():
        suitable = [
            path for path in paths if minimum_days <= event_days(path, backend) <= maximum_days
        ]
        if len(suitable) < count:
            suitable = [path for path in paths if event_days(path, backend) >= minimum_days]
        if not suitable:
            raise RuntimeError(f"No {year} events have at least {minimum_days} days.")
        indices = rng.choice(len(suitable), size=min(count, len(suitable)), replace=False)
        selected[year] = [suitable[int(index)] for index in sorted(indices)]
    return selected


def finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    """Return compact finite-value statistics for one raster band."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"finite_count": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "finite_count": int(finite.size),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "mean": float(finite.mean()),
    }


def inspect_geotiff_event(event: Path) -> dict[str, Any]:
    """Inspect one raw GeoTIFF event and representative first/last days."""

    import rasterio

    files = sorted(event.glob("*.tif"))
    representatives = sorted({files[0], files[len(files) // 2], files[-1]})
    samples: list[dict[str, Any]] = []
    for path in representatives:
        with rasterio.open(path) as dataset:
            data = dataset.read()
            samples.append(
                {
                    "filename": path.name,
                    "shape": list(data.shape),
                    "dtypes": list(dataset.dtypes),
                    "crs": str(dataset.crs),
                    "transform": list(dataset.transform)[:6],
                    "descriptions": list(dataset.descriptions),
                    "tags": dataset.tags(),
                    "band_tags": {
                        str(index): dataset.tags(index)
                        for index in range(1, dataset.count + 1)
                        if dataset.tags(index)
                    },
                    "band_stats": {
                        CHANNEL_NAMES[index]: finite_stats(data[index])
                        for index in range(min(dataset.count, EXPECTED_CHANNELS))
                    },
                    "active_fire_positive_pixels": (
                        int(
                            np.count_nonzero(
                                np.nan_to_num(data[ACTIVE_FIRE_CHANNEL_ZERO_BASED], nan=0.0) > 0
                            )
                        )
                        if dataset.count > ACTIVE_FIRE_CHANNEL_ZERO_BASED
                        else None
                    ),
                }
            )
    return {
        "event_id": event.name,
        "days": len(files),
        "filenames": [path.name for path in files],
        "representative_samples": samples,
    }


def inspect_hdf5_event(event: Path) -> dict[str, Any]:
    """Inspect one converted HDF5 event and its metadata."""

    import h5py

    with h5py.File(event, "r") as handle:
        dataset = handle["data"]
        days = int(dataset.shape[0])
        indices = sorted({0, days // 2, days - 1})
        samples = []
        for index in indices:
            data = np.asarray(dataset[index])
            samples.append(
                {
                    "day_index": index,
                    "shape": list(data.shape),
                    "band_stats": {
                        CHANNEL_NAMES[band]: finite_stats(data[band])
                        for band in range(min(data.shape[0], EXPECTED_CHANNELS))
                    },
                    "active_fire_positive_pixels": (
                        int(
                            np.count_nonzero(
                                np.nan_to_num(data[ACTIVE_FIRE_CHANNEL_ZERO_BASED], nan=0.0) > 0
                            )
                        )
                        if data.shape[0] > ACTIVE_FIRE_CHANNEL_ZERO_BASED
                        else None
                    ),
                }
            )
        attrs = {key: json_safe(value) for key, value in dataset.attrs.items()}
        return {
            "event_id": event.stem,
            "days": days,
            "dataset_shape": list(dataset.shape),
            "dataset_dtype": str(dataset.dtype),
            "dataset_attributes": attrs,
            "representative_samples": samples,
        }


def json_safe(value: Any) -> Any:
    """Convert NumPy and byte values into JSON-safe values."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def copy_event(event: Path, target_year: Path, backend: str) -> None:
    """Copy one complete event while preserving the official layout."""

    target_year.mkdir(parents=True, exist_ok=True)
    if backend == "geotiff":
        shutil.copytree(event, target_year / event.name)
    else:
        shutil.copy2(event, target_year / event.name)


def sha256(path: Path) -> str:
    """Compute a SHA-256 digest for the final archive."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Create the selected subset, metadata manifest, and ZIP archive."""

    args = parse_args()
    backend = detect_backend(args.source) if args.backend == "auto" else args.backend
    selected = choose_events(
        discover_events(args.source, backend),
        backend,
        args.events_per_year,
        args.minimum_days,
        args.maximum_days,
        args.seed,
    )
    subset_root = args.output / "wsts_subset"
    if subset_root.exists():
        raise FileExistsError(
            f"{subset_root} already exists; remove or rename it before rerunning."
        )
    subset_root.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": backend,
        "source_root_name": args.source.name,
        "expected_channel_count": EXPECTED_CHANNELS,
        "active_fire_channel_zero_based": ACTIVE_FIRE_CHANNEL_ZERO_BASED,
        "channel_names": CHANNEL_NAMES,
        "selection": {
            "events_per_year": args.events_per_year,
            "minimum_days": args.minimum_days,
            "maximum_days": args.maximum_days,
            "seed": args.seed,
        },
        "events": {},
    }

    lengths: Counter[int] = Counter()
    for year, events in selected.items():
        year_reports = []
        for event in events:
            print(f"Copying and inspecting {year}/{event.name}")
            copy_event(event, subset_root / str(year), backend)
            report = (
                inspect_geotiff_event(event) if backend == "geotiff" else inspect_hdf5_event(event)
            )
            lengths[report["days"]] += 1
            year_reports.append(report)
        manifest["events"][str(year)] = year_reports
    manifest["event_length_histogram"] = dict(sorted(lengths.items()))

    manifest_path = subset_root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive_base = args.output / "wsts_subset"
    archive_path = Path(
        shutil.make_archive(str(archive_base), "zip", args.output, subset_root.name)
    )
    checksum = sha256(archive_path)
    (args.output / "wsts_subset.sha256.txt").write_text(
        f"{checksum}  {archive_path.name}\n", encoding="utf-8"
    )
    print(f"\nCreated: {archive_path}")
    print(f"Size: {archive_path.stat().st_size / (1024**3):.2f} GiB")
    print(f"SHA-256: {checksum}")


if __name__ == "__main__":
    main()
