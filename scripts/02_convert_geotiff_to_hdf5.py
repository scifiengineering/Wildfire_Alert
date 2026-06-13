"""Convert raw WSTS GeoTIFF event directories into per-event HDF5 files."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from wildfire_alert.data.channels import N_CHANNELS

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def main() -> None:
    """Convert complete GeoTIFF fire events into resumable HDF5 event files."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("wsts_data"))
    parser.add_argument("--target-dir", type=Path, default=Path("wsts_hdf5"))
    parser.add_argument("--years", type=int, nargs="+", default=[2018, 2019, 2020, 2021])
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="none",
        help="Use 'none' for fastest training reads; gzip saves space but is slower.",
    )
    parser.add_argument("--max-events", type=int, help="Debug cap per entire conversion run.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    events = discover_geotiff_events(args.data_dir, tuple(args.years))
    if args.max_events is not None:
        events = events[: args.max_events]
    if not events:
        raise SystemExit(f"No GeoTIFF events found under {args.data_dir}.")

    args.target_dir.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        (args.target_dir / str(year)).mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    for year, event_dir in tqdm(events, desc="Converting WSTS events"):
        target_path = args.target_dir / str(year) / f"{event_dir.name}.hdf5"
        if target_path.exists() and not args.overwrite:
            skipped += 1
            continue
        convert_event(
            event_dir,
            target_path,
            year=year,
            compression=None if args.compression == "none" else args.compression,
        )
        converted += 1

    print(
        json.dumps(
            {
                "source": str(args.data_dir),
                "target": str(args.target_dir),
                "events_seen": len(events),
                "converted": converted,
                "skipped": skipped,
                "compression": args.compression,
            },
            indent=2,
        )
    )


def discover_geotiff_events(data_dir: Path, years: tuple[int, ...]) -> list[tuple[int, Path]]:
    """Return sorted ``(year, event_dir)`` pairs for raw WSTS GeoTIFF events."""

    events: list[tuple[int, Path]] = []
    for year in years:
        year_dir = data_dir / str(year)
        if not year_dir.exists():
            continue
        for event_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
            if any(event_dir.glob("*.tif")):
                events.append((year, event_dir))
    return events


def convert_event(
    event_dir: Path,
    target_path: Path,
    year: int,
    compression: str | None,
) -> None:
    """Convert one complete event directory to one HDF5 file."""

    import h5py
    import rasterio  # type: ignore[import-untyped]

    tif_paths = tuple(sorted(event_dir.glob("*.tif")))
    if not tif_paths:
        raise ValueError(f"No GeoTIFF files found in {event_dir}.")

    daily_arrays: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    metadata: dict[str, Any] | None = None
    descriptions: tuple[str | None, ...] = ()
    for index, tif_path in enumerate(tif_paths):
        with rasterio.open(tif_path) as dataset:
            array = dataset.read().astype(np.float32)
            if array.shape[0] != N_CHANNELS:
                raise ValueError(f"{tif_path} has {array.shape[0]} bands; expected {N_CHANNELS}.")
            daily_arrays.append(array)
            if index == 0:
                metadata = {
                    "crs": str(dataset.crs),
                    "transform": tuple(dataset.transform)[:6],
                    "height": dataset.height,
                    "width": dataset.width,
                    "dtypes": dataset.dtypes,
                    "tags": dataset.tags(),
                }
                descriptions = dataset.descriptions

    imgs = np.stack(daily_arrays)
    dates = np.asarray([path.stem for path in tif_paths], dtype="S10")
    temp_path = target_path.with_suffix(".hdf5.tmp")
    if temp_path.exists():
        temp_path.unlink()

    with h5py.File(temp_path, "w") as handle:
        dataset = handle.create_dataset(
            "data",
            data=imgs,
            compression=compression,
            chunks=(1, N_CHANNELS, imgs.shape[2], imgs.shape[3]),
        )
        dataset.attrs["year"] = year
        dataset.attrs["fire_name"] = event_dir.name
        dataset.attrs["img_dates"] = dates
        dataset.attrs["channel_descriptions"] = np.asarray(
            [value or "" for value in descriptions],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        if metadata is not None:
            dataset.attrs["crs"] = metadata["crs"]
            dataset.attrs["transform"] = metadata["transform"]
            dataset.attrs["height"] = metadata["height"]
            dataset.attrs["width"] = metadata["width"]
            dataset.attrs["dtypes"] = np.asarray(
                metadata["dtypes"],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            dataset.attrs["tags_json"] = json.dumps(metadata["tags"], sort_keys=True)

    temp_path.replace(target_path)


if __name__ == "__main__":
    main()
