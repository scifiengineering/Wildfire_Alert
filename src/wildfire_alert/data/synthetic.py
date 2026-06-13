"""Generate tiny WSTS-compatible event files for local development and tests."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from wildfire_alert.data.channels import ACTIVE_FIRE_CHANNEL, N_CHANNELS


def create_synthetic_dataset(
    root: Path,
    years: tuple[int, ...] = (2018, 2019, 2020, 2021),
    events_per_year: int = 2,
    days: int = 7,
    size: int = 64,
    seed: int = 42,
) -> None:
    """Create deterministic synthetic event time series beneath ``root``."""

    if days < 2 or size < 8:
        raise ValueError("Synthetic events require at least two days and an 8x8 raster.")

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    for year in years:
        year_dir = root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        for event_number in range(events_per_year):
            data = rng.normal(size=(days, N_CHANNELS, size, size)).astype(np.float32)
            data[:, 7] = 270.0
            data[:, 13] = 90.0
            data[:, 19] = 270.0
            data[:, ACTIVE_FIRE_CHANNEL] = 0.0
            center_y = size // 2 + event_number
            center_x = size // 3
            for day_number in range(days):
                radius = 2 + day_number
                spread_x = center_x + day_number
                burning = (yy - center_y) ** 2 + (xx - spread_x) ** 2 <= radius**2
                data[day_number, ACTIVE_FIRE_CHANNEL, burning] = 1200.0

            first_date = date(year, 7, 1) + timedelta(days=event_number * 14)
            dates = np.asarray(
                [(first_date + timedelta(days=offset)).isoformat() for offset in range(days)]
            )
            np.savez_compressed(
                year_dir / f"synthetic_{event_number:03d}.npz", data=data, dates=dates
            )
