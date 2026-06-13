"""Canonical channel metadata for raw WildfireSpreadTS daily rasters."""

from dataclasses import dataclass

WSTS_CHANNEL_NAMES: tuple[str, ...] = (
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

DIRECTIONAL_CHANNELS: tuple[int, ...] = (7, 13, 19)
ACTIVE_FIRE_CHANNEL = 22
N_CHANNELS = 23


@dataclass(frozen=True)
class ChannelMetadata:
    """Describes the canonical raw WSTS channels."""

    names: tuple[str, ...] = WSTS_CHANNEL_NAMES
    directional_indices: tuple[int, ...] = DIRECTIONAL_CHANNELS
    active_fire_index: int = ACTIVE_FIRE_CHANNEL
