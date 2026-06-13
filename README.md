# Wildfire Alert

Research pipeline for next-day wildfire spread prediction and discriminative per-location alerting
using WildfireSpreadTS (WSTS).

## Phase 0 status

The repository currently provides:

- configuration-driven paths and dataset options;
- the official 12-fold train/validation/test year assignments;
- a loader for raw WSTS GeoTIFF directories, per-event HDF5 files, and small NPZ fixtures;
- synthetic WSTS-compatible events for development without the complete 48.4 GB archive;
- physically consistent directional transforms;
- a batch visualization script and focused unit tests.

## Setup

Python 3.11 or newer and `uv` are recommended:

```powershell
uv sync --extra data --extra dev
uv run pytest
```

To include the later ML dependencies:

```powershell
uv sync --extra data --extra ml --extra dev
```

## Develop without the full dataset

Generate a small synthetic dataset with the same event/day/channel contract:

```powershell
uv run python scripts/00_create_synthetic_data.py --output data/processed/synthetic
uv run python scripts/00_visualize_batch.py --data-root data/processed/synthetic
```

The synthetic directory follows:

```text
data/processed/synthetic/<year>/<event_id>.npz
```

Each file stores `data` with shape `(days, 23, height, width)` and ISO-formatted `dates`.

## Connect real WSTS data

Raw GeoTIFF layout:

```text
<data_root>/<year>/<fire_name>/*.tif
```

HDF5 layout:

```text
<data_root>/<year>/<fire_name>.hdf5
```

Point `configs/data.yaml` at a local or mounted Google Drive directory. Drive-mounted data is useful
for inspecting a small subset; full training should use fast local or cloud-attached storage.

Inventory and validate a bounded number of real events without reading the full dataset:

```powershell
uv run python scripts/00_audit_data.py --data-root G:\WSTS --backend geotiff --check-events 4
```

## Dataset provenance

- Dataset DOI: <https://doi.org/10.5281/zenodo.8006177>
- Dataset version: `v1.0`
- Official experiment code: <https://github.com/SebastianGer/WildfireSpreadTS>

The original dataset creation code reported a wind-direction computation correction in February
2026. The downloaded dataset's wind channels must be audited before full training.

## Real subset observations

The first validated raw GeoTIFF subset uses 23 `float32` bands at 375 m resolution. Non-burning
pixels in the active-fire band are represented by `NaN`, while burning pixels contain detection-time
values. The preprocessing layer converts this final band to a binary mask before model ingestion.
Raster dimensions vary by event, so Stage 1 batching will require configured crops or padding.

## Stage 1 smoke test

The initial Stage 1 implementation concatenates temporal days along the channel axis and optionally
replaces the three degree-valued direction channels with sine/cosine pairs.

```powershell
.\.venv\Scripts\python.exe scripts\01_smoke_stage1.py
```

Run a bounded two-batch training smoke test on fold 0:

```powershell
.\.venv\Scripts\python.exe scripts\03_train_stage1.py `
  --fold 0 `
  --epochs 1 `
  --max-train-batches 2 `
  --max-val-batches 2 `
  --no-wandb
```

## Convert GeoTIFF to HDF5

The WSTS authors recommend per-event HDF5 files for faster training than raw GeoTIFF reads.
The fastest training layout is uncompressed HDF5, but it may require substantially more disk space.

```powershell
.\.venv\Scripts\python.exe scripts\02_convert_geotiff_to_hdf5.py `
  --data-dir wsts_data `
  --target-dir wsts_hdf5 `
  --compression none
```

Then train with:

```powershell
.\.venv\Scripts\python.exe scripts\03_train_stage1.py `
  --data-root wsts_hdf5 `
  --backend hdf5 `
  --fold 0 `
  --epochs 30 `
  --batch-size 2 `
  --encoder-weights imagenet `
  --output-dir outputs/checkpoints/stage1 `
  --no-wandb
```
