"""Visualize one WSTS temporal sample and its next-day target."""

import argparse
from pathlib import Path

from wildfire_alert.data.wsts_loader import WSTSDataset


def main() -> None:
    """Render selected input channels and the active-fire masks."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install the 'data' extra to create visualizations.") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/synthetic"))
    parser.add_argument("--backend", choices=("npz", "geotiff", "hdf5"), default="npz")
    parser.add_argument("--input-days", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase0_batch.png"))
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Sample index to display. Defaults to the first sample with a positive target.",
    )
    args = parser.parse_args()

    dataset = WSTSDataset(
        args.data_root,
        years=(2018, 2019, 2020, 2021),
        input_days=args.input_days,
        backend=args.backend,
    )
    if not dataset:
        raise SystemExit("No samples found beneath the configured data root.")
    sample_index = args.sample_index
    if sample_index is None:
        sample_index = next(
            (index for index in range(len(dataset)) if dataset[index].target.any()),
            0,
        )
    sample = dataset[sample_index]
    final_day = sample.inputs[-1]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(final_day[3], cmap="YlGn")
    axes[0].set_title("NDVI")
    axes[1].imshow(final_day[6], cmap="viridis")
    axes[1].set_title("Wind speed")
    axes[2].imshow(final_day[22] > 0, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Current active fire")
    axes[3].imshow(sample.target, cmap="Reds", vmin=0, vmax=1)
    axes[3].set_title("Next-day target")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{sample.event_id}: {sample.input_dates[-1]} -> {sample.target_date}")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
