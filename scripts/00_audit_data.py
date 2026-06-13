"""Inventory and validate a local or mounted WSTS dataset."""

import argparse
import json
from pathlib import Path

from wildfire_alert.data.audit import audit_dataset
from wildfire_alert.data.wsts_loader import WSTSDataset


def main() -> None:
    """Print and optionally save a bounded dataset audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("npz", "geotiff", "hdf5"), required=True)
    parser.add_argument("--input-days", type=int, default=5)
    parser.add_argument("--check-events", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = WSTSDataset(
        args.data_root,
        years=(2018, 2019, 2020, 2021),
        input_days=args.input_days,
        backend=args.backend,
    )
    report = audit_dataset(dataset, check_events=args.check_events)
    serialized = json.dumps(report.as_dict(), indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
