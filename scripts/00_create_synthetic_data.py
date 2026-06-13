"""Create a small synthetic WSTS-compatible dataset."""

import argparse
from pathlib import Path

from wildfire_alert.data.splits import write_official_folds
from wildfire_alert.data.synthetic import create_synthetic_dataset


def main() -> None:
    """Generate synthetic events and official split metadata."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/processed/synthetic"))
    parser.add_argument("--events-per-year", type=int, default=2)
    args = parser.parse_args()
    create_synthetic_dataset(args.output, events_per_year=args.events_per_year)
    write_official_folds(Path("data/splits/official_year_folds.json"))
    print(f"Created synthetic WSTS events under {args.output}")


if __name__ == "__main__":
    main()
