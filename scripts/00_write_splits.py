"""Write the official WSTS 12-fold year assignments."""

from pathlib import Path

from wildfire_alert.data.splits import write_official_folds


def main() -> None:
    """Write fold definitions to the standard data directory."""

    output = Path("data/splits/official_year_folds.json")
    write_official_folds(output)
    print(f"Wrote official folds to {output}")


if __name__ == "__main__":
    main()
