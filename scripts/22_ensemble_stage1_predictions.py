"""Average matching Stage 1 prediction rasters from multiple frozen folds."""

import argparse
import json
from pathlib import Path

from wildfire_alert.stage1.artifacts import ensemble_prediction_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble Stage 1 prediction directories.")
    parser.add_argument("--prediction-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if len(args.prediction_dirs) < 2:
        raise ValueError("Provide at least two --prediction-dirs.")
    relative_paths = prediction_paths(args.prediction_dirs[0])
    records: list[dict[str, object]] = []
    for relative_path in relative_paths:
        source_paths = tuple(directory / relative_path for directory in args.prediction_dirs)
        missing = [str(path) for path in source_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing ensemble inputs: {missing[:3]}")
        records.append(
            ensemble_prediction_artifacts(
                source_paths,
                args.output_dir / relative_path,
                threshold=args.threshold,
                top_fraction=args.top_fraction,
            )
        )

    expected = set(relative_paths)
    for directory in args.prediction_dirs[1:]:
        actual = set(prediction_paths(directory))
        if actual != expected:
            raise ValueError(
                f"Prediction path mismatch for {directory}: "
                f"missing={len(expected - actual)}, extra={len(actual - expected)}"
            )

    manifest = {
        "prediction_dirs": [str(path) for path in args.prediction_dirs],
        "threshold": args.threshold,
        "top_fraction": args.top_fraction,
        "ensemble_size": len(args.prediction_dirs),
        "sample_count": len(records),
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


def prediction_paths(directory: Path) -> list[Path]:
    return sorted(path.relative_to(directory) for path in directory.glob("*/*.npz"))


if __name__ == "__main__":
    main()
