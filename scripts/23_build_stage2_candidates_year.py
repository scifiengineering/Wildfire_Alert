"""Build fair Stage 2 candidates for every eligible event-day in one year."""

import argparse
import csv
import json
from pathlib import Path

from wildfire_alert.stage1.dataset import Stage1TorchDataset
from wildfire_alert.stage2.candidates import feature_rows_for_sample, load_prediction_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-year Stage 2 candidates.")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("wsts_hdf5"))
    parser.add_argument("--backend", choices=("geotiff", "hdf5", "npz"), default="hdf5")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--input-days", type=int, default=5)
    parser.add_argument("--target-offset-days", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--exclude-empty-current-fire", action="store_true")
    parser.add_argument("--ring-radius-km", type=float, default=4.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    dataset = Stage1TorchDataset(
        args.data_root,
        years=(args.year,),
        input_days=args.input_days,
        target_offset_days=args.target_offset_days,
        backend=args.backend,
        crop_size=args.crop_size,
        crop_mode="center",
        circular_directions=False,
        exclude_empty_current_fire=args.exclude_empty_current_fire,
        seed=0,
    )
    event_ids: set[str] = set()
    candidate_count = 0
    positive_count = 0
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter | None = None
        for index in range(len(dataset)):
            sample = dataset[index]
            prediction_path = args.predictions_dir / sample.event_id / f"{sample.target_date}.npz"
            if not prediction_path.exists():
                raise FileNotFoundError(f"Missing Stage 1 prediction: {prediction_path}")
            event_ids.add(sample.event_id)
            rows = feature_rows_for_sample(
                sample,
                load_prediction_artifact(prediction_path),
                fold=-1,
                split=f"{args.year}_external_test",
                include_threshold_candidates=False,
                include_target_candidates=False,
                ring_radius_km=args.ring_radius_km,
            )
            if not rows:
                continue
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            candidate_count += len(rows)
            positive_count += sum(int(row["label"]) for row in rows)
    if candidate_count == 0:
        raise RuntimeError("No Stage 2 candidates were generated.")
    summary = {
        "year": args.year,
        "predictions_dir": str(args.predictions_dir),
        "event_count": len(event_ids),
        "sample_count": len(dataset),
        "candidate_count": candidate_count,
        "positive_count": positive_count,
        "positive_rate": positive_count / candidate_count,
        "ring_radius_km": args.ring_radius_km,
        "include_threshold_candidates": False,
        "include_target_candidates": False,
        "output_csv": str(args.output_csv),
    }
    summary_path = args.output_csv.with_name(f"{args.output_csv.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
