"""Build Stage 2 candidate feature tables from Stage 1 prediction artifacts."""

import argparse
import csv
import json
from pathlib import Path

from wildfire_alert.data.event_split import make_event_kfold_split, make_event_split
from wildfire_alert.stage1.dataset import Stage1TorchDataset
from wildfire_alert.stage2.candidates import feature_rows_for_sample, load_prediction_artifact


def main() -> None:
    """Extract Stage 2 candidate rows for one event split."""

    parser = argparse.ArgumentParser(description="Build Stage 2 candidate feature rows.")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("wsts_hdf5"))
    parser.add_argument("--backend", choices=("geotiff", "hdf5", "npz"), default="hdf5")
    parser.add_argument("--years", type=int, nargs="+", default=[2019])
    parser.add_argument("--split-mode", choices=("event", "event_kfold"), default="event_kfold")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--event-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--input-days", type=int, default=5)
    parser.add_argument("--target-offset-days", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--exclude-empty-current-fire", action="store_true")
    parser.add_argument("--ring-radius-km", type=float, default=5.0)
    parser.add_argument(
        "--include-threshold-candidates",
        action="store_true",
        help="Also include every calibrated threshold-mask pixel as a candidate.",
    )
    parser.add_argument(
        "--include-target-candidates",
        action="store_true",
        help="Also include true next-day burn pixels as candidates. Diagnostic only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage2/candidates"),
    )
    args = parser.parse_args()

    if args.split_mode == "event_kfold":
        split = make_event_kfold_split(
            args.data_root,
            years=tuple(args.years),
            backend=args.backend,
            fold=args.fold,
            n_splits=args.event_folds,
            seed=args.seed,
        )
    else:
        split = make_event_split(
            args.data_root,
            years=tuple(args.years),
            backend=args.backend,
            validation_fraction=args.val_fraction,
            seed=args.seed,
        )
    allowed_event_ids = (
        split.train_event_ids if args.split == "train" else split.validation_event_ids
    )
    dataset = Stage1TorchDataset(
        args.data_root,
        years=tuple(args.years),
        input_days=args.input_days,
        target_offset_days=args.target_offset_days,
        backend=args.backend,
        crop_size=args.crop_size,
        crop_mode="center",
        circular_directions=False,
        exclude_empty_current_fire=args.exclude_empty_current_fire,
        allowed_event_ids=allowed_event_ids,
        seed=args.seed,
    )

    rows: list[dict[str, int | float | str]] = []
    missing_predictions: list[str] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        prediction_path = args.predictions_dir / sample.event_id / f"{sample.target_date}.npz"
        if not prediction_path.exists():
            missing_predictions.append(str(prediction_path))
            continue
        artifact = load_prediction_artifact(prediction_path)
        rows.extend(
            feature_rows_for_sample(
                sample=sample,
                artifact=artifact,
                fold=args.fold,
                split=args.split,
                include_threshold_candidates=args.include_threshold_candidates,
                include_target_candidates=args.include_target_candidates,
                ring_radius_km=args.ring_radius_km,
            )
        )

    if missing_predictions:
        preview = "\n".join(missing_predictions[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_predictions)} prediction artifacts. First missing:\n{preview}"
        )
    if not rows:
        raise RuntimeError("No Stage 2 candidate rows were generated.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"stage2_fold{args.fold}_{args.split}_candidates.csv"
    write_csv(csv_path, rows)

    positive_count = sum(int(row["label"]) for row in rows)
    summary = {
        "fold": args.fold,
        "split": args.split,
        "split_mode": args.split_mode,
        "years": list(args.years),
        "seed": args.seed,
        "event_count": len(allowed_event_ids),
        "sample_count": len(dataset),
        "candidate_count": len(rows),
        "positive_count": positive_count,
        "positive_rate": positive_count / len(rows),
        "ring_radius_km": args.ring_radius_km,
        "include_threshold_candidates": args.include_threshold_candidates,
        "include_target_candidates": args.include_target_candidates,
        "csv_path": str(csv_path),
    }
    summary_path = args.output_dir / f"stage2_fold{args.fold}_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    """Write candidate rows as CSV."""

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
