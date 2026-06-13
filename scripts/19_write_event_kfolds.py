"""Write mutually exclusive event-level k-fold assignments."""

import argparse
import json
from pathlib import Path

from wildfire_alert.data.event_split import make_event_kfolds


def main() -> None:
    """Create and audit event-level k-fold split metadata."""

    parser = argparse.ArgumentParser(description="Write leakage-free event k-fold metadata.")
    parser.add_argument("--data-root", type=Path, default=Path("wsts_hdf5"))
    parser.add_argument("--backend", choices=("geotiff", "hdf5", "npz"), default="hdf5")
    parser.add_argument("--years", type=int, nargs="+", default=[2019])
    parser.add_argument("--event-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/splits/2019_event_3folds.json"))
    args = parser.parse_args()

    folds = make_event_kfolds(
        args.data_root,
        years=tuple(args.years),
        backend=args.backend,
        n_splits=args.event_folds,
        seed=args.seed,
    )
    validation_sets = [fold.validation_event_ids for fold in folds]
    overlaps: list[dict[str, object]] = []
    for left_index, left in enumerate(validation_sets):
        for right_index, right in enumerate(
            validation_sets[left_index + 1 :], start=left_index + 1
        ):
            overlap = sorted(left & right)
            if overlap:
                overlaps.append({"fold_a": left_index, "fold_b": right_index, "overlap": overlap})

    payload = {
        "years": list(args.years),
        "backend": args.backend,
        "event_folds": args.event_folds,
        "seed": args.seed,
        "has_validation_overlap": bool(overlaps),
        "validation_overlaps": overlaps,
        "folds": [
            {
                "fold": index,
                "train_event_count": len(fold.train_event_ids),
                "validation_event_count": len(fold.validation_event_ids),
                "train_event_ids": sorted(fold.train_event_ids),
                "validation_event_ids": sorted(fold.validation_event_ids),
            }
            for index, fold in enumerate(folds)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
