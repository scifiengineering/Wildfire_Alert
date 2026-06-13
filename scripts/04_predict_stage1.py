"""Generate Stage 1 probability rasters for one fold split.

Loads the best checkpoint for the given fold, runs inference over the requested
split (train / validation / test), and saves per-event NPZ files.

Usage
-----
    python scripts/04_predict_stage1.py --fold 0 --split test
    python scripts/04_predict_stage1.py --fold 0 --split test --no-crop
"""

import argparse
from pathlib import Path

import torch

from wildfire_alert.data.splits import official_year_folds
from wildfire_alert.stage1.predict import predict_fold


def main() -> None:
    """Parse arguments and run Stage 1 inference."""

    parser = argparse.ArgumentParser(description="Generate Stage 1 probability rasters.")
    parser.add_argument("--data-root", type=Path, default=Path("wsts_data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints/stage1"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-days", type=int, default=5)
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Pass full raster through (recommended for final predictions).",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    args = parser.parse_args()

    fold = official_year_folds()[args.fold]
    checkpoint_path = args.checkpoint_dir / f"stage1_fold_{args.fold}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = args.output_dir / f"fold_{args.fold}" / args.split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_size = None if args.no_crop else args.crop_size

    print(f"fold={args.fold}  split={args.split}  device={device}")
    print(f"checkpoint={checkpoint_path}")
    print(f"output_dir={output_dir}")

    records = predict_fold(
        checkpoint_path=checkpoint_path,
        data_root=args.data_root,
        fold=fold,
        split=args.split,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_days=args.input_days,
        crop_size=crop_size,
        device=device,
    )
    print(f"Done. {len(records)} predictions written to {output_dir}")


if __name__ == "__main__":
    main()
