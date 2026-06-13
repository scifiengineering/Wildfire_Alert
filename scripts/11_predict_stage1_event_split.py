"""Generate calibrated Stage 1 predictions for a 2019 event split."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from wildfire_alert.data.event_split import make_event_kfold_split, make_event_split
from wildfire_alert.stage1.dataset import Stage1TorchDataset, collate_stage1_samples
from wildfire_alert.stage1.unet import SpreadUNet


def main() -> None:
    """Run event-split inference and save per-sample NPZ predictions."""

    parser = argparse.ArgumentParser(
        description="Generate Stage 1 predictions for a train/validation event split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path)
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--top-fraction", type=float)
    parser.add_argument("--exclude-empty-current-fire", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions/stage1_2019_event"),
    )
    args = parser.parse_args()

    threshold, top_fraction = resolve_operating_point(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        exclude_empty_current_fire=args.exclude_empty_current_fire,
        allowed_event_ids=allowed_event_ids,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_stage1_samples,
    )
    model = load_model(args.checkpoint, args.input_days, device)
    output_dir = args.output_dir / f"fold_{args.fold}" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"predicting fold={args.fold} split={args.split} years={tuple(args.years)} "
        f"device={device} samples={len(dataset)} events={len(allowed_event_ids)} "
        f"threshold={threshold:.8g} top_fraction={top_fraction:.6g}"
    )
    records = write_predictions(
        model=model,
        loader=loader,
        output_dir=output_dir,
        device=device,
        threshold=threshold,
        top_fraction=top_fraction,
    )
    manifest = {
        "checkpoint": str(args.checkpoint),
        "calibration_json": str(args.calibration_json) if args.calibration_json else None,
        "fold": args.fold,
        "split": args.split,
        "split_mode": args.split_mode,
        "years": list(args.years),
        "seed": args.seed,
        "backend": args.backend,
        "input_days": args.input_days,
        "crop_size": args.crop_size,
        "threshold": threshold,
        "top_fraction": top_fraction,
        "event_count": len(allowed_event_ids),
        "sample_count": len(records),
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(records)} predictions to {output_dir}")
    print(f"saved manifest to {manifest_path}")


def resolve_operating_point(args: argparse.Namespace) -> tuple[float, float]:
    """Resolve threshold and top fraction from args or calibration JSON."""

    threshold = args.threshold
    top_fraction = args.top_fraction
    if args.calibration_json is not None:
        report = json.loads(args.calibration_json.read_text(encoding="utf-8"))
        if threshold is None:
            threshold = float(report["best_threshold_by_dice"]["threshold"])
        if top_fraction is None:
            top_fraction = float(report["best_top_fraction_by_dice"]["top_fraction"])

    if threshold is None:
        raise ValueError("Provide --threshold or --calibration-json.")
    if top_fraction is None:
        top_fraction = 0.0005
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one.")
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between zero and one.")
    return float(threshold), float(top_fraction)


def load_model(checkpoint_path: Path, input_days: int, device: torch.device) -> SpreadUNet:
    """Load a Stage 1 checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SpreadUNet(channels_per_day=26, input_days=input_days, encoder_weights=None).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def write_predictions(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    output_dir: Path,
    device: torch.device,
    threshold: float,
    top_fraction: float,
) -> list[dict[str, object]]:
    """Write NPZ prediction artifacts and return manifest records."""

    records: list[dict[str, object]] = []
    for batch in loader:
        inputs, targets = tensors_from_batch(batch, device)
        event_ids = batch["event_ids"]
        target_dates = batch["target_dates"]
        if not isinstance(event_ids, list) or not isinstance(target_dates, list):
            raise TypeError("Collate function returned non-list metadata.")

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
        probabilities = torch.sigmoid(logits).detach().cpu().float()
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)

        for index, (event_id, target_date) in enumerate(zip(event_ids, target_dates, strict=True)):
            probability = probabilities[index, 0].numpy().astype(np.float32)
            target = (targets[index, 0].detach().cpu().numpy() > 0).astype(np.uint8)
            threshold_mask = (probability >= threshold).astype(np.uint8)
            top_mask, top_threshold = top_fraction_mask(probability, top_fraction)
            save_dir = output_dir / str(event_id)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{target_date}.npz"

            np.savez_compressed(
                save_path,
                probability=probability,
                target=target,
                threshold_mask=threshold_mask,
                top_mask=top_mask,
                event_id=np.bytes_(str(event_id)),
                target_date=np.bytes_(str(target_date)),
                threshold=np.float32(threshold),
                top_fraction=np.float32(top_fraction),
                top_threshold=np.float32(top_threshold),
            )
            records.append(
                {
                    "event_id": str(event_id),
                    "target_date": str(target_date),
                    "path": str(save_path),
                    "probability_max": float(probability.max()),
                    "target_positive_pixels": int(target.sum()),
                    "threshold_positive_pixels": int(threshold_mask.sum()),
                    "top_positive_pixels": int(top_mask.sum()),
                    "top_threshold": float(top_threshold),
                }
            )
    return records


def tensors_from_batch(batch: dict[str, object], device: torch.device) -> tuple[Tensor, Tensor]:
    """Move collated tensors to the requested device."""

    inputs, targets = batch["inputs"], batch["targets"]
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("Collate function returned non-tensor fields.")
    return (
        torch.nan_to_num(
            inputs.to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        ),
        torch.nan_to_num(
            targets.to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        ),
    )


def top_fraction_mask(probability: np.ndarray, top_fraction: float) -> tuple[np.ndarray, float]:
    """Return a binary mask for the top fraction of pixels in one sample."""

    flat = probability.reshape(-1)
    count = max(1, int(round(flat.size * top_fraction)))
    selected = np.argpartition(flat, -count)[-count:]
    mask = np.zeros(flat.shape, dtype=np.uint8)
    mask[selected] = 1
    threshold = float(flat[selected].min())
    return mask.reshape(probability.shape), threshold


if __name__ == "__main__":
    main()
