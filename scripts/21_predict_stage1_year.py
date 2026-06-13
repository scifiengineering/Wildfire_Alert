"""Run a frozen Stage 1 checkpoint on every eligible event-day in a year."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from wildfire_alert.stage1.artifacts import save_prediction_artifact
from wildfire_alert.stage1.dataset import Stage1TorchDataset, collate_stage1_samples
from wildfire_alert.stage1.unet import SpreadUNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 1 predictions for a full year.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("wsts_hdf5"))
    parser.add_argument("--backend", choices=("geotiff", "hdf5", "npz"), default="hdf5")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--input-days", type=int, default=5)
    parser.add_argument("--target-offset-days", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--exclude-empty-current-fire", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    threshold, top_fraction = read_calibration(args.calibration_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Stage1TorchDataset(
        args.data_root,
        years=(args.year,),
        input_days=args.input_days,
        target_offset_days=args.target_offset_days,
        backend=args.backend,
        crop_size=args.crop_size,
        crop_mode="center",
        exclude_empty_current_fire=args.exclude_empty_current_fire,
        seed=0,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"year={args.year} device={device} samples={len(dataset)} "
        f"threshold={threshold:.8g} top_fraction={top_fraction:.6g}"
    )
    records = predict(
        model,
        loader,
        args.output_dir,
        device,
        threshold=threshold,
        top_fraction=top_fraction,
    )
    event_count = len({str(record["event_id"]) for record in records})
    manifest = {
        "checkpoint": str(args.checkpoint),
        "calibration_json": str(args.calibration_json),
        "year": args.year,
        "backend": args.backend,
        "input_days": args.input_days,
        "target_offset_days": args.target_offset_days,
        "crop_size": args.crop_size,
        "exclude_empty_current_fire": args.exclude_empty_current_fire,
        "threshold": threshold,
        "top_fraction": top_fraction,
        "event_count": event_count,
        "sample_count": len(records),
        "records": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(records)} predictions to {args.output_dir}")


def read_calibration(path: Path) -> tuple[float, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return (
        float(report["best_threshold_by_dice"]["threshold"]),
        float(report["best_top_fraction_by_dice"]["top_fraction"]),
    )


def load_model(path: Path, input_days: int, device: torch.device) -> SpreadUNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SpreadUNet(channels_per_day=26, input_days=input_days, encoder_weights=None).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    output_dir: Path,
    device: torch.device,
    threshold: float,
    top_fraction: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for batch in loader:
        inputs, targets = batch_tensors(batch, device)
        event_ids = batch["event_ids"]
        target_dates = batch["target_dates"]
        if not isinstance(event_ids, list) or not isinstance(target_dates, list):
            raise TypeError("Collate function returned non-list metadata.")
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probabilities = torch.sigmoid(model(inputs)).detach().cpu().float()
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
        for index, (event_id, target_date) in enumerate(zip(event_ids, target_dates, strict=True)):
            path = output_dir / str(event_id) / f"{target_date}.npz"
            records.append(
                save_prediction_artifact(
                    path,
                    probabilities[index, 0].numpy().astype(np.float32),
                    (targets[index, 0].cpu().numpy() > 0).astype(np.uint8),
                    str(event_id),
                    str(target_date),
                    threshold,
                    top_fraction,
                )
            )
    return records


def batch_tensors(batch: dict[str, object], device: torch.device) -> tuple[Tensor, Tensor]:
    inputs, targets = batch["inputs"], batch["targets"]
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("Collate function returned non-tensor fields.")
    return (
        torch.nan_to_num(inputs.to(device=device, dtype=torch.float32)),
        torch.nan_to_num(targets.to(device=device, dtype=torch.float32)),
    )


if __name__ == "__main__":
    main()
