"""Calibrate Stage 1 alert thresholds on validation predictions."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from wildfire_alert.data.event_split import make_event_kfold_split, make_event_split
from wildfire_alert.evaluation.metrics import binary_segmentation_metrics
from wildfire_alert.stage1.dataset import Stage1TorchDataset, collate_stage1_samples
from wildfire_alert.stage1.unet import SpreadUNet

DEFAULT_TOP_FRACTIONS = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01)


def main() -> None:
    """Run threshold and top-risk calibration for one Stage 1 checkpoint."""

    parser = argparse.ArgumentParser(
        description="Calibrate Stage 1 segmentation thresholds on validation samples."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("wsts_hdf5"))
    parser.add_argument("--backend", choices=("geotiff", "hdf5", "npz"), default="hdf5")
    parser.add_argument("--split-mode", choices=("event", "event_kfold"), default="event_kfold")
    parser.add_argument("--years", type=int, nargs="+", default=[2019])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--event-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--input-days", type=int, default=5)
    parser.add_argument("--target-offset-days", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--threshold-count", type=int, default=200)
    parser.add_argument("--top-fractions", type=float, nargs="+", default=DEFAULT_TOP_FRACTIONS)
    parser.add_argument("--exclude-empty-current-fire", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/calibration"),
    )
    args = parser.parse_args()

    if args.threshold_count < 2:
        raise ValueError("--threshold-count must be at least 2.")

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
    dataset = Stage1TorchDataset(
        args.data_root,
        years=tuple(args.years),
        input_days=args.input_days,
        target_offset_days=args.target_offset_days,
        backend=args.backend,
        crop_size=args.crop_size,
        crop_mode="center",
        exclude_empty_current_fire=args.exclude_empty_current_fire,
        allowed_event_ids=split.validation_event_ids,
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
    model = load_model(args.checkpoint, input_days=args.input_days, device=device)

    print(
        f"calibrating fold={args.fold} years={tuple(args.years)} device={device} "
        f"split_mode={args.split_mode} val_samples={len(dataset)} "
        f"val_events={len(split.validation_event_ids)}"
    )
    probabilities, labels = collect_validation_predictions(
        model=model,
        loader=loader,
        device=device,
        max_batches=args.max_val_batches,
    )
    report = build_calibration_report(
        probabilities=probabilities,
        labels=labels,
        checkpoint=args.checkpoint,
        args=args,
        validation_event_count=len(split.validation_event_ids),
        validation_sample_count=len(dataset),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"stage1_fold{args.fold}_thresholds.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    best_threshold = report["best_threshold_by_dice"]
    best_top_fraction = report["best_top_fraction_by_dice"]
    print(
        "best threshold "
        f"{best_threshold['threshold']:.6g}: dice={best_threshold['dice']:.6f} "
        f"precision={best_threshold['precision']:.6f} recall={best_threshold['recall']:.6f}"
    )
    print(
        "best top fraction "
        f"{best_top_fraction['top_fraction']:.6g}: dice={best_top_fraction['dice']:.6f} "
        f"precision={best_top_fraction['precision']:.6f} recall={best_top_fraction['recall']:.6f}"
    )
    print(f"saved calibration report to {output_path}")


def load_model(checkpoint_path: Path, input_days: int, device: torch.device) -> SpreadUNet:
    """Load a trained Stage 1 U-Net from a checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SpreadUNet(channels_per_day=26, input_days=input_days, encoder_weights=None).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_validation_predictions(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    max_batches: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect flattened validation probabilities and labels."""

    model.eval()
    probability_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets = tensors_from_batch(batch, device)
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            print(f"  ! skipping validation batch {batch_index}: non-finite inputs/targets")
            continue
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
        if not torch.isfinite(logits).all():
            print(f"  ! skipping validation batch {batch_index}: non-finite logits")
            continue

        probabilities = torch.sigmoid(logits).detach().cpu().float()
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
        probability_chunks.append(probabilities.numpy().reshape(-1).astype(np.float64))
        label_chunks.append(targets.detach().cpu().float().numpy().reshape(-1).astype(np.float64))

    if not probability_chunks:
        raise RuntimeError("No validation batches were consumed.")
    return np.concatenate(probability_chunks), np.concatenate(label_chunks)


def tensors_from_batch(batch: dict[str, object], device: torch.device) -> tuple[Tensor, Tensor]:
    """Move collated Stage 1 tensors to the requested device."""

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


def build_calibration_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    checkpoint: Path,
    args: argparse.Namespace,
    validation_event_count: int,
    validation_sample_count: int,
) -> dict[str, Any]:
    """Build a JSON-serializable calibration report."""

    labels = (labels > 0).astype(np.float64)
    threshold_values = candidate_thresholds(probabilities, args.threshold_count)
    threshold_sweep = [
        metric_record(
            binary_segmentation_metrics(probabilities, labels, threshold=float(threshold)),
            threshold=float(threshold),
        )
        for threshold in threshold_values
    ]
    top_fraction_sweep = [
        top_fraction_record(probabilities, labels, top_fraction)
        for top_fraction in args.top_fractions
    ]
    base_metrics = binary_segmentation_metrics(probabilities, labels, threshold=0.5)

    return {
        "checkpoint": str(checkpoint),
        "fold": args.fold,
        "split_mode": args.split_mode,
        "years": list(args.years),
        "seed": args.seed,
        "backend": args.backend,
        "input_days": args.input_days,
        "crop_size": args.crop_size,
        "validation_event_count": validation_event_count,
        "validation_sample_count": validation_sample_count,
        "pixel_count": int(labels.size),
        "positive_pixel_count": int(labels.sum()),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()),
        "metrics_at_0_5": base_metrics.as_dict(),
        "best_threshold_by_dice": max(threshold_sweep, key=lambda item: item["dice"]),
        "best_threshold_by_iou": max(threshold_sweep, key=lambda item: item["iou"]),
        "best_top_fraction_by_dice": max(top_fraction_sweep, key=lambda item: item["dice"]),
        "best_top_fraction_by_iou": max(top_fraction_sweep, key=lambda item: item["iou"]),
        "threshold_sweep": threshold_sweep,
        "top_fraction_sweep": top_fraction_sweep,
    }


def candidate_thresholds(probabilities: np.ndarray, threshold_count: int) -> np.ndarray:
    """Return threshold candidates concentrated over the observed score range."""

    finite_scores = probabilities[np.isfinite(probabilities)]
    if finite_scores.size == 0:
        raise ValueError("No finite probabilities available for threshold calibration.")
    quantiles = np.linspace(0.0, 1.0, threshold_count)
    observed = np.quantile(finite_scores, quantiles)
    high_tail = np.quantile(finite_scores, np.linspace(0.95, 1.0, threshold_count))
    candidates = np.concatenate(
        [
            np.array([0.0, 0.5, 1.0], dtype=np.float64),
            observed,
            high_tail,
            np.array([float(finite_scores.max()) + 1e-12], dtype=np.float64),
        ]
    )
    return np.unique(np.clip(candidates, 0.0, 1.0))


def metric_record(
    metrics: Any,
    threshold: float,
) -> dict[str, float]:
    """Return a metric dict with the threshold included."""

    return {"threshold": threshold, **metrics.as_dict()}


def top_fraction_record(
    probabilities: np.ndarray,
    labels: np.ndarray,
    top_fraction: float,
) -> dict[str, float]:
    """Compute metrics for a global top-risk pixel budget."""

    if not 0 < top_fraction < 1:
        raise ValueError("top fractions must be between zero and one.")
    flat = probabilities.reshape(-1)
    count = max(1, int(round(flat.size * top_fraction)))
    threshold = float(np.partition(flat, -count)[-count])
    metrics = binary_segmentation_metrics(probabilities, labels, threshold=threshold)
    record = metric_record(metrics, threshold=threshold)
    record["top_fraction"] = float(top_fraction)
    record["selected_pixel_count"] = float(np.count_nonzero(flat >= threshold))
    return record


if __name__ == "__main__":
    main()
