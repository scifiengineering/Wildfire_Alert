"""Train Stage 1 U-Net on one official WSTS fold.

Supports mixed-precision training (torch.autocast + GradScaler), cosine LR warm
restarts, and optional Weights & Biases logging with an offline fallback.

Typical usage
-------------
Full training run (fold 0, 30 epochs, batch size 2, ImageNet encoder):
    python scripts/03_train_stage1.py --fold 0 --epochs 30 --batch-size 2 \\
        --encoder-weights imagenet --output-dir outputs/checkpoints/stage1

Smoke test (2 batches, no W&B):
    python scripts/03_train_stage1.py --fold 0 --epochs 1 \\
        --max-train-batches 2 --max-val-batches 2 --no-wandb
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, optim
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LRScheduler
from torch.utils.data import DataLoader

from wildfire_alert.data.event_split import make_event_kfold_split, make_event_split
from wildfire_alert.data.splits import official_year_folds
from wildfire_alert.evaluation.metrics import binary_segmentation_metrics
from wildfire_alert.stage1.dataset import Stage1TorchDataset, collate_stage1_samples
from wildfire_alert.stage1.losses import CombinedFocalDiceLoss
from wildfire_alert.stage1.unet import SpreadUNet
from wildfire_alert.utils.seeds import seed_everything


def main() -> None:
    """Parse arguments, build components, and run the training loop."""

    parser = argparse.ArgumentParser(description="Train Stage 1 U-Net on one WSTS fold.")
    parser.add_argument("--data-root", type=Path, default=Path("wsts_data"))
    parser.add_argument("--backend", choices=("geotiff", "hdf5", "npz"), default="geotiff")
    parser.add_argument("--split-mode", choices=("year", "event", "event_kfold"), default="year")
    parser.add_argument(
        "--years", type=int, nargs="+", help="Years to use with --split-mode event."
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--event-folds", type=int, default=3)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--input-days", type=int, default=5)
    parser.add_argument("--target-offset-days", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--lr-restart-epochs", type=int, default=10, help="T_0 for CosineAnnealingWarmRestarts."
    )
    parser.add_argument(
        "--scheduler",
        choices=("warm_restarts", "cosine", "none"),
        default="warm_restarts",
        help="Use cosine for reduced-data runs to avoid late LR restarts.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Cap training batches per epoch (debug only).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Cap validation batches per epoch (debug only).",
    )
    parser.add_argument("--encoder-weights", choices=("none", "imagenet"), default="imagenet")
    parser.add_argument("--exclude-empty-current-fire", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision training.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints/stage1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="wildfire-stage1")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("epochs must be at least one.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    split = _resolve_split(args)

    wandb_run = _init_wandb(args, split, use_amp)

    train_dataset = Stage1TorchDataset(
        args.data_root,
        years=split["train_years"],
        input_days=args.input_days,
        target_offset_days=args.target_offset_days,
        backend=args.backend,
        crop_size=args.crop_size,
        crop_mode="random",
        exclude_empty_current_fire=args.exclude_empty_current_fire,
        allowed_event_ids=split["train_event_ids"],
        seed=args.seed,
    )
    validation_dataset = Stage1TorchDataset(
        args.data_root,
        years=split["validation_years"],
        input_days=args.input_days,
        target_offset_days=args.target_offset_days,
        backend=args.backend,
        crop_size=args.crop_size,
        crop_mode="center",
        exclude_empty_current_fire=args.exclude_empty_current_fire,
        allowed_event_ids=split["validation_event_ids"],
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_stage1_samples,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_stage1_samples,
    )

    model = SpreadUNet(
        channels_per_day=26,
        input_days=args.input_days,
        encoder_weights=None if args.encoder_weights == "none" else "imagenet",
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_scheduler(args, optimizer)
    scaler: GradScaler | None = GradScaler() if use_amp else None
    loss_function = CombinedFocalDiceLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"split={args.split_mode}  fold={args.fold}  train={split['train_years']}"
        f"  val={split['validation_years']}  test={split['test_years']}"
        f"  device={device}  amp={use_amp}"
        f"  train_samples={len(train_dataset)}  val_samples={len(validation_dataset)}"
        f"  train_events={split['train_event_count']}  val_events={split['validation_event_count']}"
    )

    history: list[dict[str, Any]] = []
    best_ap = -1.0

    for epoch in range(args.epochs):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            scaler,
            device,
            use_amp,
            args.grad_clip,
            args.max_train_batches,
        )
        if scheduler is not None:
            if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step(epoch)
            else:
                scheduler.step()
        validation = _evaluate(
            model,
            validation_loader,
            loss_function,
            device,
            use_amp,
            args.max_val_batches,
        )

        record: dict[str, Any] = {
            "fold": args.fold,
            "split_mode": args.split_mode,
            "epoch": epoch,
            "train_years": list(split["train_years"]),
            "validation_years": list(split["validation_years"]),
            "test_years": list(split["test_years"]),
            "train_event_count": split["train_event_count"],
            "validation_event_count": split["validation_event_count"],
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in validation.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))

        if wandb_run is not None:
            wandb_run.log(record, step=epoch)

        val_ap = float(validation["average_precision"])
        validation_loss = float(validation["loss"])
        finite_record = _record_is_finite(record)
        if finite_record:
            latest_path = args.output_dir / f"stage1_fold_{args.fold}_latest.pt"
            _save_checkpoint(latest_path, model, optimizer, scheduler, record)
        else:
            print("  ! non-finite metrics detected; skipping latest checkpoint write")

        if finite_record and val_ap > best_ap and np.isfinite(validation_loss):
            best_ap = val_ap
            best_path = args.output_dir / f"stage1_fold_{args.fold}_best.pt"
            _save_checkpoint(best_path, model, optimizer, scheduler, record)
            print(f"  ^ new best AP {best_ap:.6f} -> {best_path}")
        elif not np.isfinite(validation_loss):
            print("  ! validation loss is non-finite; best checkpoint not updated")

    history_path = args.output_dir / f"history_fold_{args.fold}.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"Training complete. Best val AP: {best_ap:.6f}. History: {history_path}")

    if wandb_run is not None:
        wandb_run.finish()


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    loss_function: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: "GradScaler | None",
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    max_batches: int | None,
) -> float:
    """Run one training epoch; return mean batch loss."""

    model.train()
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets = _tensors_from_batch(batch, device)
        if not _finite_tensor_pair(inputs, targets):
            print(f"  ! skipping train batch {batch_index}: non-finite inputs/targets")
            continue
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(inputs)
            loss = loss_function(logits, targets)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            print(f"  ! skipping train batch {batch_index}: non-finite logits/loss")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        losses.append(float(loss.detach().cpu()))

    if not losses:
        raise RuntimeError("No training batches were consumed.")
    return float(np.mean(losses))


def _build_scheduler(
    args: argparse.Namespace,
    optimizer: optim.Optimizer,
) -> LRScheduler | None:
    """Create the requested learning-rate scheduler."""

    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs)
    return CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_restart_epochs, T_mult=1)


def _resolve_split(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve year-based or event-based train/validation split settings."""

    if args.split_mode == "year":
        fold = official_year_folds()[args.fold]
        return {
            "train_years": fold.train_years,
            "validation_years": (fold.validation_year,),
            "test_years": (fold.test_year,),
            "train_event_ids": None,
            "validation_event_ids": None,
            "train_event_count": None,
            "validation_event_count": None,
        }

    years = tuple(args.years or [])
    if not years:
        raise ValueError("--years is required when --split-mode event or event_kfold.")
    if args.split_mode == "event_kfold":
        event_split = make_event_kfold_split(
            args.data_root,
            years=years,
            backend=args.backend,
            fold=args.fold,
            n_splits=args.event_folds,
            seed=args.seed,
        )
    else:
        event_split = make_event_split(
            args.data_root,
            years=years,
            backend=args.backend,
            validation_fraction=args.val_fraction,
            seed=args.seed,
        )
    return {
        "train_years": years,
        "validation_years": years,
        "test_years": (),
        "train_event_ids": event_split.train_event_ids,
        "validation_event_ids": event_split.validation_event_ids,
        "train_event_count": len(event_split.train_event_ids),
        "validation_event_count": len(event_split.validation_event_ids),
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    loss_function: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None,
) -> dict[str, float]:
    """Evaluate on validation set; return metrics dict."""

    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets = _tensors_from_batch(batch, device)
        if not _finite_tensor_pair(inputs, targets):
            print(f"  ! skipping validation batch {batch_index}: non-finite inputs/targets")
            continue

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(inputs)
            loss = loss_function(logits, targets)
        if not torch.isfinite(logits).all():
            print(f"  ! skipping validation batch {batch_index}: non-finite logits")
            continue

        losses.append(float(loss.detach().cpu()))
        probability_tensor = torch.sigmoid(logits).detach().cpu().float()
        probability_tensor = torch.nan_to_num(probability_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        probabilities.append(probability_tensor.numpy().astype(np.float64))
        labels.append(targets.detach().cpu().float().numpy().astype(np.float64))

    if not losses:
        raise RuntimeError("No validation batches were consumed.")

    metrics = binary_segmentation_metrics(np.concatenate(probabilities), np.concatenate(labels))
    return {"loss": float(np.mean(losses)), **metrics.as_dict()}


def _tensors_from_batch(batch: dict[str, object], device: torch.device) -> tuple[Tensor, Tensor]:
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


def _finite_tensor_pair(inputs: Tensor, targets: Tensor) -> bool:
    """Return whether model inputs and targets are finite."""

    return bool(torch.isfinite(inputs).all() and torch.isfinite(targets).all())


def _record_is_finite(record: dict[str, Any]) -> bool:
    """Return whether all numeric metrics in a history record are finite."""

    for value in record.values():
        if isinstance(value, float) and not np.isfinite(value):
            return False
    return True


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler | None,
    record: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "record": record,
        },
        path,
    )


def _init_wandb(args: argparse.Namespace, split: dict[str, Any], use_amp: bool) -> Any:
    """Initialise W&B run; return run object or None on failure/disabled."""

    if args.no_wandb:
        return None
    try:
        import wandb  # type: ignore[import-not-found]

        run = wandb.init(
            project=args.wandb_project,
            name=f"fold{args.fold}_stage1",
            config={
                "fold": args.fold,
                "split_mode": args.split_mode,
                "train_years": list(split["train_years"]),
                "validation_years": list(split["validation_years"]),
                "test_years": list(split["test_years"]),
                "input_days": args.input_days,
                "backend": args.backend,
                "crop_size": args.crop_size,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "scheduler": args.scheduler,
                "lr_restart_epochs": args.lr_restart_epochs,
                "encoder_weights": args.encoder_weights,
                "amp": use_amp,
                "seed": args.seed,
            },
        )
        return run
    except Exception as exc:  # noqa: BLE001
        print(f"W&B init failed ({exc}); continuing without logging.")
        return None


if __name__ == "__main__":
    main()
