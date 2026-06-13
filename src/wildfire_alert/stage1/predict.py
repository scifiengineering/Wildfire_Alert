"""Stage 1 inference: load a checkpoint and generate per-event probability rasters.

For each event-day in the target years, runs the trained U-Net and saves a float32
probability map alongside the ground-truth mask as a compressed NPZ file.

Each saved .npz contains:
    ``probability``  — float32 array (H, W), values in [0, 1]
    ``target``       — uint8 array (H, W), binary ground-truth next-day mask
    ``event_id``     — str
    ``target_date``  — str (YYYY-MM-DD)

Usage
-----
    python scripts/04_predict_stage1.py \\
        --checkpoint outputs/checkpoints/stage1/stage1_fold_0_best.pt \\
        --fold 0 \\
        --split test \\
        --output-dir outputs/predictions/fold_0
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wildfire_alert.data.splits import Fold
from wildfire_alert.stage1.dataset import Stage1TorchDataset, collate_stage1_samples
from wildfire_alert.stage1.unet import SpreadUNet


@dataclass(frozen=True)
class PredictionRecord:
    """Paths and metadata for one saved prediction."""

    event_id: str
    target_date: str
    path: Path


def predict_fold(
    checkpoint_path: Path,
    data_root: Path,
    fold: Fold,
    split: str,
    output_dir: Path,
    batch_size: int = 4,
    num_workers: int = 2,
    input_days: int = 5,
    crop_size: int | None = None,
    device: torch.device | None = None,
) -> list[PredictionRecord]:
    """Run inference for one fold split and save probability rasters.

    Args:
        checkpoint_path: Path to a ``stage1_fold_N_best.pt`` checkpoint.
        data_root: Root of the WSTS data directory.
        fold: The Fold whose years determine which events are predicted.
        split: One of ``"train"``, ``"validation"``, or ``"test"``.
        output_dir: Directory where per-event NPZ files are written.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        input_days: Number of input days (must match checkpoint).
        crop_size: Crop applied during inference. ``None`` means no crop — the
            full raster is passed through. Use ``None`` for final predictions to
            preserve spatial resolution.
        device: Torch device. Defaults to CUDA if available, else CPU.

    Returns:
        List of ``PredictionRecord`` for each event-day processed.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"split must be 'train', 'validation', or 'test'; got '{split}'.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    years = _years_for_split(fold, split)
    model, channels_per_day = _load_model(checkpoint_path, input_days, device)

    dataset = Stage1TorchDataset(
        data_root,
        years=years,
        input_days=input_days,
        crop_size=crop_size,
        crop_mode="center",
        seed=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_stage1_samples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[PredictionRecord] = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"]
            targets = batch["targets"]
            event_ids = batch["event_ids"]
            target_dates = batch["target_dates"]

            if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                raise TypeError("Collate function returned non-tensor fields.")
            if not isinstance(event_ids, list) or not isinstance(target_dates, list):
                raise TypeError("Collate function returned non-list metadata.")

            inputs = inputs.to(device=device, dtype=torch.float32)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inputs)
            probs = torch.sigmoid(logits).detach().cpu().float().numpy()

            for i, (event_id, target_date) in enumerate(zip(event_ids, target_dates, strict=True)):
                prob_map = probs[i, 0]
                target_map = (targets[i, 0].cpu().numpy() > 0).astype(np.uint8)

                save_dir = output_dir / str(event_id)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{target_date}.npz"

                np.savez_compressed(
                    save_path,
                    probability=prob_map.astype(np.float32),
                    target=target_map,
                    event_id=np.bytes_(str(event_id)),
                    target_date=np.bytes_(str(target_date)),
                )
                records.append(
                    PredictionRecord(
                        event_id=str(event_id),
                        target_date=str(target_date),
                        path=save_path,
                    )
                )

    print(f"Saved {len(records)} predictions to {output_dir}")
    return records


def load_probability_raster(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved prediction NPZ; return (probability, target) arrays.

    Args:
        path: Path to a ``.npz`` written by :func:`predict_fold`.

    Returns:
        Tuple of ``(probability, target)`` as float32 and uint8 arrays respectively.
    """

    data = np.load(path)
    return data["probability"], data["target"]


def _years_for_split(fold: Fold, split: str) -> tuple[int, ...]:
    if split == "train":
        return fold.train_years
    if split == "validation":
        return (fold.validation_year,)
    return (fold.test_year,)


def _load_model(
    checkpoint_path: Path,
    input_days: int,
    device: torch.device,
) -> tuple[SpreadUNet, int]:
    """Load SpreadUNet weights from a training checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    channels_per_day = 26
    model = SpreadUNet(
        channels_per_day=channels_per_day,
        input_days=input_days,
        encoder_weights=None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, channels_per_day
