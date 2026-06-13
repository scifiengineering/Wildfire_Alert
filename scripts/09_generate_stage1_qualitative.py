"""Generate qualitative Stage 1 prediction figures for event-split checkpoints."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from wildfire_alert.data.event_split import make_event_kfold_split, make_event_split
from wildfire_alert.stage1.dataset import Stage1BatchSample, Stage1TorchDataset
from wildfire_alert.stage1.unet import SpreadUNet


def main() -> None:
    """Load a Stage 1 checkpoint and save side-by-side validation examples."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("wsts_hdf5"))
    parser.add_argument("--backend", choices=("geotiff", "hdf5", "npz"), default="hdf5")
    parser.add_argument("--years", type=int, nargs="+", default=[2019])
    parser.add_argument("--split-mode", choices=("event", "event_kfold"), default="event_kfold")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--event-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--fold-label", type=str, required=True)
    parser.add_argument("--input-days", type=int, default=5)
    parser.add_argument("--target-offset-days", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-fraction", type=float, default=0.001)
    parser.add_argument("--max-figures", type=int, default=8)
    parser.add_argument(
        "--selection",
        choices=("first_positive", "target_positive_max_probability"),
        default="first_positive",
        help="How to choose validation samples for qualitative figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/stage1_qualitative"),
    )
    args = parser.parse_args()

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
        backend=args.backend,
        input_days=args.input_days,
        target_offset_days=args.target_offset_days,
        crop_size=args.crop_size,
        crop_mode="center",
        allowed_event_ids=split.validation_event_ids,
        seed=args.seed,
    )
    model = load_model(args.checkpoint, input_days=args.input_days, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for sample, probability in selected_examples(dataset, model, device, args.selection):
        save_figure(
            sample=sample,
            probability=probability,
            threshold=args.threshold,
            top_fraction=args.top_fraction,
            output_path=args.output_dir
            / f"{args.fold_label}_{saved:02d}_{sample.event_id}_{sample.target_date}.png",
        )
        saved += 1
        if saved >= args.max_figures:
            break

    if saved == 0:
        raise RuntimeError(
            "No positive-target validation samples were found for qualitative figures."
        )
    print(f"Saved {saved} qualitative figures to {args.output_dir}")


def selected_examples(
    dataset: Stage1TorchDataset,
    model: torch.nn.Module,
    device: torch.device,
    selection: str,
) -> list[tuple[Stage1BatchSample, np.ndarray]]:
    """Return positive-target validation examples for plotting."""

    examples: list[tuple[Stage1BatchSample, np.ndarray]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample.target.sum().item() <= 0:
            continue
        probability = predict_probability(model, sample.inputs, device)
        examples.append((sample, probability))

    if selection == "first_positive":
        return examples
    if selection == "target_positive_max_probability":
        return sorted(examples, key=lambda item: float(item[1].max()), reverse=True)
    raise ValueError(f"Unknown selection mode: {selection}")


def load_model(checkpoint_path: Path, input_days: int, device: torch.device) -> SpreadUNet:
    """Load a trained Stage 1 U-Net from a checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SpreadUNet(channels_per_day=26, input_days=input_days, encoder_weights=None).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_probability(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Predict a probability map for one tensorized sample."""

    batch = inputs.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(batch)
    probability = torch.sigmoid(logits)[0, 0].detach().cpu().float()
    probability = torch.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)
    return probability.numpy()


def save_figure(
    sample: Stage1BatchSample,
    probability: np.ndarray,
    threshold: float,
    top_fraction: float,
    output_path: Path,
) -> None:
    """Save one four-panel qualitative prediction figure."""

    inputs = sample.inputs
    target = sample.target.squeeze(0).numpy()
    event_id = sample.event_id
    target_date = sample.target_date
    ndvi = inputs[-1, 3].numpy()
    current_fire = inputs[-1, 25].numpy()
    predicted_mask = probability >= threshold
    top_mask = top_fraction_mask(probability, top_fraction)
    vmax = max(float(np.quantile(probability, 0.995)), float(probability.max()), 1e-6)

    figure, axes = plt.subplots(1, 6, figsize=(21, 4))
    axes[0].imshow(ndvi, cmap="YlGn")
    axes[0].set_title("NDVI")
    axes[1].imshow(current_fire, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Current Fire")
    im = axes[2].imshow(probability, cmap="magma", vmin=0, vmax=vmax)
    axes[2].set_title(f"Probability\nmax={probability.max():.2e}")
    axes[3].imshow(predicted_mask, cmap="Reds", vmin=0, vmax=1)
    axes[3].set_title(f"Prediction >= {threshold}")
    axes[4].imshow(top_mask, cmap="Reds", vmin=0, vmax=1)
    axes[4].set_title(f"Top {top_fraction:.2%} Risk")
    axes[5].imshow(target, cmap="Reds", vmin=0, vmax=1)
    axes[5].set_title("Next-Day Target")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    figure.suptitle(f"{event_id} -> {target_date}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def top_fraction_mask(probability: np.ndarray, top_fraction: float) -> np.ndarray:
    """Return a binary mask of the highest-risk pixels."""

    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between zero and one.")
    flat = probability.reshape(-1)
    count = max(1, int(round(flat.size * top_fraction)))
    selected = np.argpartition(flat, -count)[-count:]
    mask = np.zeros(flat.shape, dtype=bool)
    mask[selected] = True
    return mask.reshape(probability.shape)


if __name__ == "__main__":
    main()
