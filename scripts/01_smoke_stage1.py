"""Run a real-data Stage 1 forward/loss/backward smoke test."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional

from wildfire_alert.data.preprocessing import circular_direction_channels, prepare_raw_inputs
from wildfire_alert.data.wsts_loader import WSTSDataset
from wildfire_alert.stage1.losses import CombinedFocalDiceLoss
from wildfire_alert.stage1.unet import SpreadUNet


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 32) -> torch.Tensor:
    """Pad the final two dimensions to a model-friendly multiple."""

    height, width = tensor.shape[-2:]
    padding_height = (-height) % multiple
    padding_width = (-width) % multiple
    return functional.pad(tensor, (0, padding_width, 0, padding_height))


def main() -> None:
    """Load one sample and verify a complete optimization step."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("wsts_subset/wsts_subset"))
    parser.add_argument("--backend", choices=("npz", "geotiff", "hdf5"), default="geotiff")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--input-days", type=int, default=5)
    args = parser.parse_args()

    dataset = WSTSDataset(
        args.data_root,
        years=(args.year,),
        input_days=args.input_days,
        backend=args.backend,
    )
    sample_index = next(
        (index for index in range(len(dataset)) if dataset[index].target.any()),
        0,
    )
    sample = dataset[sample_index]
    inputs = circular_direction_channels(prepare_raw_inputs(sample.inputs))
    input_tensor = pad_to_multiple(torch.from_numpy(inputs).unsqueeze(0))
    target_tensor = pad_to_multiple(
        torch.from_numpy(np.asarray(sample.target, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    )

    model = SpreadUNet(
        channels_per_day=inputs.shape[1],
        input_days=args.input_days,
        encoder_weights=None,
    )
    loss_function = CombinedFocalDiceLoss()
    logits = model(input_tensor)
    loss = loss_function(logits, target_tensor)
    loss.backward()
    print(
        {
            "event_id": sample.event_id,
            "target_date": sample.target_date,
            "input_shape": tuple(input_tensor.shape),
            "logit_shape": tuple(logits.shape),
            "positive_target_pixels": int(target_tensor.sum()),
            "loss": float(loss.detach()),
        }
    )


if __name__ == "__main__":
    main()
