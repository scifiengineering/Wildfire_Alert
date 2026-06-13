"""ResNet-18 U-Net wrapper for single-day and channel-concatenated multi-day inputs."""

from typing import Literal, cast

from torch import Tensor, nn


class SpreadUNet(nn.Module):
    """Predict one next-day burn logit per raster cell."""

    def __init__(
        self,
        channels_per_day: int,
        input_days: int = 1,
        encoder_name: str = "resnet18",
        encoder_weights: Literal["imagenet"] | None = "imagenet",
    ) -> None:
        super().__init__()
        if channels_per_day < 1 or input_days < 1:
            raise ValueError("channels_per_day and input_days must both be positive.")
        try:
            import segmentation_models_pytorch as smp  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install the 'ml' project extra to construct SpreadUNet.") from exc

        self.channels_per_day = channels_per_day
        self.input_days = input_days
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=channels_per_day * input_days,
            classes=1,
            activation=None,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return logits shaped ``(batch, 1, height, width)``."""

        if inputs.ndim == 5:
            batch, days, channels, height, width = inputs.shape
            if days != self.input_days or channels != self.channels_per_day:
                raise ValueError(
                    f"Expected ({self.input_days}, {self.channels_per_day}) day/channel axes; "
                    f"received ({days}, {channels})."
                )
            inputs = inputs.reshape(batch, days * channels, height, width)
        elif inputs.ndim == 4:
            expected_channels = self.input_days * self.channels_per_day
            if inputs.shape[1] != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} flattened channels; received {inputs.shape[1]}."
                )
        else:
            raise ValueError(f"Expected a four- or five-dimensional tensor; got {inputs.shape}.")
        return cast(Tensor, self.model(inputs))
