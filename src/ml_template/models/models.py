from typing import List, Tuple
from ml_template.models.components import ConvBlock
from ml_template.models.components import (
    LSConvBlockBN,
    RSConvBlockBN,
)
from torch import Tensor
from typing import cast
from torch import nn


class DummyModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.m = nn.Sequential(ConvBlock(3, 9), nn.Conv2d(9, 3, 3), nn.Flatten(1))

    def forward(self, x: Tensor):
        return cast(Tensor, self.m(x)).mean(1).unsqueeze(1)


class NormalizedUNet(nn.Module):
    """
    A copy of the vanilla UNet model presented here: https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_segmentation_masks: int = 4,
        magnitude: float = 1.0,
        n_levels: int = 4,
    ):
        """
        Args:
            input_channels: The number of channels of the input tensor.
            output_segmentation_masks: The number of classes to consider for segmentation (including the
                background class).
            magnitude: This coefficient multiplies the number of input and output channels in order to scale them accordingly.
                The magnitude of the official UNet corresponds to 1.
            n_levels: The number of levels that the UNet will have. The UNet presented in the original paper had 4 levels
        """
        assert (
            output_segmentation_masks < 7
        ), "For now it's just better to limit model capabilities, since the network isn't too big."
        assert (
            magnitude >= 0.0157
        ), "The magnitude value has to be at least of 0.0157, in order to make sure that the first layer hasn't 0 output channels."
        assert n_levels >= 1, "There has to be at least 1 level."
        super().__init__()

        # GENERATING NETWORK SETTINGS THROUGH THE magnitude AND n_levels PARAMETERS
        channels: List[int] = [
            int((2 ** (6 + i)) * magnitude) for i in range(n_levels + 1)
        ]
        ls_channels: List[Tuple[int, int]] = list(
            zip([input_channels] + channels[:-2], channels[:-1])
        )
        rs_channels: List[Tuple[int, int]] = list(
            zip(channels[::-1][:-1], channels[::-1][1:])
        )

        # BUILDING NETWORK ARCHITECTURE THROUGH THE PREVIOUSLY BUILT NETWORK SETTINGS
        self.ls_conv_blocks = nn.ModuleList(
            [LSConvBlockBN(in_ch, out_ch, 3) for in_ch, out_ch in ls_channels]
        )
        self.valley_conv = RSConvBlockBN(channels[-2], channels[-1], 3)
        self.rs_conv_blocks = nn.ModuleList(
            [
                RSConvBlockBN(in_ch, out_ch, 3, upscale=out_ch != channels[0])
                for in_ch, out_ch in rs_channels
            ]
        )
        self.prediction_conv = nn.Conv2d(channels[0], output_segmentation_masks, 1)

    def forward(self, x: Tensor):
        """
        WARNING: the model is slightly revised so that it doesn't lose 1 spatial dimension during
        each convolution. The Width and Height dimensions should be multiples of 8 in order for
        the model to work.
        """
        skips = []
        for ls_cv_b in self.ls_conv_blocks:
            skip, x = ls_cv_b(x)
            skips.append(skip)

        x = self.valley_conv(x)

        for skip, rs_cv_b in zip(skips[::-1], self.rs_conv_blocks):
            x = rs_cv_b(x, skip)

        return self.prediction_conv(x)
