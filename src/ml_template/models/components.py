from typing import Optional
from torch import Tensor
from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.m = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor):
        return self.m(x)


class LSConvBlockBN(nn.Module):
    """
    Left-side Convolutional Block of the UNet architecture, presented here: https://arxiv.org/abs/1505.04597
    A Batch Normalization layer has been added to make the training more stable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.downscale = nn.MaxPool2d(2, 2)

    def forward(self, x: Tensor):
        x = self.m(x)
        downscaled_x = self.downscale(x)
        return x, downscaled_x


class RSConvBlockBN(nn.Module):
    """
    Right-side Convolutional Block of the UNet architecture, presented here: https://arxiv.org/abs/1505.04597
    A Batch Normalization layer has been added to make the training more stable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        upscale=True,
    ):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.upscaler = nn.ConvTranspose2d(out_channels, out_channels // 2, 2, 2)
        self.upscale = upscale

    def forward(self, x: Tensor, skip: Optional[Tensor] = None):
        if skip is not None:
            x = torch.concat([x, skip], 1)
        x = self.m(x)
        if self.upscale:
            x = self.upscaler(x)
        return x
