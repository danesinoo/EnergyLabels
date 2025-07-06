from torchvision.transforms import functional as f
from torchvision.transforms import v2 as t
from typing import Tuple, List
from torch import Tensor

import torchvision.tv_tensors as TV
import random
import torch


class Clip:
    def __init__(self, lower_bound: float, upper_bound: float):
        self.lb = lower_bound
        self.ub = upper_bound

    def __call__(self, *args: Tuple[Tensor]) -> Tuple[Tensor]:
        tens: List[Tensor] = []
        for t in args:
            if isinstance(t, TV.Image):
                t = t.clip(self.lb, self.ub)
            tens.append(t)
        return tuple(tens)


class RandomRotate90:
    def __init__(self):
        self.angles = [90, 180, 270]

    def __call__(self, *args: Tuple[Tensor]) -> Tuple[Tensor]:
        tens: List[Tensor] = []
        choosen_angle_index = random.randint(0, len(self.angles) - 1)
        for t in args:
            tens.append(f.rotate(t, self.angles[choosen_angle_index]))
        return tuple(tens)


class RandomElasticTransform:
    def __init__(self, p=0.5, alpha=50, sigma=5):
        self.transform = t.ElasticTransform(alpha, sigma)
        self.p = p

    def __call__(self, *args: Tuple[Tensor]) -> Tuple[Tensor]:
        apply_transform = random.random() < self.p
        return self.transform(args) if apply_transform else args


class RandomGaussianNoise:
    def __init__(self, p=0.5, mean=0, sigma=0.005, clip=False):
        self.transform = t.GaussianNoise(mean=mean, sigma=sigma, clip=clip)
        self.p = p

    def __call__(self, *args: Tuple[Tensor]) -> Tuple[Tensor]:
        apply_transform = random.random() < self.p
        return self.transform(args) if apply_transform else args


class ChannelDivider:
    def __init__(self, range_points: List[Tuple[int, int]], include_original=True):
        assert len(range_points) > 0, "You must provide at least one threshold point."
        self.rp = range_points
        self.include_original = include_original

    def divide_channels(self, x: Tensor) -> Tensor:
        extracted_channels_list = []
        for lower_b, upper_b in self.rp:
            extracted_channels_list.append(x.clip(lower_b, upper_b))
        if self.include_original:
            extracted_channels_list.append(x)
        return torch.concat(extracted_channels_list, dim=0)

    def __call__(self, *args):
        tens: List[Tensor] = []
        for t in args:
            if not isinstance(t, TV.Mask):
                t = self.divide_channels(t)
            tens.append(t)
        return tuple(tens)
