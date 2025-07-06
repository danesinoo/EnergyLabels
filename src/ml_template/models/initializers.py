from torch.nn import init
from torch import nn
from abc import ABC, abstractmethod

import torch


class Initializer(ABC):
    """
    Initiliazier abstract class whose main purpose is to provide an interface to implement many intiliazers that relies
    on a random number generator to make experiments reproducible.
    """

    def __init__(self, seed=42):
        super().__init__()
        self.gen = torch.Generator().manual_seed(seed)

    @abstractmethod
    def __call__(self, m: nn.Module):
        """
        Weight initialization logic that relies on a random number generator, configured by the seed provided during
        object initialization.
        """
        pass


class XavierInitializer(Initializer):
    def __init__(self, seed=42):
        super().__init__(seed)

    def __call__(self, m: nn.Module):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight, generator=self.gen)
