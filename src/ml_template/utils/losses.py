from typing import Optional
from torch import Tensor

import torch.nn.functional as F
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for image segmentation.
    """

    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth: The smoothing factor to avoid divide-by-zero operations, and to better up numerical stability.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Args:
            y_pred: The output tensor of the model, with a shape of [N, C, W, H]. It is expected to be in raw logits.
            y_true: The label tensor coming from the dataset. It can have a shape of [N, W, H] (in case of discrete
                indexes for the various classes) or [N, C, W, H] (in case of one-hot class encodings).

        Returns:
            A differentiable scalar value representing the dice loss.
        """

        # Handling the discrete labels scenario
        if y_true.ndim == 3:
            y_true = (
                F.one_hot(y_true, num_classes=y_pred.shape[1])
                .permute(0, 3, 1, 2)
                .float()
            )

        # Empirically prooven that the sigmoid allows for a more stable training and better results
        y_pred = y_pred.sigmoid()
        intersection = (y_pred * y_true).sum(dim=(-2, -1))  # Sum over H, W
        union = y_pred.sum(dim=(-2, -1)) + y_true.sum(dim=(-2, -1))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()  # Average over the batch and channels
        return loss


class SegmentationLoss(nn.Module):
    """
    A combination of the Cross Entropy and Dice loss functions, for segmentation tasks.
    """

    def __init__(self, smooth: Optional[float], alpha=0.5):
        """
        Args:
            smooth: The smoothing factor to avoid divide-by-zero operations, and to better up numerical stability.
            alpha: The weight between the Cross Entropy and the Dice loss functions.
        """
        super(SegmentationLoss, self).__init__()
        self.smooth = smooth
        self.cross_en = nn.CrossEntropyLoss()
        self.dice_l = DiceLoss(smooth)
        self.alpha = alpha

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Args:
            y_pred: The output tensor of the model, with a shape of [N, C, W, H]. It is expected to be in raw logits.
            y_true: The label tensor coming from the dataset. It can have a shape of [N, W, H] (in case of discrete
                indexes for the various classes) or [N, C, W, H] (in case of one-hot class encodings).

        Returns:
            A differentiable scalar value representing the combination of the cross entropy and dice loss.
        """

        # Handling the case of discrete indexes but using four dimensions (IN CASE OF BINARY SEGMENTATION, TWO CHANNELS
        # SHOULD BE USED)
        if y_true.ndim == 4 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)

        return self.alpha * self.cross_en(y_pred, y_true.long()) + (
            1 - self.alpha
        ) * self.dice_l(y_pred, y_true)
