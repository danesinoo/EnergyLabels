from torchmetrics import Accuracy, Precision, Recall
from typing import Literal, Tuple, cast, Optional
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.segmentation import DiceScore
from torch.optim.optimizer import Optimizer
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor

import torch.nn.functional as F
import torch.nn as nn
import lightning as L
import torch


class BinaryClassifier(L.LightningModule):
    def __init__(
        self,
        model: DictConfig,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig],
        weight_initializer: DictConfig,
        pos_weight: float = 1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        assert 0 <= label_smoothing < 1, "Label smoothing must be in [0, 1)."
        self.save_hyperparameters(ignore="model")

        # MODEL
        self.model = cast(nn.Module, instantiate(model))
        self.model.apply(instantiate(weight_initializer))
        self.model.compile()

        # METRICS - Accuracy, Precision and Recall
        self.train_acc = Accuracy(task="binary")
        self.eval_acc = Accuracy(task="binary")
        self.train_prec = Precision(task="binary")
        self.eval_prec = Precision(task="binary")
        self.train_rec = Recall(task="binary")
        self.eval_rec = Recall(task="binary")

        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def smooth_labels(self, targets: Tensor):
        smoothing = self.hparams.label_smoothing
        return targets * (1 - smoothing) + 0.5 * smoothing

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        input, target = batch
        pred: Tensor = self.model.forward(input)
        loss = F.binary_cross_entropy_with_logits(
            pred,
            self.smooth_labels(target),
            pos_weight=self.pos_weight,
        )

        # Calculating metrics
        pred = pred.sigmoid()
        batch_acc = self.train_acc(pred, target)
        batch_prec = self.train_prec(pred, target)
        batch_rec = self.train_rec(pred, target)

        self.log_dict(
            {
                "train/batch_loss": loss,
                "train/batch_acc": batch_acc,
                "train/batch_prec": batch_prec,
                "train/batch_rec": batch_rec,
            },
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        return loss

    def on_train_epoch_end(self):
        training_set_acc = self.train_acc.compute()
        training_set_prec = self.train_prec.compute()
        training_set_rec = self.train_rec.compute()
        self.log_dict(
            {
                "train/epoch_acc": training_set_acc,
                "train/epoch_prec": training_set_prec,
                "train/epoch_rec": training_set_rec,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.train_acc.reset()
        self.train_prec.reset()
        self.train_rec.reset()

    def _shared_validation_logic(
        self, batch: Tuple[Tensor, Tensor], prefix: Literal["val", "test"]
    ):
        input, target = batch
        pred: Tensor = self.model.forward(input)
        loss = F.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=self.pos_weight,
        )

        # Calculating metrics
        pred = pred.sigmoid()
        self.eval_acc.update(pred, target)
        self.eval_prec.update(pred, target)
        self.eval_rec.update(pred, target)

        self.log(
            f"{prefix}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def _shared_evaluation_logic(self, prefix: Literal["val", "test"]):
        set_acc = self.eval_acc.compute()
        set_prec = self.eval_prec.compute()
        set_rec = self.eval_rec.compute()
        self.log_dict(
            {
                f"{prefix}/epoch_acc": set_acc,
                f"{prefix}/epoch_prec": set_prec,
                f"{prefix}/epoch_rec": set_rec,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.eval_acc.reset()
        self.eval_prec.reset()
        self.eval_rec.reset()

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self._shared_validation_logic(batch, "val")

    def on_validation_epoch_end(self):
        self._shared_evaluation_logic("val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self._shared_validation_logic(batch, "test")

    def on_test_epoch_end(self):
        self._shared_evaluation_logic("test")

    def configure_optimizers(self):
        optimizer = cast(
            Optimizer,
            instantiate(
                self.hparams.optimizer,
                params=self.parameters(),
            ),
        )

        if self.hparams.scheduler is None:
            return optimizer

        scheduler = cast(
            LRScheduler,
            instantiate(self.hparams.scheduler.scheduler, optimizer=optimizer),
        )

        scheduler_configs = {"scheduler": scheduler}
        monitor = self.hparams.scheduler.get("monitor")
        interval = self.hparams.scheduler.get("interval")
        frequency = self.hparams.scheduler.get("frequency")
        if monitor:
            scheduler_configs["monitor"] = monitor
        if interval:
            scheduler_configs["interval"] = interval
        if frequency:
            scheduler_configs["frequency"] = frequency

        return {"optimizer": optimizer, "lr_scheduler": scheduler_configs}


class MultiClassSegmentator(L.LightningModule):
    """
    Lightning Module to train image segmentators.
    """

    def __init__(
        self,
        model: DictConfig,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig],
        loss: DictConfig,
        weight_initializer: DictConfig,
        num_classes=4,
    ):
        """
        Note: this init method is designed to be used by the Hydra `instantiate` method.

        Args:
            model: The instantiable model configuration to be used for the training.
            optimizer: The instantiable optimizer configuration to be used for the training.
            scheduler: The instantiable (and optional) scheduler configuration to be used for the training.
            loss: The instantiable loss function configuration to be used for the training.
            weight_initializer: The instantiable weight initializer configuration to be used for the training.
            num_classes: The number of segmentation classes (comprehending the background class).
        """
        super().__init__()
        self.save_hyperparameters()

        # MODEL
        self.model = cast(nn.Module, instantiate(model))
        self.model.apply(instantiate(weight_initializer))
        self.model.compile()

        # METRICS - Dice Score (one to be used for training, and one for evaluation phases)
        self.train_dice = DiceScore(self.hparams.num_classes, True, "micro")
        self.eval_dice = DiceScore(self.hparams.num_classes, True, "micro")

        # LOSS FUNCTION
        self.loss = cast(nn.Module, instantiate(loss))

    def __prepare_labels_for_dice(self, label_mask: Tensor) -> Tensor:
        """
        Given the label segmentation tensor, it makes sure that the tensor has the appropriate shape.

        Args:
            label_mask: The label segmentation tensor. It can be

        Returns:
            The correctly shaped label segmentation tensor.
        """
        # Handling the 4-dims discrete labels scenario
        if label_mask.ndim == 4 and label_mask.shape[1] == 1:
            label_mask = label_mask.squeeze(1)
        # Handling the 3-dims discrete labels scenario
        if label_mask.ndim == 3:
            label_mask = F.one_hot(
                label_mask, num_classes=self.hparams.num_classes
            ).permute(0, 3, 1, 2)
        return label_mask.float()

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        input, target = batch
        pred: Tensor = self.model.forward(input)
        loss = self.loss(pred, target)

        # Calculating metrics
        batch_dice = self.train_dice(
            pred.sigmoid(), self.__prepare_labels_for_dice(target)
        )

        self.log_dict(
            {
                "train/batch_loss": loss,
                "train/batch_dice": batch_dice,
            },
            prog_bar=False,  # Put on false since everything is logged onto the WandB web app
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        return loss

    def on_train_epoch_end(self):
        training_dice = self.train_dice.compute()
        self.log_dict(
            {"train/epoch_dice": training_dice},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.train_dice.reset()

    def _shared_validation_logic(
        self, batch: Tuple[Tensor, Tensor], prefix: Literal["val", "test"]
    ):
        """
        Method shared between the Validation phase and the Test phase. The main purpose of this method is to accumulate
        metrics of the evaluation dataset of interest.
        """
        input, target = batch
        pred: Tensor = self.model.forward(input)
        loss = self.loss(pred, target)

        # Accumulating metrics
        self.eval_dice(pred.sigmoid(), self.__prepare_labels_for_dice(target))

        self.log(
            f"{prefix}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def _shared_evaluation_logic(self, prefix: Literal["val", "test"]):
        """
        Method shared between the Validation phase and the Test phase. The main purpose of this method is to compute the
        accumulated metrics of the evaluation dataset of interest.
        """
        set_dice = self.eval_dice.compute()
        self.log_dict(
            {
                f"{prefix}/epoch_dice": set_dice,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.eval_dice.reset()

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self._shared_validation_logic(batch, "val")

    def on_validation_epoch_end(self):
        self._shared_evaluation_logic("val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self._shared_validation_logic(batch, "test")

    def on_test_epoch_end(self):
        self._shared_evaluation_logic("test")

    def configure_optimizers(self):
        optimizer = cast(
            Optimizer,
            instantiate(
                self.hparams.optimizer,
                params=self.parameters(),
            ),
        )

        if self.hparams.scheduler is None:
            return optimizer

        scheduler = cast(
            LRScheduler,
            instantiate(self.hparams.scheduler.scheduler, optimizer=optimizer),
        )

        scheduler_configs = {"scheduler": scheduler}
        monitor = self.hparams.scheduler.get("monitor")
        interval = self.hparams.scheduler.get("interval")
        frequency = self.hparams.scheduler.get("frequency")
        if monitor:
            scheduler_configs["monitor"] = monitor
        if interval:
            scheduler_configs["interval"] = interval
        if frequency:
            scheduler_configs["frequency"] = frequency

        return {"optimizer": optimizer, "lr_scheduler": scheduler_configs}
