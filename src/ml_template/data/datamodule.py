from torch.utils.data import random_split, DataLoader, Dataset
from ml_template.data.components import AugmentImageData
from torchvision.transforms.v2 import Transform
from typing import Optional, Tuple, cast
from hydra.utils import instantiate
from omegaconf import DictConfig

import lightning as L
import torch


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        train_val_test: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size=32,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        train_data_augs_config: Optional[DictConfig] = None,
        eval_data_augs_config: Optional[DictConfig] = None,
        random_seed=42,
    ):
        super().__init__()
        assert batch_size >= 1, "The batch size should be of at least 1 sample."
        assert (
            abs(sum(train_val_test) - 1.0) < 1e-6
        ), "The percentages of the train/val/test partitions must add up to 1."
        self.save_hyperparameters()

        self.dtst: Optional[Dataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.train_data_augs: Optional[Transform] = None
        self.eval_data_augs: Optional[Transform] = None

    def setup(self, stage):
        # Dataset creation
        if self.dtst is None:
            self.dtst = cast(Dataset, instantiate(self.hparams.dataset))

        # Dataset splitting
        if self.data_train is None and self.data_val is None and self.data_test is None:
            generator = torch.Generator().manual_seed(self.hparams.random_seed)
            self.data_train, self.data_val, self.data_test = random_split(
                self.dtst, self.hparams.train_val_test, generator=generator
            )

        # Dataset augmentation
        if self.train_data_augs is None and self.hparams.train_data_augs_config:
            self.train_data_augs = cast(
                Transform, instantiate(self.hparams.train_data_augs_config)
            )
        if self.eval_data_augs is None and self.hparams.eval_data_augs_config:
            self.eval_data_augs = cast(
                Transform, instantiate(self.hparams.eval_data_augs_config)
            )
        if stage in ("fit", "validate", None):
            self.val_set = AugmentImageData(
                dataset=self.data_val, transforms=self.eval_data_augs
            )
        if stage in ("fit", None):
            self.train_set = AugmentImageData(
                dataset=self.data_train,
                transforms=self.train_data_augs,
            )
        if stage in ("test", None):
            self.test_set = AugmentImageData(
                dataset=self.data_test,
                transforms=self.eval_data_augs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )
