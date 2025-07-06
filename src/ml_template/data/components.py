"""
The purpose of this file is to group and build all the Dataset objects, which will then be passed to the Datamodule.
"""

from torch.utils.data import Dataset
from typing import List, Optional
from os.path import join
from typing import Tuple
from torch import Tensor

import torchvision.tv_tensors as TV
import torchvision.transforms.v2 as t
import nibabel as nib  # Install nibabel for nifti file segmentations
import pandas as pd
import torch


class AugmentImageData(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        transforms: Optional[t.Transform],
    ):
        self.dtst = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dtst)

    def __getitem__(self, index):
        image, label = self.dtst[index]
        image, label = self.transforms(image, label)
        return image, label


class TSLikeDataset(Dataset):
    def __init__(
        self,
        absolute_dataset_folder: str,
        csv_file_name: str,
        transforms: Optional[t.Transform] = None,
    ):
        self.df = pd.read_csv(join(absolute_dataset_folder, csv_file_name))
        self.absolute_dataset_folder = absolute_dataset_folder
        self.tensorizer = t.Compose(
            [t.ToImage(), t.ToDtype(torch.float32, scale=False)]
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def load_scan(self, abs_file_path: str) -> Tensor:
        try:
            image = nib.load(abs_file_path)
        except FileNotFoundError:
            print(f"Error: The file '{abs_file_path}' was not found.")
            raise
        except Exception as e:
            print(
                f"An error occurred while opening or processing the file '{abs_file_path}': {e}"
            )
            raise
        return self.tensorizer(image.get_fdata()).squeeze()

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        scan_fn, sub_fat_seg_fn, torso_fat_seg_fn, skeletal_seg_fn = self.df.iloc[index]
        scan = self.load_scan(join(self.absolute_dataset_folder, scan_fn)).unsqueeze(0)
        sub_fat_seg = self.load_scan(join(self.absolute_dataset_folder, sub_fat_seg_fn))
        torso_fat_seg = self.load_scan(
            join(self.absolute_dataset_folder, torso_fat_seg_fn)
        )
        skeletal_seg = self.load_scan(
            join(self.absolute_dataset_folder, skeletal_seg_fn)
        )

        # Calculating the background class
        all_foregrounds_stacked = torch.stack(
            [sub_fat_seg, torso_fat_seg, skeletal_seg], dim=0
        )
        is_any_foreground = torch.any(all_foregrounds_stacked > 0.5, dim=0)
        background_seg = (~is_any_foreground).float()

        all_classes = [
            background_seg,
            sub_fat_seg,
            torso_fat_seg,
            skeletal_seg,
        ]

        # Stack along a new dimension (channel dimension)
        # Each mask is (H,W,D), after unsqueeze(0) it's (1,H,W,D)
        # Then concat along dim 0 makes it (NumClasses, H,W,D)
        segmentations_multichannel = torch.stack(
            all_classes, dim=0
        )  # Shape: (4, H, W, D)

        # Apply argmax to get the final segmentation map (H, W, D)
        # Class labels will be: 0 (background), 1 (sub_fat), 2 (torso_fat), 3 (skeletal)
        segmentations = segmentations_multichannel.argmax(dim=0)
        if self.transforms:
            scan, segmentations = (
                self.transforms(scan),
                self.transforms(segmentations),
            )
        return TV.Image(scan), TV.Mask(segmentations.unsqueeze(0))


class DummyDtst(Dataset):
    def __init__(self, sample_size: List[int], dataset_size: int):
        super().__init__()
        self.sample_size = tuple(sample_size)
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return torch.rand(self.sample_size), torch.randint(0, 2, (1,)).float()
