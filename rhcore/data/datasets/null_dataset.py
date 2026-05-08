import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor

from rhcore.data.datasets import transforms

class NullDataset(Dataset):
    """
    Null dataset that returns empty dicts.
    Output: {'image': float32 tensor, 'label': long tensor}
    """

    def __init__(self, **dataset_config):
        super().__init__()

        self.dataset_size = dataset_config.get("dataset_size", 256)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Return an empty dict for null dataset
        return {}