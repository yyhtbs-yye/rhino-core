import torch
from torch.utils.data import Dataset

class ImageFolderNoiseGenerator(Dataset):
    """
    Synthetic dataset for pure generation.

    Output:
        {
            'noise': float32 tensor shaped (in_channels, H, W),
            'label': long tensor in [0, num_classes - 1],
        }
    """

    def __init__(self, **dataset_config):
        super().__init__()

        self.dataset_size = int(dataset_config.get("dataset_size", 256))

        input_size = dataset_config.get("input_size", None)
        in_channels = dataset_config.get("in_channels", None)
        num_classes = dataset_config.get("num_classes", None)

        if input_size is None:
            raise ValueError("dataset_config must contain 'input_size'.")
        if in_channels is None:
            raise ValueError("dataset_config must contain 'in_channels'.")
        if num_classes is None:
            raise ValueError("dataset_config must contain 'num_classes'.")

        if isinstance(input_size, int):
            self.height = input_size
            self.width = input_size
        elif isinstance(input_size, (list, tuple)) and len(input_size) == 2:
            self.height = int(input_size[0])
            self.width = int(input_size[1])
        else:
            raise ValueError("'input_size' must be an int or a 2-element list/tuple.")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)

        if self.dataset_size <= 0:
            raise ValueError("'dataset_size' must be > 0.")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("'input_size' values must be > 0.")
        if self.in_channels <= 0:
            raise ValueError("'in_channels' must be > 0.")
        if self.num_classes <= 0:
            raise ValueError("'num_classes' must be > 0.")

        self.data_config = {
            "input_size": input_size,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
        }

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        noise = torch.randn(self.in_channels, self.height, self.width, dtype=torch.float32)
        label = torch.randint(0, self.num_classes, size=(), dtype=torch.long)

        return {
            "noise": noise,
            "label": label,
        }
