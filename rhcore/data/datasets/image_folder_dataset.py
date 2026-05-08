import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor

from rhcore.data.datasets import transforms

class ImageFolderDataset(Dataset):
    """
    ImageFolder dataset wrapper that returns dicts and runs a custom pipeline.
    Output: {'image': float32 tensor, 'label': long tensor}
    """

    def __init__(self, **dataset_config):
        super().__init__()

        folder_path = dataset_config.get("folder_path", None)  # single root
        data_prefix = dataset_config.get("data_prefix", "")    # optional subdir
        self.max_dataset_size = dataset_config.get("max_dataset_size", None)

        if folder_path is None:
            raise ValueError("dataset_config must contain 'folder_path' (single root directory).")

        root = os.path.join(folder_path, data_prefix)
        if not os.path.isdir(root):
            raise ValueError(f"Directory not found: {root}")

        # Build ImageFolder in init (no custom scanning code)
        # Note: ImageFolder expects class-subfolders under `root`.
        self.ds = ImageFolder(root=root, transform=None, target_transform=None)

        if len(self.ds) == 0:
            raise ValueError(f"No images found by ImageFolder under: {root}")

        pipeline_cfg = dataset_config.get("pipeline", [])
        self.transform_pipeline = self._build_pipeline(pipeline_cfg)

    def _build_pipeline(self, pipeline_cfg):
        """Build the data processing pipeline using getattr for dynamic class loading."""
        transforms_list = []
        for transform_cfg in pipeline_cfg:
            transform_cfg = transform_cfg.copy()
            transform_type = transform_cfg.pop("type")
            transform_class = getattr(transforms, transform_type)
            transforms_list.append(transform_class(**transform_cfg))
        return transforms_list

    def __len__(self):
        if self.max_dataset_size is not None:
            return min(len(self.ds), self.max_dataset_size)
        return len(self.ds)

    def __getitem__(self, idx):
        # ImageFolder returns (PIL_image, int_label) when transform=None
        image, label = self.ds[idx]

        if not torch.is_tensor(image):
            # Convert PIL/numpy/etc -> float tensor in [0,1]
            image = to_tensor(image)  # float32 by default
        else:
            image = image.to(dtype=torch.float32)

        # Wrap into dict for your pipeline
        data = {"image": image, "label": label}

        # Keep your pipeline behavior
        for transform in self.transform_pipeline:
            data = transform(data)

        # Enforce required keys
        if "image" not in data or "label" not in data:
            raise KeyError("Pipeline must output a dict with keys 'image' and 'label'.")

        lab = data["label"]
        lab = torch.as_tensor(lab, dtype=torch.long)

        return {"image": data["image"], "label": lab}