import os
import random
from pathlib import Path
from torch.utils.data import Dataset
from rhcore.data.datasets import transforms

class RandomImageDataset(Dataset):
    """Unpaired / randomly shuffled image dataset.

    - folder_paths: dict like {"A": "/path/to/domainA", "B": "/path/to/domainB", ...}
    - data_prefix: optional dict of subfolders per key
    - pipeline: same transform pipeline as in BasicImageDataset

    For each __getitem__, we sample ONE random image per key, independently.
    So (A, B, C, ...) are not matched by basename at all.
    """

    def __init__(self, **dataset_config):
        super().__init__()

        # Same config structure as your BasicImageDataset
        self.folder_paths = dataset_config.get('folder_paths', {})
        self.data_prefix = dataset_config.get('data_prefix', {})
        self.max_dataset_size = dataset_config.get('max_dataset_size', None)

        if not self.folder_paths:
            raise ValueError("No folder_paths specified")

        # For unpaired: store list of paths per key, no intersection
        self.paths_by_key = self._scan_images()

        pipeline_cfg = dataset_config.get('pipeline', [])
        self.transform_pipeline = self._build_pipeline(pipeline_cfg)

        # Decide how long an "epoch" is.
        # Here: length = max number of images in any domain, unless capped.
        base_size = max(len(v) for v in self.paths_by_key.values())
        if self.max_dataset_size is not None:
            base_size = min(base_size, self.max_dataset_size)
        self.dataset_size = base_size

    def _scan_images(self):
        """Scan images in each folder independently, no filename intersection."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        paths_by_key = {}

        for key, folder_path in self.folder_paths.items():
            path_prefix = os.path.join(folder_path, self.data_prefix.get(key, ''))
            image_paths = []

            for ext in extensions:
                folder_paths = list(Path(path_prefix).glob(f'**/*{ext}'))
                image_paths.extend([str(path) for path in folder_paths])

            if len(image_paths) == 0:
                raise ValueError(f'No images found for key "{key}" in {path_prefix}')

            # Sort first for reproducibility, then we’ll sample randomly in __getitem__
            image_paths = sorted(image_paths)
            paths_by_key[key] = image_paths
            print(f'Found {len(image_paths)} images for key "{key}" in {path_prefix}')

        return paths_by_key

    def _build_pipeline(self, pipeline_cfg):
        """Same dynamic transform loading as your original dataset."""
        transforms_list = []
        for transform_cfg in pipeline_cfg:
            cfg = transform_cfg.copy()
            transform_type = cfg.pop('type')
            transform_class = getattr(transforms, transform_type)
            transform = transform_class(**cfg)
            transforms_list.append(transform)
        return transforms_list

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # idx is only used to bound the epoch length; sampling per key is random.
        sample = {}

        for key, paths in self.paths_by_key.items():
            # Choose a random image for this key/domain
            rand_idx = random.randint(0, len(paths) - 1)
            sample[f"{key}_path"] = paths[rand_idx]

        # Apply the same transform pipeline as in your BasicImageDataset
        data = sample
        for transform in self.transform_pipeline:
            data = transform(data)

        return data
