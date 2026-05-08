import os
from pathlib import Path

from safetensors.torch import load_file

from rhcore.data.datasets.basic_image_dataset import BasicImageDataset


class ImageWithTokenEmbeddingDataset(BasicImageDataset):
    """
    Attach precomputed prompt embeddings from .safetensors files.

    Matching rule:
        image:      xxx.png / xxx.jpg / xxx.jpeg / xxx.bmp
        embedding:  xxx.safetensors

    Each safetensors file must contain:
        - embeds
        - pooled_embeds

    Assumption:
        - embeds are already padded / fixed-length
        - no length tracking or extra padding is needed
    """

    def __init__(self, **dataset_config):
        self.embeds_folder = dataset_config.get("embeds_folder")
        self.embeds_extension = dataset_config.get("embeds_extension", ".safetensors")

        self.embeds_key = dataset_config.get("embeds_key", "prompt_embeds")
        self.pooled_embeds_key = dataset_config.get("pooled_embeds_key", "pooled_prompt_embeds")

        self.store_embeds_path = dataset_config.get("store_embeds_path", False)
        self.embedding_device = dataset_config.get("embedding_device", "cpu")
        self.embedding_dtype = dataset_config.get("embedding_dtype", None)

        if self.embeds_folder is None:
            raise ValueError("`embeds_folder` must be provided.")

        super().__init__(**dataset_config)
        self._attach_embeddings()

    def _attach_embeddings(self):
        embeds_root = Path(self.embeds_folder)
        if not embeds_root.exists():
            raise ValueError(f"Embeds folder does not exist: {embeds_root}")

        embeds_files = list(embeds_root.glob(f"**/*{self.embeds_extension}"))
        if not embeds_files:
            raise ValueError(f"No embedding files found in {embeds_root}")

        embeds_map = {}
        for path in embeds_files:
            stem = path.stem
            if stem in embeds_map:
                raise ValueError(f"Duplicate embedding filename stem found: '{stem}'")
            embeds_map[stem] = str(path)

        missing = []
        for image_base_name, sample in self.image_paths.items():
            image_stem = Path(image_base_name).stem
            embeds_path = embeds_map.get(image_stem)

            if embeds_path is None:
                missing.append(image_base_name)
                continue

            embeds, pooled_embeds = self._load_embedding_tensors(embeds_path)
            sample[self.embeds_key] = embeds
            sample[self.pooled_embeds_key] = pooled_embeds

            if self.store_embeds_path:
                sample[f"{self.embeds_key}_path"] = embeds_path

        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"Missing embedding files for {len(missing)} images. Examples: {preview}"
            )

    def _load_embedding_tensors(self, path: str):
        tensors = load_file(path, device=self.embedding_device)

        if "prompt_embeds" not in tensors:
            raise KeyError(f"'prompt_embeds' not found in {path}")
        if "pooled_prompt_embeds" not in tensors:
            raise KeyError(f"'pooled_prompt_embeds' not found in {path}")

        embeds = tensors["prompt_embeds"]
        pooled_embeds = tensors["pooled_prompt_embeds"]

        # Optional: squeeze leading batch dim if saved as [1, L, D] / [1, D]
        if embeds.ndim == 3 and embeds.shape[0] == 1:
            embeds = embeds.squeeze(0)
        if pooled_embeds.ndim == 2 and pooled_embeds.shape[0] == 1:
            pooled_embeds = pooled_embeds.squeeze(0)

        if self.embedding_dtype is not None:
            embeds = embeds.to(self.embedding_dtype)
            pooled_embeds = pooled_embeds.to(self.embedding_dtype)

        return embeds.contiguous(), pooled_embeds.contiguous()
