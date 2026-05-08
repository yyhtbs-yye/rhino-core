import os
from pathlib import Path
from rhcore.data.datasets.basic_image_dataset import BasicImageDataset

class ImageWithPromptDataset(BasicImageDataset):
    """
    Extends BasicImageDataset by adding a `prompt` key for each sample.

    Matching rule:
        image:  xxx.png / xxx.jpg / xxx.jpeg / xxx.bmp
        prompt: xxx.txt

    Example:
        img_001.png  ->  img_001.txt
    """

    def __init__(self, **dataset_config):
        self.prompt_folder = dataset_config.get("prompt_folder", None)
        self.prompt_prefix = dataset_config.get("prompt_prefix", "")
        self.prompt_key = dataset_config.get("prompt_key", "prompt")
        self.prompt_extension = dataset_config.get("prompt_extension", ".txt")
        self.load_prompt_text = dataset_config.get("load_prompt_text", True)
        self.store_prompt_path = dataset_config.get("store_prompt_path", False)

        if self.prompt_folder is None:
            raise ValueError("`prompt_folder` must be provided.")

        super().__init__(**dataset_config)

        # Add prompt info into each sample after image matching is done
        self._attach_prompts()

    def _attach_prompts(self):
        prompt_root = Path(os.path.join(self.prompt_folder, self.prompt_prefix))

        if not prompt_root.exists():
            raise ValueError(f"Prompt folder does not exist: {prompt_root}")

        prompt_files = list(prompt_root.glob(f"**/*{self.prompt_extension}"))
        if len(prompt_files) == 0:
            raise ValueError(f"No prompt files found in {prompt_root}")

        # Map: stem -> prompt file path
        prompt_map = {}
        for prompt_path in prompt_files:
            stem = prompt_path.stem
            if stem in prompt_map:
                raise ValueError(
                    f"Duplicate prompt filename stem found: '{stem}' in {prompt_root}"
                )
            prompt_map[stem] = str(prompt_path)

        missing_prompts = []

        for image_base_name, sample in self.image_paths.items():
            # image_base_name is something like "abc.png"
            image_stem = Path(image_base_name).stem
            matched_prompt_path = prompt_map.get(image_stem, None)

            if matched_prompt_path is None:
                missing_prompts.append(image_base_name)
                continue

            if self.load_prompt_text:
                with open(matched_prompt_path, "r", encoding="utf-8") as f:
                    sample[self.prompt_key] = f.read().strip()
            else:
                sample[self.prompt_key] = matched_prompt_path

            if self.store_prompt_path:
                sample[f"{self.prompt_key}_path"] = matched_prompt_path

        if missing_prompts:
            preview = ", ".join(missing_prompts[:10])
            raise ValueError(
                f"Missing prompt files for {len(missing_prompts)} images. "
                f"Examples: {preview}"
            )